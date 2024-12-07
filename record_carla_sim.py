from utils import *
import glob
import os
import sys
import argparse
import time
from datetime import datetime
import random
import numpy as np
from matplotlib import cm
import open3d as o3d
from queue import Queue
from queue import Empty
from utils import *
from collections import deque

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main(arg):

    # Load simulation configuration from YAML file
    config = load_simulation_config('./simulation_config.yml')
    sensor_coords = load_simulation_config('./sensor_coords.yml')[config['ego_vehicle']]

    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    # set traffic manager
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    
    # random seed
    tm.set_random_device_seed(config['seed'])
    random.seed(config['seed'])

    world = client.load_world(config['map'])
    
    spectator = world.get_spectator()
    
    original_settings = world.get_settings()
    settings = world.get_settings()

    sensor_name_list = []
    sensor_obj_list = []
    
    try:

        settings.fixed_delta_seconds = config['fixed_delta_seconds']
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        ego_vehicle = world.try_spawn_actor(blueprint_library.find(config['ego_vehicle']), random.choice(spawn_points)) #spawn_points[2]) # 
        ego_vehicle.set_autopilot(config['ego_autopilot'], tm_port)
        
        spectator.set_transform(ego_vehicle.get_transform())
        update_sensor_location(world, spectator, ego_vehicle)
        
        # Load and spawn the traffic vehicles, configure traffic manager parameters
        for i in range(config['traffic_agents']):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
            while vehicle_bp.id in config['exiled_vehicles']:
                vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
            vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            while vehicle is None:
                vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if vehicle is not None:
                vehicle.set_autopilot(True, tm_port)
                for attr, value in config['traffic_manager'].items():
                    method = getattr(tm, attr)
                    method(vehicle, value)
               
        ##################################################################################################################################################

        # create semantic lidar sensor blueprint
        lidar_properties = config['sensors']['sensor.lidar.ray_cast_semantic']
        print(lidar_properties)
        lidar_bp, lidar_save_path = generate_lidar_bp(arg, world, blueprint_library, lidar_properties, config['fixed_delta_seconds'])

        # spawn semantic lidar sensor at a specific locations
        _x,_y,_z, _roll, _pitch, _yaw = sensor_coords['s'+str(lidar_properties['location'])]
        lidar_transform = carla.Transform(carla.Location(x=_x, y=_y, z=_z))
        lidar_transform.rotation = carla.Rotation(_pitch, _yaw, _roll)
        lidar = world.spawn_actor(lidar_bp, lidar_transform) #, attach_to=ego_vehicle) # dont attach to vehicle TODO: update position in loop

        ##################################################################################################################################################

        # create point cloud object to store semantic lidar data and visualize
        dynamic_actors_idx_queue = deque(maxlen=60) # keeps track of recently detected actors
        point_list = o3d.geometry.PointCloud()
        lidar.listen(lambda data: semantic_lidar_callback(world, data, point_list, lidar_save_path, dynamic_actors_idx_queue, ego_vehicle))

        # create open3d visualizer for Lidar
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=960,
            left=480,
            top=480)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        add_open3d_axis(vis)

        frame = 0
        dt0 = datetime.now()

        for frame_count in range(args.frames):
            world.tick()
            print('frame count', frame_count)
            
            # record the position and attributes of the ego_vehicle
            # save_ego_data(world, ego_vehicle, config['ego_save_path'])
            # move the xy position of the lidar sensor to match the ego_vehicle
            update_sensor_location(world, lidar, ego_vehicle)

            # update the position of the spectator
            if frame == 5: # wait for car to "fall to the ground" upon sim start
                update_sensor_location(world, spectator, ego_vehicle)
            time.sleep(.2)
            
            # take care of 3d plot stuff for live inspection
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)

            vis.poll_events()
            vis.update_renderer()
            
            process_time = datetime.now() - dt0
            sys.stdout.write('\r' + 'processing FPS: ' + str(1.0 / process_time.total_seconds()))
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1
            
    finally:
        lidar.destroy()
        for v in world.get_actors().filter('vehicle.*'):
            v.destroy()
        vis.destroy_window()
        world.apply_settings(original_settings)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-rendering', action='store_true', help='Use no-rendering mode for performance')
    parser.add_argument('--frames', default=1000, type=int, help='Number of frames to run the simulation')

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print(' - Exited by user.')