

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
import carla
import yaml
import cv2
from PIL import Image
import math


(255, 255, 255)
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # 0 None
    (76, 76, 76),    # 1 road
    (100, 40, 40),   # 2 sidewalk
    (55, 90, 80),    # 3 Other obstacles? (keep)
    (220, 20, 60),   # 4 fence?
    (153, 153, 153), # 5 Pole?
    (157, 234, 50),  # 6 sign posts
    (128, 64, 128),  # 7
    (244, 35, 232),  # 8 
    (107, 142, 35),  # 9 Vegetation
    (0, 0, 142),     # 10
    (102, 102, 156), # 11
    (220, 220, 0),   # 12
    (70, 130, 180),  # 13 humans
    (81, 0, 81),     # 14 CARS (not busses or motorcycles
    (150, 100, 100), # 15 VW Bus lol
    (230, 150, 140), # 16 RailTrack (not sure)
    (180, 165, 180), # 17 
    (250, 170, 30),  # 18 Motorcycle
    (110, 190, 160), # 19 Bicycle
    (170, 120, 50),  # 20 post chain boundary
    (45, 60, 150),   # 21 trash bins and restaurant tripod signs road paper
    (145, 170, 100), # 22 
    (145, 170, 100), # 23 
    (145, 170, 100), # 24 roadlines
    (145, 170, 100), # 25 medians
    (145, 170, 100), # 26 
    (0, 0, 0), # 27
    (255, 255, 255), # 28 terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses



def semantic_lidar_callback(world, point_cloud, point_list, lidar_save_path, dynamic_actors_idx_queue, ego_vehicle):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), 
        ('y', np.float32), 
        ('z', np.float32),
        ('CosAngle', np.float32), # angle incident on objects
        ('ObjIdx', np.uint32), 
        ('ObjTag', np.uint32)]))
        
    pc_dtype = [
        ('x', float),
        ('y', float),
        ('z', float),
        ('ObjIdx', int)
    ]

    # Prepare the dtype for the structured array
    actor_dtype = [
        ('frame_index', int),
        ('ObjIdx', int),
        ('transform', float, (6,)),  # (x, y, z, pitch, yaw, roll)
        ('velocity', float, (3,)),  # (x, y, z)
        ('angular_velocity', float, (3,)),  # (x, y, z)
        ('acceleration', float, (3,)),  # (x, y, z)
        ('bounding_box', float, (7,))  # (center_x, center_y, center_z, extent_x, extent_y, extent_z, rotation_yaw)
    ]


    # get the unique object IDs that belong to moving objects in the scene
    dynamic_tags = [13,14,15,18,19]
    detected_dynamic_actor_ids = np.unique(data['ObjIdx'][np.isin(data['ObjTag'], dynamic_tags)]).astype(int).tolist()
    # dynamic_actors_idx_queue.put([point_cloud.frame, detected_dynamic_actor_ids])
    dynamic_actors_idx_queue.append(detected_dynamic_actor_ids)
    
    # convert the queue of lists into a set
    temp = list(set([item for sublist in dynamic_actors_idx_queue for item in sublist]))
  
    # TODO: put this in the main loop!    
    moving_actors = world.get_actors(temp)
    print('moving_actors', temp)
  
    actor_data = np.empty(len(moving_actors), dtype=actor_dtype)
    # save the position, velocity, angular velocity, and acceleration, and bounding box info for each moving actor
    for i, actor in enumerate(moving_actors):
        transform = actor.get_transform()
        actor_data[i] = (
            point_cloud.frame,
            actor.id,
            (transform.location.x, transform.location.y, transform.location.z, transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
            (actor.get_velocity().x, actor.get_velocity().y, actor.get_velocity().z),
            (actor.get_angular_velocity().x, actor.get_angular_velocity().y, actor.get_angular_velocity().z),
            (actor.get_acceleration().x, actor.get_acceleration().y, actor.get_acceleration().z),
            (actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z, actor.bounding_box.extent.x, actor.bounding_box.extent.y, actor.bounding_box.extent.z, actor.bounding_box.rotation.yaw)
        )
        
    # save the actor data to a numpy file TODO: make this happen outside of the callback -> add detected moving vehicles to queue
    np.save(lidar_save_path+'/actor_data/frame_%07d.npy' % point_cloud.frame, actor_data)
         
         
         
    ego_data = np.empty(1, dtype=actor_dtype)
    # save the position, velocity, angular velocity, and acceleration, and bounding box of the ego vehicle
    ego_transform = ego_vehicle.get_transform()
    ego_data = (
        point_cloud.frame,
        ego_vehicle.id,
        (ego_transform.location.x, ego_transform.location.y, ego_transform.location.z, ego_transform.rotation.pitch, ego_transform.rotation.yaw, ego_transform.rotation.roll),
        (ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y, ego_vehicle.get_velocity().z),
        (ego_vehicle.get_angular_velocity().x, ego_vehicle.get_angular_velocity().y, ego_vehicle.get_angular_velocity().z),
        (ego_vehicle.get_acceleration().x, ego_vehicle.get_acceleration().y, ego_vehicle.get_acceleration().z),
        (ego_vehicle.bounding_box.location.x, ego_vehicle.bounding_box.location.y, ego_vehicle.bounding_box.location.z, ego_vehicle.bounding_box.extent.x, ego_vehicle.bounding_box.extent.y, ego_vehicle.bounding_box.extent.z, ego_vehicle.bounding_box.rotation.yaw)
    )
        # save the actor data to a numpy file TODO: make this happen outside of the callback -> add detected moving vehicles to queue
    np.save(lidar_save_path+'/ego_data/frame_%07d.npy' % point_cloud.frame, ego_data)
        
        
    # filter point cloud to only include static objects in the environment
    static_obstacle_tags = [3,4,5,6,20]

    # now filter the data the way we filtered the points
    f_data = data[np.isin(data['ObjTag'], static_obstacle_tags)]
    # Create an empty structured array with the dtype defined above
    filtered_data = np.empty(f_data.shape[0], dtype=pc_dtype)
    # Save the data into the structured array
    filtered_data['x'] = f_data['x']
    filtered_data['y'] = f_data['y']
    filtered_data['z'] = f_data['z']
    filtered_data['ObjIdx'] = f_data['ObjIdx']
    
    # save the point cloud data to a numpy file
    np.save(lidar_save_path+'/point_cloud/frame_%07d.npy' % point_cloud.frame, filtered_data)
    
    #################################################################################
    # fpr display purposes
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T
    points = points[np.isin(data['ObjTag'], static_obstacle_tags + dynamic_tags)]
    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    # int_color = LABEL_COLORS[labels]
    int_color = np.array([[0,255,0] for _ in range(len(points))]) / 255.0

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    #################################################################################


def generate_lidar_bp(arg, world, blueprint_library, lidar_properties, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    
    for attr, val in lidar_properties.items():
        print(attr, val)
        if attr in ['location']: 
            continue
        elif attr in ['save_path']:
            # Generate a timestamp for the run
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = val+'/recording_'+timestamp  # Include the timestamp in the path
            print('save path', save_path)
            os.makedirs(save_path+'/point_cloud', exist_ok=True)
            os.makedirs(save_path+'/actor_data', exist_ok=True)
            os.makedirs(save_path+'/ego_data', exist_ok=True)
        else:
            lidar_bp.set_attribute(attr, str(val))
    
    # handle this one separately
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    return lidar_bp, save_path
    
    
def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)
    

def load_simulation_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_world_weather(world, weather_config):
    weather = carla.WeatherParameters()
    for param, value in weather_config.items():
        setattr(weather, param, value)
    world.set_weather(weather)
    print('weather set')


def update_sensor_location(world, sensor, ego_vehicle):
    # Get the current location and rotation of the ego car
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_rotation = ego_transform.rotation

    # Get the current location and rotation of the camera
    sensor_transform = sensor.get_transform()
    sensor_location = sensor_transform.location
    sensor_rotation = sensor_transform.rotation

    # Calculate the desired x and y coordinates for the camera
    desired_x = ego_location.x
    desired_y = ego_location.y
    desired_yaw = ego_rotation.yaw
    # print('yaw', desired_yaw)

    # Update the camera's location with the new x and y coordinates while keeping the other attributes
    new_location = carla.Location(x=desired_x, y=desired_y, z=sensor_location.z)

    # # Update the camera's rotation to match the yaw of the ego vehicle
    # new_rotation = carla.Rotation(pitch=-90, yaw=desired_yaw, roll=0)
    # sensor.set_transform(carla.Transform(new_location, new_rotation))

    # do the transformation without changing the rotation
    sensor.set_transform(carla.Transform(new_location, sensor.get_transform().rotation))


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # print(sensor_name)
    sensor_type = sensor_name[:-4]
    # Do stuff with the sensor_data data like save it to disk
    if sensor_type == 'rgb':
        sensor_data.save_to_disk('output/' + sensor_name + '/' + 'frame_%06d' % sensor_data.frame)

    elif sensor_type == 'optical_flow':
        opt_flow = np.frombuffer(sensor_data.raw_data, dtype=np.float32)
        opt_flow = np.reshape(opt_flow, (sensor_data.height, sensor_data.width, 2))
        np.save('output/' + sensor_name + '/' + 'frame_%06d.npy' % sensor_data.frame, opt_flow)

    elif sensor_type == 'semantic_segmentation':
        sensor_data.save_to_disk('output/' + sensor_name + '/' + 'frame_%06d' % sensor_data.frame, carla.ColorConverter.CityScapesPalette)

    elif sensor_type == 'instance_segmentation':
        sensor_data.save_to_disk('output/' + sensor_name + '/' + 'frame_%06d' % sensor_data.frame)

    elif sensor_type == 'depth':
        depth = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
        depth = depth.reshape((sensor_data.height, sensor_data.width, 4)) # # Array of BGRA 32-bit pixels
        R, G, B = depth[:, :, 2], depth[:, :, 1], depth[:, :, 0] # isolate the RGB channels
        depth = 1000 * (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0) # convert RGB to float
        np.save('output/' + sensor_name + '/frame_%06d.npy' % sensor_data.frame, depth)

    else:  
        print('sensor type not recognized', sensor_type)

    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data.raw_data))
    # print(sensor_queue)


def set_dynamic_weather_ticks(world, t, config):
    
    # Set period in terms of simulation time
    period_ticks_cloudiness = config['period_ticks_cloudiness']
    period_ticks_precipitation = config['period_ticks_precipitation']
    period_ticks_precipitation_deposits = config['period_ticks_precipitation_deposits']
    period_ticks_sun_azimuth = config['period_ticks_sun_azimuth']
    period_ticks_sun_altitude = config['period_ticks_sun_altitude']

    # Set the min and max values for each weather property
    min_max_cloudiness = config['min_max_cloudiness']
    min_max_precipitation = config['min_max_precipitation']
    min_max_precipitation_deposits = config['min_max_precipitation_deposits']

    # Set the min and max values for sun azimuth and altitude angles
    min_max_sun_azimuth = config['min_max_sun_azimuth']
    min_max_sun_altitude = config['min_max_sun_altitude']
    
    # Get the current weather
    weather = world.get_weather()

    # Calculate frequency for sinusoidal oscillation (2 * PI / period)
    frequency_cloudiness = 2 * math.pi / period_ticks_cloudiness
    frequency_precipitation = 2 * math.pi / period_ticks_precipitation
    frequency_precipitation_deposits = 2 * math.pi / period_ticks_precipitation_deposits

    # Apply sinusoidal oscillation to cloudiness, precipitation, and precipitation deposits
    cloudiness_range = min_max_cloudiness[1] - min_max_cloudiness[0]
    weather.cloudiness = cloudiness_range * 0.5 * (math.sin(frequency_cloudiness * t) + 1) + min_max_cloudiness[0]
    
    precipitation_range = min_max_precipitation[1] - min_max_precipitation[0]
    weather.precipitation = precipitation_range * 0.5 * (math.sin(frequency_precipitation * t) + 1) + min_max_precipitation[0]
    
    precipitation_deposits_range = min_max_precipitation_deposits[1] - min_max_precipitation_deposits[0]
    weather.precipitation_deposits = precipitation_deposits_range * 0.5 * (math.sin(frequency_precipitation_deposits * t) + 1) + min_max_precipitation_deposits[0]

    # Calculate frequency for sun oscillation
    frequency_azimuth = 2 * math.pi / period_ticks_sun_azimuth
    frequency_altitude = 2 * math.pi / period_ticks_sun_altitude

    # Update sun azimuth and altitude angles
    print(min_max_sun_azimuth)
    sun_azimuth_range = min_max_sun_azimuth[1] - min_max_sun_azimuth[0]
    weather.sun_azimuth_angle = sun_azimuth_range * 0.5 * (math.sin(frequency_azimuth * t) + 1) + min_max_sun_azimuth[0]

    sun_altitude_range = min_max_sun_altitude[1] - min_max_sun_altitude[0]
    weather.sun_altitude_angle = sun_altitude_range * 0.5 * (math.sin(frequency_altitude * t) + 1) + min_max_sun_altitude[0]

    # Apply the new weather
    world.set_weather(weather)

def get_3_points_on_vehicle(vehicle):
    # Vehicle Details
    location_x, location_y = vehicle.get_transform().location.x, vehicle.get_transform().location.y
    yaw = vehicle.get_transform().rotation.yaw
    extent = vehicle.bounding_box.extent.x

    # Calculate points and rotate them according to yaw
    rotate = lambda x: (location_x + math.cos(math.radians(yaw)) * x, 
                        location_y + math.sin(math.radians(yaw)) * x)

    return rotate(extent), rotate(0), rotate(-extent)

def pixel_to_world(pixel_coords, camera_coords, image_width, image_height, camera_height, fov, yaw):
    # Check if pixel_coords is not empty
    if len(pixel_coords) == 0:
        return np.array([])  # return an empty array or whatever is appropriate in your context

    # Compute the size of the world view in meters at the ground level based on the FOV
    world_width = 2 * camera_height * np.tan(np.radians(fov/2))

    # Compute pixel per meter scale
    ppm = image_width / world_width

    # Convert pixel_coords and camera_coords to NumPy arrays
    pixel_coords = np.array(pixel_coords)
    camera_coords = np.array(camera_coords)

    # Translate pixel coordinates to world coordinates, considering the image center as the origin
    d_coords = (pixel_coords - image_width / 2) / ppm

    # Check if the pixel_coords is 1D or 2D
    if pixel_coords.ndim == 1:
        # print('here')
        d_coords = np.array([-d_coords[0], d_coords[1]])  # Swap x and y components
    else:  # if it's 2D
        # print('there')
        d_coords = np.column_stack((-d_coords[:, 0], d_coords[:, 1])) # close

    # Rotate coordinates according to the camera's yaw
    yaw_rad = np.radians(yaw)
    rot_matrix = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                           [np.sin(yaw_rad),  np.cos(yaw_rad)]])
    d_coords = d_coords @ rot_matrix.T  # Note that the rotation matrix is transposed as we're rotating points, not transforming basis

    # Add the camera_coords to get the world coordinates
    world_coords = d_coords + camera_coords

    return world_coords


def world_to_pixel(world_coords, camera_coords, image_width, image_height, camera_height, fov, yaw):
    # Check if world_coords is not empty
    if len(world_coords) == 0:
        return np.array([])  # return an empty array or whatever is appropriate in your context

    # Compute the size of the world view in meters at the ground level based on the FOV
    world_width = 2 * camera_height * np.tan(np.radians(fov/2))

    # Compute pixel per meter scale
    ppm = image_width / world_width

    # Convert world_coords and camera_coords to NumPy arrays
    world_coords = np.array(world_coords)
    camera_coords = np.array(camera_coords)

    # Calculate the relative coordinates with respect to the camera position
    d_coords = world_coords - camera_coords

    yaw_rad = np.radians(-yaw)
    rot_matrix = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                           [np.sin(yaw_rad),  np.cos(yaw_rad)]])
    d_coords = d_coords @ rot_matrix.T  # Note that the rotation matrix is transposed as we're rotating points, not transforming basis

    # Convert relative coordinates to pixel coordinates using the pixel per meter scale
    pixel_coords = d_coords * ppm

    # Add the image center to get the final pixel coordinates
    pixel_coords += image_width / 2
    # flip the x axis
    pixel_coords[:, 0] = image_width - pixel_coords[:, 0]

    return np.round(pixel_coords).astype(int)



def create_rgb_image(velocity, vmin=None, vmax=None):
    # Ensure that velocity has the correct shape
    assert len(velocity.shape) == 3 and velocity.shape[2] == 2, \
        "Input array must have shape (height, width, 2)"

    # If vmin or vmax are provided, clip the velocities
    if vmin is not None or vmax is not None:
        velocity = np.clip(velocity, vmin, vmax)

    # Separate the velocity components
    vx, vy = velocity[..., 0], velocity[..., 1]

    # Calculate magnitude and angle from vx and vy
    magnitude = np.sqrt(vx**2 + vy**2)
    angle = (np.arctan2(vy, vx) + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Create an HSV image where:
    # - Hue (H) is the angle (which gives the direction of the vector)
    # - Saturation (S) is always 1 (full color)
    # - Value (V) is the magnitude (which gives the length of the vector)
    hsv_image = np.zeros((*vx.shape, 3))  # 3 for H, S, V
    hsv_image[..., 0] = angle * 180  # OpenCV expects H in range [0, 180]
    hsv_image[..., 1] = 255  # OpenCV expects S in range [0, 255]
    hsv_image[..., 2] = (magnitude / np.max(magnitude)) * 255  # Normalize magnitude and scale to [0, 255]

    # Convert HSV image to RGB
    hsv_image = hsv_image.astype(np.uint8)
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return rgb_image
