import os
import pickle
import csv
import numpy as np
from environment.py_polyshp import Polyshp
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import rclpy
from custom_interfaces.srv import BoundsService
import yaml
from typing import Dict

def get_bounds():
    rclpy.init()
    node = rclpy.create_node('bounds_service_client')
    client = node.create_client(BoundsService, 'bounds_service')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('BoundsService not available, waiting again...')
    request = BoundsService.Request()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        response = future.result()
        node.get_logger().info(
            f"Received bounds: max_lat={response.max_lat}, min_lat={response.min_lat}, "
            f"max_lon={response.max_lon}, min_lon={response.min_lon}"
        )
        rclpy.shutdown()
        return response
    else:
        node.get_logger().error('BoundsService call failed %r' % (future.exception(),))
        rclpy.shutdown()
        return None

def validate_samples(samples, polygons, lat_min, lat_max, lon_min, lon_max):
    """ Checks if the sample lies in water and has depth greater than zero"""
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(samples[:, 1], samples[:, 0]))
    valid_points = points[(points.geometry.x >= lon_min) & 
                          (points.geometry.x <= lon_max) & 
                          (points.geometry.y >= lat_min) & 
                          (points.geometry.y <= lat_max)]
    for polygon in polygons:
        valid_points = valid_points[~valid_points.within(polygon)]
    valid_indices = valid_points.index
    valid_indices = valid_indices[samples[valid_indices, 2] > 0]
    return samples[valid_indices]

def plot_species_samples(species_samples, polygons, directory):
    fig, ax = plt.subplots()
    polygons.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.5)
    for species, samples in species_samples.items():
        depths = samples[:, 2]
        scatter = ax.scatter(samples[:, 1], samples[:, 0], c=depths, cmap='Reds', label=f"Species {species}", s=10, alpha=0.8)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title('Scatter Plot of Mammal Samples by Species with Shoreline Outline')
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    plt.autoscale(enable=True, axis='both', tight=True)
    filename = "Mammals_initial_loc.png"
    plt.savefig(os.path.join(directory, filename))
    plt.close()

def export_samples_to_yaml(species_samples: Dict[int, np.ndarray], directory_path: str) -> Dict[str, str]:
    yaml_files = {}
    for species_id, samples in species_samples.items():
        for index, sample in enumerate(samples, start=1):
            node_name = f'm{species_id}_{index}'
            params = {
                'latitude': float(sample[0]),
                'longitude': float(sample[1]),
                'depth': float(sample[2]),
                'heading': float(sample[3]),
                'speed': float(sample[4]),
                'topic': f'{node_name}_state'
            }
            yaml_file_path = os.path.join(directory_path, f'{node_name}_params.yaml')
            with open(yaml_file_path, 'w') as file:
                yaml.dump({node_name: {'ros__parameters': params}}, file, default_flow_style=False)
            yaml_files[node_name] = yaml_file_path
    return yaml_files
