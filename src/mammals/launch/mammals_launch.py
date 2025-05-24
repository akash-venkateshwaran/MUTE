"""ROS 2 script to manage marine‑mammal nodes for a case study.

Features
--------
1. Interactively create a new case‑study directory (error if it already exists).
2. Subscribe to /clock (simulation time) and /ETA (expected end‑time) – wait until both are
   available, printing “waiting…” meanwhile.
3. While the current simulation time is < ETA, offer a command‑line menu to either
   ● Initialise a new mammal node (manual values or KDE sampling)
   ● Terminate an existing mammal node (sets `end_clock` in its YAML file)
4. YAML files live inside the new case folder, one per mammal, e.g. `m0_1_params.yaml`.


"""

from launch import LaunchDescription
from launch_ros.actions import Node
import matplotlib.pyplot as plt
import rclpy

import os
import pickle
import numpy as np
from environment.py_polyshp import Polyshp
import geopandas as gpd
from shapely.geometry import box
import shutil
from rclpy.parameter import Parameter
import launch.logging

from custom_interfaces.srv import BoundsService


from mammals.launch_utils import plot_species_samples, validate_samples, export_samples_to_yaml

from pathlib import Path

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



def find_src_dir(start_path: Path) -> Path:
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate a 'src' directory.")

src_dir = find_src_dir(Path(__file__))

base_directory = str(src_dir / 'mammals')
case_directory = 'case_studies/Case_B' 
TL_model_path = str(src_dir / 'noise_propagation_model' /'noise_propagation_model' / 'models' / 'RBF' / 'model')


# Note this case study path is to store the transmission loss as ros2 bag file next to the ship ros2 bag file. The reason I have to export is cause transmission loss computation might
# not catup with the simulation time. Hence we need to run it in the bg
bag_export_dir = str(src_dir / 'ship' / case_directory / 'mammal_data')



case_directory_path = os.path.join(base_directory, case_directory)
clear_directory = True
if os.path.exists(case_directory_path) and os.listdir(case_directory_path):
    while True:
        permission = input(f"The directory {case_directory_path} is not empty. Do you want to clear it? (y/n): ")
        if permission.lower() == 'y':
            shutil.rmtree(case_directory_path)
            os.makedirs(case_directory_path)
            break
        elif permission.lower() == 'n':
            print("Directory not cleared. Proceeding with existing contents.")
            clear_directory = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if os.path.exists(bag_export_dir):
    user_input = input(f"The directory {bag_export_dir} already exists for bag files. Do you want to delete it? (y/n): ").strip().lower()

    if user_input == 'y':
        try:
            shutil.rmtree(bag_export_dir)
            print(f"Deleted existing directory: {bag_export_dir}")
        except Exception as e:
            print(f"Failed to delete directory: {e}")
            raise
    else:
        print("User chose not to delete the directory. Aborting.")
else:
    os.makedirs(bag_export_dir)
    print(f"Created new directory: {bag_export_dir}")

if clear_directory:
# Fetch bounds using the service
    bounds = get_bounds()
    lat_max = bounds.max_lat
    lat_min = bounds.min_lat
    lon_max = bounds.max_lon
    lon_min = bounds.min_lon



    pickle_files = [f for f in os.listdir(os.path.join(base_directory, "kde_models")) if f.endswith('.pickle')]
    num_species = len(pickle_files)
    print(f"Number of available species: {num_species}")
    species_selection = input("Which species numbers to consider (e.g., 0,1,2,3)? ")


    selected_species = map(int, species_selection.split(','))
    num_mammals_per_species = {}
    for species in selected_species:
        count = input(f"Enter the number of mammals for species {species}: ")
        num_mammals_per_species[species] = int(count)


    shoreline_paths = os.path.join(base_directory, "data", "shoreline")
    shoreline = Polyshp(shoreline_paths)
    polygons = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)

    print("Necessary polygon extracted")

    species_samples = {}
    for species, count in num_mammals_per_species.items():
        valid_samples = np.empty((0, 5))  # (lat lon depth heading speed)
        filename = f"kde_model_species_{species}.pickle"
        kde_pop_path = os.path.join(base_directory, "kde_models", "population_data", filename)
        kde_dep_path = os.path.join(base_directory, "kde_models", "depth_data", filename)

        if not os.path.exists(kde_pop_path) or not os.path.exists(kde_dep_path):
            print(f"Error: KDE model file for species {species} not found in both population_data and depth_data directories.")
            continue

        with open(kde_pop_path, 'rb') as file:
            kde_pop = pickle.load(file)

        with open(kde_dep_path, 'rb') as file:
            kde_dep = pickle.load(file)

        print(f"Sampling mammals of species {species}, {valid_samples.shape[0]}/{count}")
        while valid_samples.shape[0] < count:
            needed = count - valid_samples.shape[0]
            samples_pop = kde_pop.sample(needed)
            samples_pop[:, 0] = np.degrees(samples_pop[:, 0])  # Convert latitude
            samples_pop[:, 1] = np.degrees(samples_pop[:, 1])  # Convert longitude
            samples_dep = kde_dep.sample(needed)
            headings = np.random.uniform(-180, 180, size=(needed, 1))
            speeds = np.random.uniform(1, 4, size=(needed, 1)) # The speed is from 1 m/s to 4 m/s for killer whales
            samples = np.hstack((samples_pop, samples_dep, headings, speeds))
            
            valid_samples_batch = validate_samples(samples, polygons, lat_min, lat_max, lon_min, lon_max)
            valid_samples = np.vstack([valid_samples, valid_samples_batch])

        species_samples[species] = valid_samples[:count, :]

    plot_species_samples(species_samples, polygons, case_directory_path)
    yaml_files = export_samples_to_yaml(species_samples, case_directory_path)
else:
    yaml_files = {}
    temp = 0
    for filename in os.listdir(case_directory_path):
        if filename.endswith('_params.yaml'):
            node_name = filename.replace('_params.yaml', '')
            yaml_files[node_name] = os.path.join(case_directory_path, filename)
            temp += 1
    print("Total number of mammals processing:", temp)


def generate_launch_description():
    launch_description = LaunchDescription()
    logger = launch.logging.get_logger('launch')


    if not os.path.exists(TL_model_path):
        logger.error(f"TL_model_path does not exist: {TL_model_path}")
        exit(1)

    for node_name, yaml_file_path in yaml_files.items():
        print(f"Launching node {node_name} with parameters from: {yaml_file_path}")

        node = Node(
            package='mammals',
            executable='mammal_exe',
            name=node_name,
            parameters=[
                yaml_file_path, 
                {'bag_dir': bag_export_dir, 'TL_model_path': TL_model_path}
            ],
            output='screen'
        )
        launch_description.add_action(node)

    return launch_description

if __name__ == "__main__":
    generate_launch_description()