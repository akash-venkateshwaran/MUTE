from launch import LaunchDescription
from launch_ros.actions import Node
import matplotlib.pyplot as plt
import rclpy

import os
import pickle
import numpy as np
from environment.py_polyshp import Polyshp
import shutil
import launch.logging


from mammals.launch_utils import plot_species_samples, validate_samples, get_bounds, export_samples_to_yaml, CSVMammalSampler

from pathlib import Path

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

TL_MODEL_PATH   = str(src_dir / 'noise_propagation_model' /'noise_propagation_model' / 'models' / 'RBF' / 'model')
BAG_EXPORT_DIR  = str(src_dir / 'ship' / case_directory / 'mammal_data')
SHORELINE_PATH  = str(src_dir / "environment" / "shoreline_data")


case_directory_path = os.path.join(base_directory, case_directory)
clear_directory = True
bounds = get_bounds()

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



if os.path.exists(BAG_EXPORT_DIR):
    while True:
        permission = input(f"The directory {BAG_EXPORT_DIR} already exists for bag files. Do you want to delete it? (y/n): ")
        if permission.lower() == 'y':
            shutil.rmtree(BAG_EXPORT_DIR)
            os.makedirs(BAG_EXPORT_DIR)
            print(f"Cleared and recreated directory: {BAG_EXPORT_DIR}")
            break
        elif permission.lower() == 'n':
            print("Directory not cleared. Proceeding with existing contents.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
else:
    os.makedirs(BAG_EXPORT_DIR)
    print(f"Created new directory: {BAG_EXPORT_DIR}")


yaml_files = {}
temp = 0
for filename in os.listdir(case_directory_path):
        if filename.endswith('_params.yaml'):
            node_name = filename.replace('_params.yaml', '')
            yaml_files[node_name] = os.path.join(case_directory_path, filename)
            temp += 1

MAMMAL_YAML_LIST = [(name, path) for name, path in yaml_files.items()]

print("Total number of mammals processing:", temp)


def generate_launch_description():
    logger = launch.logging.get_logger('launch')

    if not os.path.exists(TL_MODEL_PATH):
        logger.error(f"TL_model_path does not exist: {TL_MODEL_PATH}")
        exit(1)
    
    bounds_dict = {
        'lat_max': bounds.max_lat,
        'lat_min': bounds.min_lat,
        'lon_max': bounds.max_lon,
        'lon_min': bounds.min_lon,
    }

    manager = Node(
        package    ='mammals',
        executable ='mammal_manager',
        name       ='mammal_manager',
        output     ='screen',
        parameters =[{
            'mammal_yaml_paths': [p for _, p in MAMMAL_YAML_LIST],
            'mammal_names'    : [n for n, _ in MAMMAL_YAML_LIST],
            'bag_dir'         : BAG_EXPORT_DIR,
            'tl_model_path'   : TL_MODEL_PATH,
            'shoreline_path'  : SHORELINE_PATH,
            # Pass bounds as individual parameters with namespace
            'bounds.lat_max'  : bounds_dict['lat_max'],
            'bounds.lat_min'  : bounds_dict['lat_min'],
            'bounds.lon_max'  : bounds_dict['lon_max'],
            'bounds.lon_min'  : bounds_dict['lon_min'],
        }],
    )

    return LaunchDescription([manager])



if __name__ == "__main__":
    generate_launch_description()