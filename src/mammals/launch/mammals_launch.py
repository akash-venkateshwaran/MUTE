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



case_directory = 'case_studies/Case_D' 
month = "Oct"
csv_path = os.path.join(
        base_directory,
        "new_kde_models",
        "posterior_mean_map.csv"
    )

csv_sampler = CSVMammalSampler(csv_path, month)

# Note this case study path is to store the transmission loss as ros2 bag file next to the ship ros2 bag file. The reason I have to export is cause transmission loss computation might
# not catup with the simulation time. Hence we need to run it in the bg

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
    user_input = input(f"The directory {BAG_EXPORT_DIR} already exists for bag files. Do you want to delete it? (y/n): ").strip().lower()

    if user_input == 'y':
        try:
            shutil.rmtree(BAG_EXPORT_DIR)
            print(f"Deleted existing directory: {BAG_EXPORT_DIR}")
        except Exception as e:
            print(f"Failed to delete directory: {e}")
            raise
    else:
        print("User chose not to delete the directory. Aborting.")
else:
    os.makedirs(BAG_EXPORT_DIR)
    print(f"Created new directory: {BAG_EXPORT_DIR}")

if clear_directory:
# Fetch bounds using the service
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


    shoreline_paths = os.path.join(src_dir, "environment", "shoreline_data")
    shoreline = Polyshp(shoreline_paths)
    polygons = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)

    print("Necessary polygon extracted")

    species_samples = {}
    for species, count in num_mammals_per_species.items():
        valid_samples = np.empty((0, 5))  # (lat lon depth heading speed)
        filename = f"kde_model_species_{species}.pickle"
        kde_dep_path = os.path.join(base_directory, "kde_models", "depth_data", filename)
        with open(kde_dep_path, 'rb') as file:
            kde_dep = pickle.load(file)

        print(f"Sampling mammals of species {species}, {valid_samples.shape[0]}/{count}")
        while valid_samples.shape[0] < count:
            needed = count - valid_samples.shape[0]

            # replace KDE sampling with CSV sampling
            samples_pop = csv_sampler.sample(needed)
            # samples_pop[:, 0] = np.degrees(samples_pop[:, 0])  # Convert latitude
            # samples_pop[:, 1] = np.degrees(samples_pop[:, 1])  # Convert longitude

            samples_dep = kde_dep.sample(needed)
            headings = np.random.uniform(-180, 180, size=(needed, 1))
            speeds = np.random.uniform(1, 4, size=(needed, 1))  # 1–4 m/s speeds

            samples = np.hstack((samples_pop, samples_dep, headings, speeds))
            valid_samples_batch = validate_samples(
                samples, polygons, lat_min, lat_max, lon_min, lon_max
            )
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