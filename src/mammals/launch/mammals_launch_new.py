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
import subprocess
import threading
from launch import LaunchDescription
from launch_ros.actions import Node
import matplotlib.pyplot as plt
import rclpy
from rclpy.executors import MultiThreadedExecutor

import os
import pickle
import numpy as np
from environment.py_polyshp import Polyshp
import geopandas as gpd
from shapely.geometry import box
import shutil
from rclpy.parameter import Parameter
import launch.logging

import time
# from mammals.launch_utils import plot_species_samples, validate_samples, get_bounds, export_samples_to_yaml
from mammals.launch_utils import find_src_dir, prompt_for_new_case, prompt_float, TimeWatcher, MammalManager
from custom_interfaces.srv import BoundsService

from pathlib import Path


def get_bounds(node):
    """
    Call the BoundsService using an existing rclpy node.
    Assumes rclpy.init() has already been called.
    """
    client = node.create_client(BoundsService, 'bounds_service')
    node.get_logger().info('Waiting for BoundsService...')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('  … still waiting for BoundsService')
    request = BoundsService.Request()
    future = client.call_async(request)
    # block until we have a response
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        resp = future.result()
        node.get_logger().info(
            f"Received bounds: max_lat={resp.max_lat}, "
            f"min_lat={resp.min_lat}, max_lon={resp.max_lon}, min_lon={resp.min_lon}"
        )
        return resp
    else:
        node.get_logger().error(f"BoundsService call failed: {future.exception()}")
        return None


def main():
    rclpy.init()
    watch = TimeWatcher()
    bounds_node = rclpy.create_node('bounds_service_client')
    executor = MultiThreadedExecutor()
    executor.add_node(watch)
    executor.add_node(bounds_node)

    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    # Setup directories and bounds
    src = find_src_dir(Path(__file__))
    base = src / "mammals"
    case_dir = prompt_for_new_case(base)
    kde_dir = base / "kde_models"
    csv_path = os.path.join(
            base,
            "new_kde_models", 
            "posterior_mean_map.csv"
        )

    try:
        print("Waiting for /clock & /ETA …")
        while rclpy.ok() and not watch.ready():
            time.sleep(0.1)
        if not rclpy.ok():
            return

        print("Time topics ready – fetching bounds …")
        bounds = get_bounds(bounds_node)
        if bounds is None:
            print("Failed to get bounds, exiting.")
            return

        # Setup paths and create mammal manager
        TL_MODEL_PATH = str(src / 'noise_propagation_model' / 'noise_propagation_model'
                           / 'models' / 'RBF' / 'model')
        case_directory = case_dir.name
        BAG_EXPORT_DIR = str(src / 'ship' / case_directory / 'mammal_data')
        SHORELINE_PATH = str(src / 'environment' / 'shoreline_data')
        bounds_dict = {
            'lat_max': bounds.max_lat, 'lat_min': bounds.min_lat,
            'lon_max': bounds.max_lon, 'lon_min': bounds.min_lon,
        }

        mammal_manager = MammalManager(
            case_dir, kde_dir, watch, bounds,
            TL_MODEL_PATH, BAG_EXPORT_DIR, SHORELINE_PATH, bounds_dict, executor, csv_path
        )

        print("Interactive session begins.")
        
        # Main loop
        while rclpy.ok() and watch.behind_eta():
            # Update scheduler every cycle
            mammal_manager.update()
            
            # Handle user input
            species = int(prompt_float("Species: "))
            mammal_manager.show_history(species=species)
            print("""
                1) Initialise mammal
                2) Terminate mammal  
                q) Quit
                """)
            sel = input("> ").strip().lower()

            if sel == "1":
                mammal_manager.initialise(species)
            elif sel == "2":
                mammal_manager.terminate(species)
            elif sel == "q":
                break
                
            time.sleep(0.1)  # Small delay to prevent busy loop

        print(f"Session completed. ETA: {watch.eta}, Sim time: {watch.simulation_time}")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("\n⏹ Shutting down ROS…")
        try:
            executor.shutdown()
            watch.destroy_node()
            bounds_node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"Shutdown error: {e}")
        
        print("🗑 Cleaning up processes…")
        subprocess.run('kill $(pgrep -f "mammal_exe")', shell=True)
        print("✔ All done.")


if __name__ == "__main__":
    main()
