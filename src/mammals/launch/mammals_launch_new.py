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

def menu(case_dir: Path, kde_dir: Path, watch: TimeWatcher, bounds):
    """
    Interactive menu for mammal lifecycle operations using the
    MammalInitializer and MammalManager classes.
    """
    mammal_manager = MammalManager(case_dir, kde_dir, watch, bounds)

    while watch.behind_eta():
        species = int(prompt_float("Species    :"))
        mammal_manager.show_history(species=species)
        print("""
                1) Initialise mammal
                2) Terminate mammal
                q) Quit
                """)
        sel = input("> ").strip().lower()
        
        if sel == "1":
            
            mammal_manager.initialise()
        elif sel == "2":

            mammal_manager.terminate(species=species)
        elif sel == "q":
            break

def kill_all_mammal_nodes():
    subprocess.run('kill $(pgrep -f "mammal_exe")', shell=True)


def main():
     # --- single init
    rclpy.init()
    # your time watcher node
    watch = TimeWatcher()
    # create one bounds‐client node
    bounds_node = rclpy.create_node('bounds_service_client')
    # executor for both
    executor = MultiThreadedExecutor()
    executor.add_node(watch)
    executor.add_node(bounds_node)

    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()

    # prompt for case and directories before spinning
    src      = find_src_dir(Path(__file__))
    base     = src / "mammals"
    case_dir = prompt_for_new_case(base)
    kde_dir  = base / "kde_models"

    # wait for clock & ETA
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

        print("Time topics ready – interactive session begins.")
        while rclpy.ok() and watch.behind_eta():
            menu(case_dir, kde_dir, watch, bounds)
            time.sleep(0.2)
        print(f"Interactive section completed. ETA: {watch.eta}, Sim time: {watch.simulation_time}")

    except KeyboardInterrupt:
        # fall through to finally
        pass

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("\n⏹  Shutting down ROS…")
        try:
            executor.shutdown()
            watch.destroy_node()
            bounds_node.destroy_node()
            rclpy.shutdown()
        except Exception as e:
            print(f"Shutdown error (can be ignored): {e}")

        print("🗑  Cleaning up any stray mammal_exe processes…")
        kill_all_mammal_nodes()
        print("✔  All done.")

if __name__ == "__main__":
    main()
