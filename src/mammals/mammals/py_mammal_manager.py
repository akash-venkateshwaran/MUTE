#!/usr/bin/env python3
"""
Starts many Mammal nodes in-process so that heavy objects are shared.
"""

from typing import List
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
import yaml
import importlib
import os
import sys
from pathlib import Path

# --- local imports --------------------------------------------------
from mammals.py_mammal import Mammal          # your existing class
from mammals import shared_assets               # new helper below
# --------------------------------------------------------------------

def load_yaml_as_dict(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # If the YAML is {node_name: {ros__parameters: {...}}}
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values()))
    if isinstance(data, dict) and 'ros__parameters' in data:
        data = data['ros__parameters']

    return data                 # FLAT dict of primitives

class MammalManager(Node):
    """Parent node – just to own parameters and clean shutdown."""
    def __init__(self):
        super().__init__('mammal_manager')

        self.declare_parameter('mammal_yaml_paths', [''])  # Use string list default
        self.declare_parameter('mammal_names', [''])       # Use string list default
        self.declare_parameter('bag_dir', '')
        self.declare_parameter('tl_model_path', '')
        self.declare_parameter('shoreline_path', '')

        self.declare_parameter('bounds.lat_max', 0.0)
        self.declare_parameter('bounds.lat_min', 0.0)
        self.declare_parameter('bounds.lon_max', 0.0)
        self.declare_parameter('bounds.lon_min', 0.0)

        # ①  Read parameters passed from launch
        yaml_paths   : List[str] = self.get_parameter(
            'mammal_yaml_paths').get_parameter_value().string_array_value
        mammal_names : List[str] = self.get_parameter(
            'mammal_names').get_parameter_value().string_array_value
        bag_dir      : str = self.get_parameter(
            'bag_dir').get_parameter_value().string_value
        tl_model_path: str = self.get_parameter(
            'tl_model_path').get_parameter_value().string_value
        shoreline_path: str = self.get_parameter(
            'shoreline_path').get_parameter_value().string_value
        bounds_dict = {
            'lat_max': self.get_parameter('bounds.lat_max').get_parameter_value().double_value,
            'lat_min': self.get_parameter('bounds.lat_min').get_parameter_value().double_value,
            'lon_max': self.get_parameter('bounds.lon_max').get_parameter_value().double_value,
            'lon_min': self.get_parameter('bounds.lon_min').get_parameter_value().double_value,
        }

        if len(yaml_paths) != len(mammal_names):
            raise RuntimeError("mammal_names and mammal_yaml_paths length mismatch")

        # ───────────────────────────────────────────────────────────
        self.get_logger().info("Loading shared assets …")
        shared_assets.init_shared(tl_model_path, shoreline_path, bounds_dict)
        # ───────────────────────────────────────────────────────────

        self._mammal_nodes = []
        for name, yaml_path in zip(mammal_names, yaml_paths):
                params_from_file = load_yaml_as_dict(yaml_path)


                # Bundle every override in one plain dict
                param_overrides = [
                        *[Parameter(k, value=v) for k, v in params_from_file.items()],
                        Parameter('bag_dir', value=bag_dir),
                ]
                # ---------------------------------------------------------------

                m = Mammal(
                        node_name=name,
                        parameter_overrides=param_overrides,   # ← list[Parameter]
                        shared_assets=shared_assets,
                )
                self._mammal_nodes.append(m)

def main(args=None):
    rclpy.init(args=args)
    manager = MammalManager()
    exec = MultiThreadedExecutor()
    exec.add_node(manager)
    for m in manager._mammal_nodes:
        exec.add_node(m)

    try:
        exec.spin()
    finally:
        for m in manager._mammal_nodes:
            m.destroy_node()
        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
