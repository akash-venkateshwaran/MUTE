from __future__ import annotations

import os, signal, psutil
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
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from shapely.geometry import Point, Polygon, MultiPolygon
import subprocess
import re
import heapq, itertools
from typing import Callable, Optional
from mammals import shared_assets
from rclpy.executors import MultiThreadedExecutor

from rclpy.parameter import Parameter
from mammals.py_mammal import Mammal
import pandas as pd


class CSVMammalSampler:
    """
    Reads a CSV of mammal-location intensities and draws random samples
    in proportion to those intensities.

    On initialization:
      - Loads the CSV and checks that the given month’s intensity column
        (e.g. "May_intensity") is present.
      - Extracts all (lat, lon) pairs.
      - Normalizes the intensity values so they sum to 1 (or uses a uniform
        distribution if all intensities are zero).

    The `sample(n)` method:
      - Uses numpy.random.choice with the normalized intensities as probabilities
        to select n indices (rows) from the data.
      - Returns the corresponding (lat, lon) coordinates as an (n, 2) array.
    """
    def __init__(self, csv_path: str, month: str):
        df = pd.read_csv(csv_path)
        if month not in df.columns:
            raise ValueError(f"'{month}' column not in CSV")
        self.month = month
        # extract lat/lon and probability
        self.latlon = df[['lat', 'lon']].to_numpy()
        prob = df[f"{month}_intensity"].values

        prob = prob / prob.sum() if prob.sum() else np.full_like(prob, 1/len(prob))
        self.prob = prob

    def sample(self, n: int) -> np.ndarray:
        idx = np.random.choice(len(self.prob), size=n, p=self.prob)
        return self.latlon[idx]



#################################################################
############################ MUTE V2 ############################





def prompt_float(label: str) -> float:
    while True:
        try:
            return float(input(f"  {label}: "))
        except ValueError:
            print("Invalid number.")


def find_src_dir(start: Path) -> Path:
    """Walk upwards until we locate the workspace /src folder."""
    current = start.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate a 'src' directory above " + str(start))


def prompt_for_new_case(base_dir: Path) -> Path:
    """Ask the user for a *new* case name, create case_studies/<name>, or die if it exists."""
    case_studies_dir = base_dir / "case_studies"
    case_studies_dir.mkdir(exist_ok=True)

    while True:
        name = input("Enter a new case name (e.g. Case_A): ").strip()
        if not name:
            print("Name cannot be empty.")
            continue
        case_dir = case_studies_dir / name
        if case_dir.exists():
            print(f"Error – directory '{case_dir}' already exists. Choose another name.")
            continue
        # create skeleton
        case_dir.mkdir(parents=True)
        print(f"Created case directory: {case_dir}")
        return case_dir

class TimeWatcher(Node):
    """Keeps track of simulation time (/clock) and ETA (/ETA), both in mins."""

    def __init__(self):
        super().__init__("time_watcher")
        self.simulation_time: Optional[int] = None
        self.eta: Optional[int] = None

        self.create_subscription(Clock, "/clock", self.clock_cb, 10)
        self.create_subscription(Clock, "/ETA",   self.eta_cb,   10)

    def clock_cb(self, msg: Clock):
        self.simulation_time = msg.clock.sec

    def eta_cb(self, msg: Clock):
        self.eta = msg.clock.sec

    def ready(self) -> bool:
        return self.simulation_time is not None and self.eta is not None

    def behind_eta(self) -> bool:
        assert self.ready()
        return self.simulation_time < self.eta


class MammalScheduler:
    """
    Simple scheduler for mammal nodes - no complex triggering system needed.
    Just checks scheduled times on each update cycle.
    """
    
    def __init__(self, executor: MultiThreadedExecutor):
        self.executor = executor
        self.scheduled_mammals: dict[str, dict] = {}  # {node_name: {start_time, end_time, node_obj, ...}}
        self.active_mammals: dict[str, rclpy.node.Node] = {}
        
    def schedule_mammal(self, node_name: str, start_time: int, end_time: int, 
                       mammal_config: dict):
        """Schedule a mammal to start and end at specific times."""
        self.scheduled_mammals[node_name] = {
            'start_time': start_time,
            'end_time': end_time,
            'config': mammal_config,
            'node_obj': None,
            'started': False,
            'terminated': False
        }
        
    def update(self, current_time: int):
        """Check and update mammal lifecycle based on current simulation time."""
        for node_name, info in self.scheduled_mammals.items():
            # Start mammals that should be running
            if (current_time >= info['start_time'] and 
                not info['started'] and 
                not info['terminated']):
                self._start_mammal(node_name, info)
                
            # Terminate mammals that should be stopped
            if (current_time >= info['end_time'] and 
                info['started'] and 
                not info['terminated']):
                self._terminate_mammal(node_name, info)
    
    def _start_mammal(self, node_name: str, info: dict):
        """Actually create and start the mammal node."""
        try:
            # Create the mammal node with config
            node = Mammal(
                node_name=node_name,
                parameter_overrides=info['config']['parameters'],
                shared_assets=info['config']['shared_assets'],
            )
            
            self.executor.add_node(node)
            self.active_mammals[node_name] = node
            info['node_obj'] = node
            info['started'] = True
            
            print(f"🚀 {node_name} started at t={info['start_time']} min")
            
        except Exception as e:
            print(f"❌ Failed to start {node_name}: {e}")
    
    def _terminate_mammal(self, node_name: str, info: dict):
        """Actually terminate the mammal node."""
        try:
            if node_name in self.active_mammals:
                node = self.active_mammals.pop(node_name)
                self.executor.remove_node(node)
                node.destroy_node()
                info['terminated'] = True
                
                print(f"✔ {node_name} terminated at t={info['end_time']} min")
                
        except Exception as e:
            print(f"❌ Failed to terminate {node_name}: {e}")
    
    def get_active_mammals(self, species: int = None) -> list[str]:
        """Get list of currently active mammal names, optionally filtered by species."""
        active = []
        for name, info in self.scheduled_mammals.items():
            if info['started'] and not info['terminated']:
                if species is None or name.startswith(f"m{species}_"):
                    active.append(name)
        return active
    
    def force_terminate(self, node_name: str, current_time: int):
        """Immediately terminate a specific mammal (user-requested)."""
        if node_name in self.scheduled_mammals:
            info = self.scheduled_mammals[node_name]
            info['end_time'] = current_time  # Update end time
            self._terminate_mammal(node_name, info)

            

class MammalManager:
    """Simplified mammal manager with CSV sampling instead of KDE models."""

    def __init__(
        self,
        case_dir: Path,
        kde_dir: Path,  # Now used for CSV models directory
        watcher: TimeWatcher,
        bounds,
        tl_model_path: str,
        bag_dir: str,
        shoreline_path: str,
        bounds_dict: dict,
        executor: MultiThreadedExecutor,
        csv_path : Path,
        month: str = "May"  # Add month parameter for CSV sampling
    ):
        self.case_dir = case_dir
        self.kde_dir = kde_dir  # Now points to new_kde_models directory
        self.bag_dir = bag_dir
        self.TL_model_path = tl_model_path
        self.watcher = watcher
        self.bounds = bounds
        self.month = month
        
        # Simple scheduler instead of complex triggering
        self.scheduler = MammalScheduler(executor)
        
        # Initialize shared assets once
        shared_assets.init_shared(tl_model_path, shoreline_path, bounds_dict)
        self.shoreline_polygons = shared_assets.SHORELINE_POLYGONS

        # Keep track of history for naming
        self.history: dict[int, dict[str, dict[str, int | None]]] = {}
        
        # Initialize CSV sampler
        self.csv_sampler = self._initialize_csv_sampler(csv_path=csv_path)

    def _initialize_csv_sampler(self, csv_path) -> CSVMammalSampler:
        """Initialize the CSV sampler for mammal population sampling."""
            
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        return CSVMammalSampler(csv_path, self.month)

    def next_node_name(self, species: int) -> str:
        """Generate next unique mammal name."""
        records = self.history.get(species, {})
        if not records:
            return f"m{species}_1"

        used = []
        for name in records.keys():
            try:
                used.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue

        next_idx = (max(used) if used else 0) + 1
        return f"m{species}_{next_idx}"

    def load_kde_depth_model(self, species: int = 0):
        """Load only the depth KDE model for species (population now uses CSV)."""
        dep_file = self.kde_dir / "depth_data" / f"kde_model_species_{species}.pickle"
        if not dep_file.exists():
            raise FileNotFoundError(f"Missing depth KDE model for species {species}: {dep_file}")
        
        import pickle
        with open(dep_file, "rb") as f:
            kde_dep = pickle.load(f)
        return kde_dep

    def sample_from_csv_and_kde(self, species: int) -> Tuple[float, float, float]:
        """Sample location from CSV and depth from KDE model."""
        # Sample location from CSV
        location_sample = self.csv_sampler.sample(1)[0]  # Get first (and only) sample
        lat, lon = float(location_sample[0]), float(location_sample[1])
        
        # Sample depth from KDE
        kde_dep = self.load_kde_depth_model(species)
        depth = float(kde_dep.sample(1))
        
        return lat, lon, depth

    def batch_sample_mammals(self, species: int, count: int) -> np.ndarray:
        """
        Batch sample mammals using CSV for population and KDE for depth.
        Returns array of shape (count, 5) with columns: lat, lon, depth, heading, speed
        """
        # Load depth KDE model
        kde_dep = self.load_kde_depth_model(species)
        
        valid_samples = np.empty((0, 5))  # (lat lon depth heading speed)
        
        print(f"Sampling mammals of species {species}, {valid_samples.shape[0]}/{count}")
        
        while valid_samples.shape[0] < count:
            needed = count - valid_samples.shape[0]
            
            # Sample population locations from CSV
            samples_pop = self.csv_sampler.sample(needed)
            
            # Sample depths from KDE
            samples_dep = kde_dep.sample(needed).reshape(-1, 1)
            
            # Generate random headings and speeds
            headings = np.random.uniform(-180, 180, size=(needed, 1))
            speeds = np.random.uniform(1, 4, size=(needed, 1))  # 1–4 m/s speeds
            
            # Combine all samples
            samples = np.hstack((samples_pop, samples_dep, headings, speeds))
            
            # Validate samples (assuming you have this function)
            try:
                valid_samples_batch = self.validate_samples(
                    samples, 
                    self.shoreline_polygons, 
                    self.bounds.min_lat, 
                    self.bounds.max_lat, 
                    self.bounds.min_lon, 
                    self.bounds.max_lon
                )
                valid_samples = np.vstack([valid_samples, valid_samples_batch])
            except AttributeError:
                # If validate_samples doesn't exist, just check bounds and land
                valid_batch = []
                for sample in samples:
                    lat, lon = sample[0], sample[1]
                    if (self.bounds.min_lat <= lat <= self.bounds.max_lat and
                        self.bounds.min_lon <= lon <= self.bounds.max_lon and
                        not self.is_on_land(lat, lon)):
                        valid_batch.append(sample)
                
                if valid_batch:
                    valid_samples = np.vstack([valid_samples, np.array(valid_batch)])
                    
        return valid_samples[:count, :]

    def validate_samples(self, samples, polygons, lat_min, lat_max, lon_min, lon_max):
        """
        Validate samples to ensure they're in water and within bounds.
        This is a placeholder - implement according to your existing validation logic.
        """
        valid_samples = []
        for sample in samples:
            lat, lon = sample[0], sample[1]
            if (lat_min <= lat <= lat_max and 
                lon_min <= lon <= lon_max and 
                not self.is_on_land(lat, lon)):
                valid_samples.append(sample)
        return np.array(valid_samples) if valid_samples else np.empty((0, 5))

    def is_on_land(self, lat: float, lon: float) -> bool:
        """Check if coordinates are on land."""
        from shapely.geometry import Point, Polygon, MultiPolygon
        point = Point(lon, lat)
        for geom in self.shoreline_polygons:
            if isinstance(geom, MultiPolygon):
                if any(poly.contains(point) for poly in geom.geoms):
                    return True
            elif isinstance(geom, Polygon) and geom.contains(point):
                return True
        return False

    def prompt_float_in_range(self, prompt: str, min_val: float, max_val: float) -> float:
        """Prompt user for float value within range."""
        while True:
            val = prompt_float(prompt)
            if min_val <= val <= max_val:
                return val
            print(f"Error: value must be between {min_val} and {max_val}")

    def get_valid_location(self) -> Tuple[float, float]:
        """Get valid water location from user."""
        while True:
            lat = self.prompt_float_in_range("Latitude", self.bounds.min_lat, self.bounds.max_lat)
            lon = self.prompt_float_in_range("Longitude", self.bounds.min_lon, self.bounds.max_lon)
            if self.is_on_land(lat, lon):
                print("Error: location is on land – choose water.")
                continue
            return lat, lon

    def show_history(self, species: int):
        """Show history for a species."""
        if species in self.history and self.history[species]:
            print(f"\n⚑ Previous mammals for species {species}")
            for n, rec in self.history[species].items():
                status = "Active" if rec['end'] is None else "Terminated"
                print(f"  • {n}: start={rec['start']} end={rec['end']} ({status})")
            print()

    def initialise(self, species: int = 0):
        """Create and schedule a new mammal."""
        # Get mammal parameters
        if input("Sample from CSV/KDE models? (y/n): ").strip().lower() == "y":
            lat, lon, depth = self.sample_from_csv_and_kde(species)
            heading = np.random.uniform(-180, 180)
            speed = np.random.uniform(1, 4)
            print(f"Sampled: lat={lat:.4f} lon={lon:.4f} depth={depth:.1f} "
                f"heading={heading:.1f} speed={speed:.1f}")
        else:
            lat, lon = self.get_valid_location()
            depth = self.prompt_float_in_range("Depth (m)", 0, 10)
            heading = self.prompt_float_in_range("Heading (deg)", -180, 180)
            speed = self.prompt_float_in_range("Speed (m/s)", 0.0, 5.0)

        start_clock = int(self.prompt_float_in_range(
            "Start-clock (min)", self.watcher.simulation_time, self.watcher.eta))
        end_clock = int(self.prompt_float_in_range(
            "End-clock (min)", start_clock, self.watcher.eta))

        # Generate name and create config
        node_name = self.next_node_name(species)
        topic = f"{node_name}_state"
        
        # Store in history
        self.history.setdefault(species, {})[node_name] = {
            "start": start_clock,
            "end": end_clock,
        }

        # Export YAML configuration
        self.export_mammal_yaml(node_name, lat, lon, depth, heading, speed, topic, start_clock, end_clock)

        # Create mammal config for scheduler
        from rclpy.parameter import Parameter
        mammal_config = {
            'parameters': [
                Parameter('latitude', value=lat),
                Parameter('longitude', value=lon),
                Parameter('depth', value=depth),
                Parameter('heading', value=heading),
                Parameter('speed', value=speed),
                Parameter('topic', value=topic),
                Parameter('bag_dir', value=self.bag_dir),
                Parameter('case_dir', value=str(self.case_dir)),
            ],
            'shared_assets': shared_assets
        }

        # Schedule with the simple scheduler
        self.scheduler.schedule_mammal(node_name, start_clock, end_clock, mammal_config)
        print(f"🕒 {node_name} scheduled: start={start_clock} end={end_clock} min")

    def terminate(self, species: int = 0):
        """Immediately terminate an active mammal."""
        active_mammals = self.scheduler.get_active_mammals(species)
        
        if not active_mammals:
            print(f"No active mammals for species {species}.")
            return

        print("Active mammals:")
        for i, name in enumerate(active_mammals, 1):
            print(f"  {i}. {name}")

        while True:
            try:
                selection = int(input("Select mammal to terminate (number): ")) - 1
                if 0 <= selection < len(active_mammals):
                    chosen = active_mammals[selection]
                    break
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a number.")

        # Terminate immediately
        self.scheduler.force_terminate(chosen, self.watcher.simulation_time)
        
        # Update history
        for species_records in self.history.values():
            if chosen in species_records:
                species_records[chosen]['end'] = self.watcher.simulation_time
                break

        print(f"✔ {chosen} terminated immediately")

    def export_mammal_yaml(self, node_name: str, lat: float, lon: float, depth: float, 
                      heading: float, speed: float, topic: str, start_clock: int, end_clock: int):
        """Export mammal configuration to YAML file."""
        import yaml
        config = {
            node_name: {
                'ros__parameters': {
                    'depth': depth,
                    'heading': heading,
                    'latitude': lat,
                    'longitude': lon,
                    'speed': speed,
                    'topic': topic
                },
                'simulation__parameters': {
                    'start_min': start_clock,
                    'end_min': end_clock
                }
            }
        }
        
        yaml_file = self.case_dir / f"{node_name}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"📄 Exported {yaml_file}")

    def update(self):
        """Update the scheduler - call this regularly."""
        if self.watcher.simulation_time is not None:
            self.scheduler.update(self.watcher.simulation_time)