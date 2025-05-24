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
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from shapely.geometry import Point, Polygon, MultiPolygon
import subprocess
import re
import heapq, itertools
from typing import Callable, Optional

#################################################################
############################ MUTE V1 ############################


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

        self._triggers: list[tuple[int, int, Callable[[], None]]] = []  # min-heap
        self._seq = itertools.count()                                   # tie-breaker

        self.create_subscription(Clock, "/clock", self.clock_cb, 10)
        self.create_subscription(Clock, "/ETA",   self.eta_cb,   10)

    # ─────────── trigger support ────────────────────────────────────────
    def add_trigger(self, t: int, callback: Callable[[], None]) -> None:
        """
        Schedule *callback* to fire once when /clock >= ``t`` (seconds).
        Fires immediately if that time is already past.
        """
        if self.simulation_time is not None and self.simulation_time >= t:
            callback()
        else:
            heapq.heappush(self._triggers, (t, next(self._seq), callback))

    # ─────────── ROS callbacks ──────────────────────────────────────────
    def clock_cb(self, msg: Clock):
        self.simulation_time = msg.clock.sec
        # dispatch any triggers that are now due
        while self._triggers and self._triggers[0][0] <= self.simulation_time:
            _, _, cb = heapq.heappop(self._triggers)
            try:
                cb()
            except Exception as exc:
                self.get_logger().error(f"Trigger callback failed: {exc}")

    def eta_cb(self, msg: Clock):
        self.eta = msg.clock.sec

    # convenience
    def ready(self) -> bool:
        return self.simulation_time is not None and self.eta is not None

    def behind_eta(self) -> bool:
        assert self.ready()
        return self.simulation_time < self.eta




class MammalManager:
    """
    Interactively create / terminate mammal nodes for a case-study run.

    • initialise()  -> schedules a launch of the mammal node
    • terminate()   -> stops a running mammal node and stamps end_clock
    """

    def __init__(self, case_dir: Path, kde_dir: Path,
                 watcher: TimeWatcher, bounds, TL_model_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/RBF/model'):
        self.case_dir = case_dir
        self.kde_dir = kde_dir
        self.watcher = watcher          # gives .simulation_time  &  .eta
        self.bounds = bounds            # dataclass with min/max lat & lon
        self.history: dict[int, dict[str, dict[str, int | None]]] = {}
        self.TL_model_path = TL_model_path
        self.bag_dir = str(self.case_dir)
        self._processes: dict[str, subprocess.Popen] = {} # To keep track of the mammal node execution for killing

        try:
            self.src_dir = next(p for p in case_dir.parents if p.name == "src")
        except StopIteration:
            raise RuntimeError("Could not locate the enclosing <workspace>/src folder")

        # shoreline polygons for land-/water tests
        shoreline_paths = self.src_dir / "environment" / "shoreline_data"
        shoreline = Polyshp(shoreline_paths)
        self.shoreline_polygons = shoreline.get_polygons(
            self.bounds.min_lat, self.bounds.max_lat,
            self.bounds.min_lon, self.bounds.max_lon,
        )

    def next_node_name(self,species: int) -> str:
        """
        Return the next unique mammal label *using in-memory history* rather than
        the filesystem.  Pattern remains ``m<species>_<idx>``.
        """
        records = self.history.get(species, {})
        if not records:                                # first mammal of this species
            return f"m{species}_1"

        # pull numeric suffixes already used → find the next integer
        used = []
        for name in records.keys():                    # e.g. "m3_2"
            try:
                used.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue                               # skip malformed

        next_idx = (max(used) if used else 0) + 1
        return f"m{species}_{next_idx}"

    def load_kde_models(self, kde_dir: Path, species: int = 0):
        pop_file = kde_dir / "population_data" / f"kde_model_species_{species}.pickle"
        dep_file = kde_dir / "depth_data" / f"kde_model_species_{species}.pickle"
        if not (pop_file.exists() and dep_file.exists()):
            raise FileNotFoundError(f"Missing KDE models for species {species}")
        with open(pop_file, "rb") as f:
            kde_pop = pickle.load(f)
        with open(dep_file, "rb") as f:
            kde_dep = pickle.load(f)
        return kde_pop, kde_dep

    def sample_from_kde(self, kde_pop, kde_dep) -> Tuple[float, float, float]:
        loc = kde_pop.sample(1).reshape(-1)
        loc[0:2] = np.degrees(loc[0:2])          # rad → deg
        depth = float(kde_dep.sample(1))
        return float(loc[0]), float(loc[1]), depth

    def is_on_land(self, lat: float, lon: float) -> bool:
        point = Point(lon, lat)
        for geom in self.shoreline_polygons:
            if isinstance(geom, MultiPolygon):
                if any(poly.contains(point) for poly in geom.geoms):
                    return True
            elif isinstance(geom, Polygon) and geom.contains(point):
                return True
        return False

    def prompt_float_in_range(self, prompt: str,
                              min_val: float, max_val: float) -> float:
        while True:
            val = prompt_float(prompt)
            if min_val <= val <= max_val:
                return val
            print(f"Error: value must be between {min_val} and {max_val}")

    def get_valid_location(self) -> Tuple[float, float]:
        while True:
            lat = self.prompt_float_in_range("Latitude",
                                             self.bounds.min_lat, self.bounds.max_lat)
            lon = self.prompt_float_in_range("Longitude",
                                             self.bounds.min_lon, self.bounds.max_lon)
            if self.is_on_land(lat, lon):
                print("Error: location is on land – choose water.")
                continue
            return lat, lon
    
    def show_history(self,species: int):
        # >>> show existing history for this species
        if species in self.history and self.history[species]:
            print("\n⚑ Previous mammals for species", species)
            for n, rec in self.history[species].items():
                print(f"  • {n}: start={rec['start']}  end={rec['end']}")
            print()

    # ─────────────────────────────────────────────────────────────────────
    # MAIN: create a new mammal node (+ deferred launch)
    # ─────────────────────────────────────────────────────────────────────
    def initialise(self, species: int = 0):
        """
        • Ask user for species id, location/sampling choice, start_clock, etc.  
        • Schedule python-launch once watcher.simulation_time == start_clock
        """

        # -- location / depth / heading / speed
        if input("Sample from KDE models? (y/n): ").strip().lower() == "y":
            kde_pop, kde_dep = self.load_kde_models(self.kde_dir, species)
            lat, lon, depth = self.sample_from_kde(kde_pop, kde_dep)
            heading = np.random.uniform(-180, 180)
            speed = np.random.uniform(1, 4)
            start_clock = int(self.prompt_float_in_range("Start-clock (min)",  self.watcher.simulation_time, self.watcher.eta))
            print(f"Sampled:  lat={lat:.4f}  lon={lon:.4f}  depth={depth:.1f}  "
                  f"heading={heading:.1f}  speed={speed:.1f}")
        else:
            lat, lon = self.get_valid_location()
            depth   = self.prompt_float_in_range("Depth (m)",     0, 10)
            heading = self.prompt_float_in_range("Heading (deg)", -180, 180)
            speed   = self.prompt_float_in_range("Speed (m/s)",   0.0, 5.0)
            start_clock = int(self.prompt_float_in_range("Start-clock (min)",  self.watcher.simulation_time, self.watcher.eta))

        # -- naming & topic
        node_name = self.next_node_name(species)
        topic     = f"{node_name}_state"
        self.history.setdefault(species, {})[node_name] = {
            "start": start_clock,
            "end":   None,
        }

        # -- schedule a launch when the clock reaches `start_clock`
        def _deferred_launch():
            cmd = [
                "ros2", "run", "mammals", "mammal_exe",
                "--ros-args",
                "-r", f"__node:={node_name}",
                # ─── all parameters inline ───────────────────────────
                "-p", f"latitude:={lat}",
                "-p", f"longitude:={lon}",
                "-p", f"depth:={depth}",
                "-p", f"heading:={heading}",
                "-p", f"speed:={speed}",
                "-p", f"topic:={topic}",
                "-p", f"bag_dir:={self.bag_dir}",
                "-p", f"case_dir:={self.case_dir}",
                "-p", f"TL_model_path:={self.TL_model_path}",
            ]
            print(f"🚀  Launching {node_name} at t={self.watcher.simulation_time} min …")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid     # <<< start a new session (=process group)
            )
            self._processes[node_name] = proc

        # register with watcher
        self.watcher.add_trigger(start_clock, _deferred_launch)
        print(f"🕒  Node will launch when /clock == {start_clock} min")

    def list_active(self) -> list[str]:
        # If you want “active” = still-running OS processes:
        return [name for name, proc in self._processes.items()
                if proc.poll() is None]

    # def terminate(self, species: int = 0):
        
        # active = self.list_active()
        # if not active:
        #     print("No active mammals found.")
        #     return

        # print("\nActive mammals:")
        # for idx, name in enumerate(active, 1):
        #     print(f"  {idx}. {name}")

        # # -- choose which to terminate
        # choice = int(self.prompt_float_in_range("Choice", 1, len(active)))
        # chosen = active[int(choice) - 1]

        # # -- stamp end_clock in YAML
        # yaml_path = self.case_dir / f"{chosen}_params.yaml"
        # if yaml_path.exists():
        #     with yaml_path.open() as f:
        #         data = yaml.safe_load(f)
        #     data["sim_parameters"]["end_clock"] = self.watcher.simulation_time
        #     with yaml_path.open("w") as f:
        #         yaml.safe_dump(data, f, sort_keys=False)
        #     print(f"✔  Patched end_clock={self.watcher.simulation_time} in {yaml_path.name}")

        # # -- politely ask ROS2 to kill the node
        # proc = self._processes.pop(chosen)
        # print(f"🚨  Terminating node {chosen} …")
        # pgid = os.getpgid(proc.pid)            # process-group id
        # os.killpg(pgid, signal.SIGINT)         # polite shutdown (rclpy catches SIGINT)

        # try:
        #     proc.wait(timeout=5)
        #     print("✔  Node exited cleanly.")
        # except subprocess.TimeoutExpired:
        #     print("⚠️  Still alive after SIGINT → sending SIGTERM …")
        #     os.killpg(pgid, signal.SIGTERM)
        #     try:
        #         proc.wait(timeout=3)
        #     except subprocess.TimeoutExpired:
        #         print("💥  Forcing SIGKILL.")
        #         os.killpg(pgid, signal.SIGKILL)
        #         proc.wait()

    def prompt_valid_idx(self, active_mammals: list[tuple[int, str]]) -> int:
        """Prompt user to select a valid mammal index from the list."""
        valid_indices = {idx for idx, _ in active_mammals}
        while True:
            for idx, name in active_mammals:
                print(f"  {idx}. {name}")
            selection = int(prompt_float("Select mammal idx from the list to terminate"))
            if selection in valid_indices:
                return selection
            else:
                print("Invalid selection. Please choose a valid index from the list.")

    def terminate(self, species: int = 0):
        """
        • Ask user which mammal to terminate
        • Get end_clock from user
        • Schedule termination when watcher.simulation_time == end_clock
        """
        # Get species history
        species_history = self.history.get(species, {})
        
        if not species_history:
            print(f"No mammals for species {species} found.")
            return
        
        # Get active mammals (ones with no end time)
        active_mammals = [(i+1, name) for i, (name, info) in 
                        enumerate(species_history.items()) if info["end"] is None]
        
        if not active_mammals:
            print(f"No active mammals for species {species}.")
            return
        
        
        # Get user selection according to order of history
        selection = self.prompt_valid_idx(active_mammals)
        chosen_idx = selection - 1
        chosen = active_mammals[chosen_idx][1]
        
        # Get end_clock from user
        end_clock = int(self.prompt_float_in_range(
            f"End-clock for {chosen} (min)",
            self.watcher.simulation_time,
            self.watcher.eta
        ))
        
        # Update history with end time
        species_history[chosen]["end"] = end_clock
        
        # Schedule termination with the TimeWatcher
        def _deferred_termination():
            # Make sure the process is still in our dict (safety check)
            if chosen in self._processes:
                # -- politely ask ROS2 to kill the node
                proc = self._processes.pop(chosen)
                print(f"🚨  Terminating node {chosen} at t={self.watcher.simulation_time} min …")
                pgid = os.getpgid(proc.pid)            # process-group id
                os.killpg(pgid, signal.SIGINT)         # polite shutdown (rclpy catches SIGINT)

                try:
                    proc.wait(timeout=5)
                    print("✔  Node exited cleanly.")
                except subprocess.TimeoutExpired:
                    print("⚠️  Still alive after SIGINT → sending SIGTERM …")
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        print("💥  Forcing SIGKILL.")
                        os.killpg(pgid, signal.SIGKILL)
                        proc.wait()
        
        # Register the termination trigger with the watcher
        self.watcher.add_trigger(end_clock, _deferred_termination)
        print(f"🕒  Node {chosen} will be terminated when /clock == {end_clock} min")