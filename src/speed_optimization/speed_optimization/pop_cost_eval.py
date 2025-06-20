#!/usr/bin/env python3
"""
Population Cost Evaluator
Imports a population from CSV and computes cost function for each speed profile.
"""
# Add horizontal colorbar at the top of the second subplot
    # Create a mappable object for the colorbar
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml
import sys
from pathlib import Path as Path2
from custom_interfaces.msg import State, HelperPosition, HeadingAngle
from speed_optimization.py_pymoo_config import SpeedOptimizationProblem
from speed_optimization.py_util import calculate_path_length
from models.jomopans_echo.model import jomopans_model_matrix
from environment.py_polyshp import Polyshp
import re
import glob

import os

def find_src_dir(start_path: Path2) -> Path2:
    """Search upward from the start_path until a directory containing a 'src' folder is found."""
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not find a 'src' directory in any parent directories.")

def load_mammal_states(src_dir: Path2, mammal_case: str) -> np.ndarray:
    """
    Load mammal states from YAML files in the specified case directory.
    
    Args:
        src_dir: Source directory path
        mammal_case: Case study name for mammals (e.g., 'Case_A', 'Case_D')
    
    Returns:
        numpy array of State objects
    """
    # Define the path to mammal case studies
    mammal_path = src_dir / 'mammals' / 'case_studies' / mammal_case
    
    if not mammal_path.exists():
        print(f"Warning: Mammal case directory does not exist: {mammal_path}")
        return np.empty((0,), dtype=State)
    
    # Find all YAML files in the directory
    yaml_files = list(mammal_path.glob('*.yaml')) + list(mammal_path.glob('*.yml'))
    
    print(f"Number of mammals found: {len(yaml_files)}")
    
    if len(yaml_files) == 0:
        print(f"No YAML files found in {mammal_path}")
        return np.empty((0,), dtype=State)
    
    mammal_states = []
    
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file)
            
            # Extract mammal data from the YAML structure
            # Assuming the structure has one main key per file
            for mammal_id, mammal_data in data.items():
                if 'ros__parameters' in mammal_data:
                    params = mammal_data['ros__parameters']
                    
                    # Create State object
                    state = State()
                    
                    # Create HelperPosition
                    position = HelperPosition()
                    position.latitude = float(params.get('latitude', 0.0))
                    position.longitude = float(params.get('longitude', 0.0))
                    position.depth = float(params.get('depth', 0.0))
                    
                    # Create HeadingAngle
                    angle = HeadingAngle()
                    angle.heading = float(params.get('heading', 0.0))
                    
                    # Set State properties
                    state.position = position
                    state.angle = angle
                    state.speed = float(params.get('speed', 0.0))

                    print(state)
                    
                    mammal_states.append(state)
                    
                    print(f"Loaded mammal {mammal_id}: "
                          f"Lat={position.latitude:.6f}, Lon={position.longitude:.6f}, "
                          f"Depth={position.depth:.3f}m, Heading={angle.heading:.1f}°, "
                          f"Speed={state.speed:.2f}m/s")
                
        except Exception as e:
            print(f"Error loading mammal data from {yaml_file}: {e}")
            continue
    
    # Convert to numpy array of State objects
    mammal_states_array = np.array(mammal_states, dtype=State)
    print(f"Successfully loaded {len(mammal_states_array)} mammal states")
    
    return mammal_states_array


class PopulationCostEvaluator:
    def __init__(self, case_study='Case_1', mammal_current_states=None, optimization_problem=None):
        """Initialize the cost evaluator with case study parameters."""
        # Find source directory
        self.src_dir = find_src_dir(Path2(__file__))
        
        # Configuration parameters
        self.case_study = case_study
        self.TL_model_path = str(self.src_dir / 'noise_propagation_model' / 'noise_propagation_model' / 'models' / 'RBF' / 'model_single_freq')
        self.acc_max = 36.60  # knots/h
        self.acc_min = -33.00  # knots/h
        self.timestep = (3/60)  # hrs
        self.obj_computation_step = 1
        
        # Ship and voyage parameters
        self.min_v = None
        self.max_v = None
        self.ship_type = None
        self.ship_length = None
        self.waypoints = []
        self.path_length = 0.0
        self.remaining_time = None
        self.ETA = None
        
        # Frequency array
        self.freq = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])
        
        # Set mammal states
        if mammal_current_states is not None:
            self.mammal_current_states = mammal_current_states
        else:
            self.mammal_current_states = np.empty((0,), dtype=State)
        
        # Load case study data if optimization problem not provided
        if optimization_problem is None:
            self.load_voyage_definition()
            self.load_shoreline_data()
        else:
            # Use pre-loaded optimization problem
            self.problem = optimization_problem
            # Extract necessary parameters from the problem
            self.min_v = optimization_problem.xl
            self.max_v = optimization_problem.xu
            self.waypoints = optimization_problem.waypoints
            self.ship_type = optimization_problem.ship_type
            self.ship_length = optimization_problem.ship_length
            self.shoreline_polygons = optimization_problem.shoreline_polygons
    
    def load_voyage_definition(self):
        """Load voyage definition and ship parameters from case study files."""
        try:
            csv_file_path = '/workspaces/MUTE_v1/imgs/Convergence plot/optimal_path.csv'
            yaml_file_path = self.src_dir / 'ship' / 'case_studies' / self.case_study / 'ship.yaml'
            
            # Load voyage data
            voyage_data = pd.read_csv(csv_file_path, header=None)
            voyage_data = voyage_data.iloc[1:].reset_index(drop=True)
            
            # Load ship data
            with open(yaml_file_path, 'r') as yamlfile:
                ship_data = yaml.safe_load(yamlfile)
            
            if 'ship_info' in ship_data:
                ship_info = ship_data['ship_info']
                self.ship_type = ship_info.get('ship_type', 'Unknown')
                self.ship_length = ship_info.get('ship_length', 0.0)
                self.min_v = ship_info.get('min_v', 0.0)
                self.max_v = ship_info.get('max_v', 0.0)
                print(f"Ship Info loaded: Type={self.ship_type}, Length={self.ship_length} feet, Min Speed={self.min_v}, Max Speed={self.max_v}")
            else:
                raise ValueError("Missing 'ship_info' field in YAML file.")
            
            # Create waypoints from voyage data (assuming columns are: lat, lon, ...)
            self.waypoints = []
            for _, row in voyage_data.iterrows():
                waypoint = HelperPosition()
                waypoint.longitude = float(row.iloc[1])  # Assuming longitude is in column 1
                waypoint.latitude = float(row.iloc[0])   # Assuming latitude is in column 2
                self.waypoints.append(waypoint)
            
            self.path_length = calculate_path_length(self.waypoints)
            print(f"Path length: {self.path_length} NM")
            
        except Exception as e:
            print(f"Error loading voyage definition: {e}")
            sys.exit(1)
    
    def load_shoreline_data(self):
        """Load shoreline polygon data."""
        try:
            # Set some default bounds - you may need to adjust these
            lat_min, lat_max = 40.0, 50.0
            lon_min, lon_max = -130.0, -120.0
            
            shoreline_paths = self.src_dir / "environment" / "shoreline_data"
            shoreline = Polyshp(shoreline_paths)
            self.shoreline_polygons = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)
            print("Shoreline data loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load shoreline data: {e}")
            self.shoreline_polygons = None
    
    def load_population_from_csv(self, csv_path):
        """Load population data from CSV file."""
        try:
            population_df = pd.read_csv(csv_path)
            population = population_df.to_numpy()
            print(f"Loaded population with shape: {population.shape}")
            return population
        except Exception as e:
            print(f"Error loading population from CSV: {e}")
            sys.exit(1)
    
    def create_optimization_problem(self, n_var):
        """Create the optimization problem instance."""
        extra_time = 0.0
        
        problem = SpeedOptimizationProblem(
            n_var=n_var,
            n_obj=1,
            acc_min=self.acc_min,
            acc_max=self.acc_max,
            xl=self.min_v,
            xu=self.max_v,
            waypoints=self.waypoints,
            timestep=self.timestep,
            mammal_states=self.mammal_current_states,
            shoreline_polygons=self.shoreline_polygons,
            freq=self.freq,
            ship_type=self.ship_type,
            ship_length=self.ship_length,
            extra_time=extra_time,
            TL_model_path=self.TL_model_path,
            obj_computation_step=self.obj_computation_step
        )
        
        return problem
    
    def evaluate_population_costs(self, population):
        """Evaluate cost function for each individual in the population."""
        print("Starting cost evaluation for population...")
        
        if not hasattr(self, 'problem'):
            n_var = population.shape[1]
            self.problem = self.create_optimization_problem(n_var)
        
        cost = self.problem.speed_objective.cost_function_noise(population, self.timestep)
        return cost
    
    def print_results(self, costs):
        """Print summary statistics of the cost evaluation."""
        print("\n" + "="*60)
        print("COST EVALUATION RESULTS")
        print("="*60)
        print(f"Population size: {len(costs)}")
        print(f"Number of variables per individual: {self.population.shape[1]}")
        print(f"Minimum cost: {np.min(costs):.6f}")
        print(f"Maximum cost: {np.max(costs):.6f}")
        print(f"Mean cost: {np.mean(costs):.6f}")
        print(f"Standard deviation: {np.std(costs):.6f}")
        print(f"Median cost: {np.median(costs):.6f}")
        
        # Find best individual
        best_idx = np.argmin(costs)
        print(f"\nBest individual (index {best_idx}):")
        print(f"Cost: {costs[best_idx]:.6f}")
        self.best_profile = self.population[best_idx]

    def get_best_individual(self, population, costs):
        """Get the best individual and its cost from the population."""
        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        best_profile = population[best_idx]
        return best_profile, best_cost, best_idx


def collect_population_csvs(folder_path: str) -> dict:
    """
    Collect all CSV files that start with 'optimal_population_' in the specified folder.
    
    Args:
        folder_path: Path to folder containing population CSV files
        
    Returns:
        Dictionary with generation number as key and full file path as value
    """
    csv_files = {}
    
    for filename in os.listdir(folder_path):
        if filename.startswith('optimal_population_') and filename.endswith('.csv'):
            # Extract generation number from filename
            match = re.search(r'optimal_population_(\d+)\.csv', filename)
            if match:
                gen_number = int(match.group(1))
                full_path = os.path.join(folder_path, filename)
                csv_files[gen_number] = full_path
    
    return csv_files


def process_all_populations(folder_path: str, case_study: str, mammal_case: str):
    """
    Process all population CSV files in the folder and extract best profiles and costs.
    
    Args:
        folder_path: Path to folder containing population CSV files
        case_study: Case study name
        mammal_case: Mammal case name
        
    Returns:
        Dictionary with generation data including best costs and profiles
    """
    # Find source directory and load mammal states once
    src_dir = find_src_dir(Path2(__file__))
    print(f"Loading mammal states from case: {mammal_case}")
    mammal_states = load_mammal_states(src_dir, mammal_case)
    
    # Create evaluator and optimization problem once
    evaluator = PopulationCostEvaluator(
        case_study=case_study,
        mammal_current_states=mammal_states
    )
    
    # Collect all CSV files
    csv_files = collect_population_csvs(folder_path)
    print(f"Found {len(csv_files)} population CSV files")
    
    # Initialize results storage
    generation_data = {}
    
    # Process each generation
    for gen_num in sorted(csv_files.keys()):
        csv_path = csv_files[gen_num]
        print(f"\nProcessing Generation {gen_num}...")
        
        # Load population
        population = evaluator.load_population_from_csv(csv_path)
        
        # Create optimization problem for first generation only
        if not hasattr(evaluator, 'problem'):
            n_var = population.shape[1]
            evaluator.problem = evaluator.create_optimization_problem(n_var)
        
        # Evaluate costs
        costs = evaluator.evaluate_population_costs(population)
        
        # Get best individual
        best_profile, best_cost, best_idx = evaluator.get_best_individual(population, costs)
        
        # Store results
        generation_data[gen_num] = {
            'best_cost': best_cost,
            'best_profile': best_profile,
            'best_idx': best_idx,
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs)
        }
        
        print(f"Generation {gen_num}: Best cost = {best_cost:.6f}, Mean cost = {np.mean(costs):.6f}")
    
    return generation_data, evaluator

def plot_convergence_and_profiles(generation_data: dict, evaluator: PopulationCostEvaluator, 
                                output_filename: str = "convergence_analysis.png"):
    """
    Create two subplots: convergence plot and overlayed speed profiles.
    
    Args:
        generation_data: Dictionary containing generation results
        evaluator: PopulationCostEvaluator instance
        output_filename: Output filename for the plot
    """
    # Set font to Times New Roman and set all font sizes to 18
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    
    # Create figure with larger size for high resolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Extract data for plotting
    generations = sorted(generation_data.keys())
    best_costs = [generation_data[gen]['best_cost'] for gen in generations]
    
    # Plot 1: Convergence of best cost
    ax1.plot(generations, best_costs, 'bo-', linewidth=3, markersize=8)
    ax1.set_xlabel('Number of generation', fontsize=18)
    ax1.set_ylabel('SEL', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(generations), max(generations))
    ax1.tick_params(labelsize=18)
    
    # Plot 2: Overlayed speed profiles with colorbar
    # Convert time to minutes
    time_axis = [(evaluator.timestep * i + evaluator.timestep / 2) * 60
                 for i in range(len(generation_data[generations[0]]['best_profile']))]
    
    # Create colormap for different generations
    colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))
    
    for i, gen in enumerate(generations):
        best_profile = generation_data[gen]['best_profile']
        ax2.plot(time_axis, best_profile, color=colors[i], linewidth=2, 
                alpha=0.8, label=f'Gen {gen}')
    
    ax2.set_xlabel('Simulated voyage time (minutes)', fontsize=18)
    ax2.set_ylabel('Ship speed (knots)', fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(time_axis))
    ax2.set_ylim(8, 16)
    ax2.tick_params(labelsize=18)
    
    # Set custom tick intervals for better resolution
    # Time axis: every 200 minutes
    time_ticks = np.arange(0, max(time_axis), 200)
    ax2.set_xticks(time_ticks)
    
    # Speed axis: every 2 units
    speed_ticks = np.arange(8, 17, 2)
    ax2.set_yticks(speed_ticks)
    
    # Apply tight_layout first to finalize subplot positions
    plt.tight_layout()
    
    # NOW get the exact position of ax2 after layout adjustment
    pos2 = ax2.get_position()
    
    # Normalize the generation numbers
    norm = mcolors.Normalize(vmin=min(generations), vmax=max(generations))
    mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
    mappable.set_array([])
    
    # Create colorbar axes above the second subplot - match width exactly
    # Use the exact x0 and width from ax2's final position
    cbar_ax = fig.add_axes([pos2.x0, pos2.y1 + 0.05, pos2.width, 0.03])
    cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Generation Number', fontsize=18, labelpad=15)
    cbar.ax.tick_params(labelsize=18)
    
    # Position the colorbar label and ticks above the colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # Set colorbar ticks to show actual generation numbers
    if len(generations) <= 10:
        cbar.set_ticks(generations)
    else:
        # If too many generations, show subset
        tick_indices = np.linspace(0, len(generations)-1, 6, dtype=int)
        tick_values = [generations[i] for i in tick_indices]
        cbar.set_ticks(tick_values)
    
    # Create output directory
    os.makedirs('imgs', exist_ok=True)
    
    # Adjust layout to accommodate colorbar (reduce top margin)
    plt.subplots_adjust(top=0.85)
    
    # Change extension to PDF and save at high resolution
    pdf_filename = output_filename.replace('.png', '.pdf')
    plt.savefig(f"imgs/Convergence plot/{pdf_filename}", dpi=600, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"Convergence analysis plot saved to imgs/Convergence plot/{pdf_filename}")

def main():
    """Main function to run the enhanced population cost evaluator."""
    # Define parameters
    case_study = 'Case_1'
    mammal_case = 'Case_D'
    
    # Define folder path containing population CSV files
    population_folder_path = "/workspaces/MUTE_v1/imgs/Convergence plot"
    
    # Process all populations
    generation_data, evaluator = process_all_populations(
        population_folder_path, case_study, mammal_case
    )
    
    # Create convergence and profile plots
    plot_convergence_and_profiles(generation_data, evaluator)
    
    # Print summary results
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS RESULTS")
    print("="*80)
    
    generations = sorted(generation_data.keys())
    for gen in generations:
        data = generation_data[gen]
        print(f"Generation {gen:4d}: Best Cost = {data['best_cost']:10.6f}, "
              f"Mean Cost = {data['mean_cost']:10.6f}, Std = {data['std_cost']:8.6f}")
    
    # Save detailed results to CSV
    results_data = []
    for gen in generations:
        data = generation_data[gen]
        results_data.append({
            'generation': gen,
            'best_cost': data['best_cost'],
            'mean_cost': data['mean_cost'],
            'std_cost': data['std_cost']
        })
    
#     results_df = pd.DataFrame(results_data)
#     output_path = 'convergence_analysis_results.csv'
#     results_df.to_csv(output_path, index=False)
#     print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()