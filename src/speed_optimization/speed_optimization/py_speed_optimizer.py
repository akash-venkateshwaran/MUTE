import rclpy
from rclpy.node import Node
from custom_interfaces.msg import Path, ShipInfo
from pyproj import Geod
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling

import yaml
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from custom_interfaces.srv import BoundsService


import re
import os

from rosgraph_msgs.msg import Clock


from environment.py_polyshp import Polyshp

from matplotlib.path import Path
from matplotlib.patches import PathPatch


from speed_optimization.py_pymoo_config import SpeedOptimizationProblem, SpeedOptimizationCallback, AdaptiveMutation, AdaptiveCrossover
from custom_interfaces.msg import State, HelperPosition, HeadingAngle, Path, Speed
from speed_optimization.py_util import calculate_path_length

from models.jomopans_echo.model import jomopans_model_matrix
from threading import Thread

from pathlib import Path as Path2

def find_src_dir(start_path: Path2) -> Path2:
    """
    Search upward from the start_path until a directory containing a 'src' folder is found.
    Returns the path to that 'src' folder.
    """
    current = start_path.resolve()
    while current != current.parent:  # stop at filesystem root
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not find a 'src' directory in any parent directories.")

src_dir = find_src_dir(Path2(__file__))


# Define the geodesic object for WGS84
GEODESIC = Geod(ellps="WGS84")

class SpeedOptimizer(Node):
    def __init__(self):
        super().__init__('speed_optimizer_node')
        self.waypoints = [] # List of HelperPositions
        self.case_study = 'Case_1'
        self.skip_process = None
        self.TL_model_path = str(src_dir / 'noise_propagation_model' / 'noise_propagation_model' / 'models' / 'RBF' / 'model_single_freq')
        self.previous_solution_path = str(src_dir / 'ship' / 'case_studies' / 'Case_1' / 'optimal_population_5000.csv')


        self.timestep = 0.15 # hrs (usually 0.15)
        self.ETA = None  # hrs
        self.simulation_time = 0.0  # hrs
        self.remaining_time = None # hrs

        self.min_v = None  # knots Note that if you set it to zero then log10(V/V_C) is not defined
        self.max_v = None # knots
        self.acc_max =  36.60 # knots/h
        self.acc_min =  -33.00 # knots/h
        self.path_length = 0.0  # NM
        self.n_waypoints = 0
        self.optimized_speed = [] # knots
        # self.waypoint_proximity = 0.0539957 # in NM = 100 m same as ship node

        self.verbose_logging = False
        self.has_received_waypoints = False

        # TODO Change freq defintion based on the transmission loss
        self.freq = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])
        self.ship_type = None
        self.ship_length = None

        self.solve_optimization_thread = None
        self.previous_solution_num_gen = 0
        self.previous_population = None
        self.num_gen = 5000

        self.load_voyage_definition(self.case_study)

        # Getting the shoreline:
        self.shoreline_polygons = None
        if not self.skip_process:
            self.get_the_shoreline()


        ######### Publishers:
        self.optimized_speed_publisher = self.create_publisher(Speed, 'optimized_speeds', 10)

        ######### Subscriptions:
        if not self.skip_process:
            self.mammal_current_states = np.empty((0,), dtype=State)
            

            self.subscription_ETA = self.create_subscription(
                Clock, 
                'ETA',
                self.ETA_callback,
                10)
            
            self.subscribe_to_mammal_topics()        

            self.subscription_path = self.create_subscription(
                Path,
                'ship_optimized_waypoints',
                self.ship_optimized_waypoints_callback,
                10)
            self.setup_logger()
            
            self.remaining_time = self.ETA       

            self.subscription_clock = self.create_subscription(
                Clock,
                '/clock',
                self.get_simulation_time,
                10)

            self.get_logger().info('Waiting for the "ship_optimized_waypoints" topic...')
            while not self.has_received_waypoints:
                rclpy.spin_once(self, timeout_sec=1.0)

        ######### Timer:
        self.timer = self.create_timer(1.0, self.all_publishers)


    def setup_logger(self):
        self.get_logger().info("Speed Optimizer Node is started!")

    def ETA_callback(self, msg):
        self.ETA = msg.clock.sec / 60.0 # min to hr

    def get_the_shoreline(self):
        """Initialize the bounds service client and get the bounds."""
        self.client = self.create_client(BoundsService, 'bounds_service')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('BoundsService not available, waiting again...')

        request = BoundsService.Request()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.lat_max = response.max_lat
            self.lat_min = response.min_lat
            self.lon_max = response.max_lon
            self.lon_min = response.min_lon
            self.get_logger().info(
                f"Received bounds: max_lat={self.lat_max}, min_lat={self.lat_min}, "
                f"max_lon={self.lon_max}, min_lon={self.lon_min}"
            )

            # Load shoreline polygons based on received bounds
            self.load_shoreline(self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        else:
            self.get_logger().error('BoundsService call failed')

    def load_shoreline(self, lat_min, lat_max, lon_min, lon_max):
        """Load shoreline data based on bounds."""
        shoreline_paths = src_dir / "environment" / "shoreline_data"
        shoreline = Polyshp(shoreline_paths)
        self.shoreline_polygons = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)

    def all_publishers(self):
            if self.previous_solution_num_gen >= self.num_gen or self.skip_process:
                msg = Speed()
                msg.opt_speeds = list(self.optimized_speed)  
                self.optimized_speed_publisher.publish(msg)

    def load_voyage_definition(self, case_path='Case_1'):
        try:
            csv_file_path = src_dir / 'path_planning' / 'case_studies' / case_path / 'case_study.csv'
            yaml_file_path = src_dir / 'ship' / 'case_studies' / case_path / 'ship.yaml'

            voyage_data = pd.read_csv(csv_file_path, header=None)
            
            if len(voyage_data.columns) < 4:
                self.get_logger().error("CSV file has insufficient columns. Expected at least 4 columns: time, longitude, latitude, etc.")
                return

            with open(yaml_file_path, 'r') as yamlfile:
                ship_data = yaml.safe_load(yamlfile)

            if 'ship_info' in ship_data:
                ship_info = ship_data['ship_info']
                self.ship_type = ship_info.get('ship_type', 'Unknown')
                self.ship_length = ship_info.get('ship_length', 0.0)
                self.min_v = ship_info.get('min_v', 0.0)
                self.max_v = ship_info.get('max_v', 0.0)
                self.get_logger().info(f"Ship Info loaded: Type={self.ship_type}, Length={self.ship_length} feet, Min Speed={self.min_v}, Max Speed={self.max_v}")
            else:
                self.get_logger().error("Missing 'ship_info' field in YAML file.")
                return

            self.get_logger().info("Voyage definition and ship info loaded from CSV and YAML.")

            print("\nChoose an option: ")
            print("(1) Run the case study")
            print("(2) Optimize the case study")
            print("(3) Import the previous pop and run the case study")

            user_choice = input("Enter your choice (1/2/3): ").strip()
            if user_choice == '1':
                self.skip_process = True
                self.optimized_speed = voyage_data.iloc[:, 1].tolist()  # Update optimized_speed with the 2nd column of CSV
                self.get_logger().info(f"Optimized {len(self.optimized_speed)} speeds in knots")
            elif user_choice == '2':
                self.skip_process = False
                self.get_logger().info("User chose to optimize the case study.")
            elif user_choice == '3':
                if not hasattr(self, 'previous_solution_path') or not os.path.exists(self.previous_solution_path):
                    self.get_logger().error("Previous population file does not exist.")
                    sys.exit("Terminating: Previous population file not found.")

                # Extract generation number from file name
                try:
                    self.previous_solution_num_gen = int(self.previous_solution_path.split('_')[-1].split('.')[0])
                except ValueError:
                    self.get_logger().error("Could not extract generation number from file name.")
                    sys.exit("Terminating: Invalid file naming format.")

                # Load the previous population as a 2D numpy array
                self.previous_population = pd.read_csv(self.previous_solution_path).to_numpy()
                self.get_logger().info(f"Loaded previous population from generation {self.previous_solution_num_gen} with shape {self.previous_population.shape}.")

            else:
                self.get_logger().warn("Invalid choice entered. Defaulting to running the case study.")
                self.skip_process = True

        except FileNotFoundError as e:
            self.get_logger().error(f"File not found: {e}")
            sys.exit("Terminating: Start state is outside the shipping lane polygons.")
        except pd.errors.EmptyDataError:
            self.get_logger().error("CSV file is empty or invalid.")
            sys.exit("Terminating: Start state is outside the shipping lane polygons.")
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file: {e}")
            sys.exit("Terminating: Start state is outside the shipping lane polygons.")
        except Exception as e:
            self.get_logger().error(f"Error loading voyage or ship data: {e}")
            sys.exit("Terminating: Start state is outside the shipping lane polygons.")


    def subscribe_to_mammal_topics(self):
        pattern = re.compile(r'^/m\d.*state$')
        while rclpy.ok():
            topics = self.get_topic_names_and_types()
            mammal_topics = [topic_name for topic_name, _ in topics if pattern.match(topic_name)]

            if mammal_topics:
                self.get_logger().info(f"Found {len(mammal_topics)} mammal topics: {mammal_topics}")
                
                user_input = input("Do you want to process these topics? (y/n): ").strip().lower()
                if user_input == 'y':
                    self.mammal_current_states = np.empty((len(mammal_topics),), dtype=State)  # Initialize the array with the correct size
                    for i, mammal_topic in enumerate(mammal_topics):
                        self.create_subscription(
                            State,
                            mammal_topic,
                            lambda msg, index=i: self.mammal_state_callback(msg, index),
                            10
                        )
                    break
                else:
                    self.get_logger().info("User chose not to process the topics. Retrying...")
            else:
                self.get_logger().warn("No mammal topics found. Retrying...")
            rclpy.spin_once(self, timeout_sec=1.0)


    def ship_optimized_waypoints_callback(self, msg):
        self.has_received_waypoints = True

        

        if any(state is None for state in self.mammal_current_states):
            self.get_logger().warn("Skipping optimization as all mammal information has not been received.")
            return
        
        if self.simulation_time is None or self.remaining_time is None:
                        self.get_logger().info(f"Simulation time ({self.simulation_time}) or remaining time ({self.remaining_time}) is None. Skipping")
        else:
            if self.solve_optimization_thread is not None and self.solve_optimization_thread.is_alive():
                self.get_logger().info("Solving")
            else:
                if calculate_path_length(msg.waypoints) != self.path_length:
                    self.waypoints = msg.waypoints  # Update the waypoints list
                    self.n_waypoints = len(self.waypoints)
                    self.path_length = calculate_path_length(self.waypoints)
                    if self.path_length / self.ETA > self.max_v:
                        self.get_logger().error(
                            f"The constraint won't be satisfied. "
                            f"Required speed: {self.path_length / self.remaining_time} kn, "
                            f"Max speed: {self.max_v} kn."
                        )
                    else:
                        if self.previous_solution_num_gen < self.num_gen: 
                                self.run_speed_optimization()
                        else:
                            self.get_logger().info(f"The number of generation already exceeded ({self.previous_solution_num_gen} >= {self.num_gen}). Skipping")
                else:
                    self.get_logger().info(f"Received the same path. Skipping ({calculate_path_length(msg.waypoints)} == {self.path_length})")

    def run_speed_optimization(self):
            try:
                self.get_logger().info(f"{self.n_waypoints} optimized waypoints received.")
                self.get_logger().info(f"Total dist to cover (NM): {self.path_length}")
                self.get_logger().info(f"Starting the optimization for num of gen: {self.num_gen - self.previous_solution_num_gen}")
                self.solve_optimization_thread = Thread(
                                target=self.optimize_speed,
                                kwargs={
                                    'total_gen': self.num_gen,
                                    'previous_gen': self.previous_solution_num_gen,
                                    'previous_pop': self.previous_population})
                self.solve_optimization_thread.start()
            except Exception as e:
                self.get_logger().error(f"Error starting solve_optimization_thread: {e}")


    def get_simulation_time(self, msg):
        if self.ETA is None:
            self.get_logger().info("Waiting for updating the ETA")
        else:
            self.simulation_time = msg.clock.sec / 60.0 # Note This is NOT in sec but in mins which is then coverted to hr
            self.remaining_time = self.ETA - (self.simulation_time)

            # self.get_logger().info(f"Simulated time: {(self.simulation_time):.2f} hrs, Remaining time: {self.remaining_time:.2f} hrs")

            # Apply remaining time constraint to your optimization problem
            if self.remaining_time < 0:
                self.get_logger().error("Simulation time exceeded the ETA, cannot continue optimization.")

    def optimize_speed(self, total_gen, previous_gen, previous_pop=None):
        if any(state is not None for state in self.mammal_current_states):

            self.get_logger().info("Performing speed optimization based on received waypoints.")
            self.get_logger().info(f"Number of variables: {int(self.remaining_time / self.timestep)}")
            n_var = int(self.remaining_time / self.timestep)
            extra_time = (self.remaining_time / self.timestep) - n_var  # The time not accounted due to divisibility

            problem = SpeedOptimizationProblem(
                n_var=n_var, n_obj=1, acc_min = self.acc_min, acc_max = self.acc_max,
                xl=self.min_v, xu=self.max_v, waypoints=self.waypoints,
                timestep=self.timestep, mammal_states=self.mammal_current_states,
                shoreline_polygons=self.shoreline_polygons, freq=self.freq,
                ship_type=self.ship_type, ship_length=self.ship_length,
                extra_time=extra_time, TL_model_path=self.TL_model_path
            )

            callback_obj = SpeedOptimizationCallback(
                export_path=src_dir / 'ship' / 'case_studies' / self.case_study,
                every_n_gen=200,
                initial_gen=previous_gen
            )

            

            adaptive_mutation = AdaptiveMutation(initial_rate=0.3, final_rate=0.01, n_gen=total_gen)
            adaptive_crossover = AdaptiveCrossover(initial_rate=0.95, final_rate=0.7, n_gen=total_gen)

            num_new_generations = total_gen - previous_gen

            if previous_pop is None:
                self.get_logger().info("Initializing population randomly.")
                sampling = FloatRandomSampling()  # Default random sampling
            else:
                self.get_logger().info("Initializing population from previous solution.")
                sampling = previous_pop
                assert previous_pop is None or previous_pop.shape[1] == n_var, (
                        f"Mismatch between the number of variables (n_var={n_var}) and the population shape {previous_pop.shape[1]}.")

            algorithm = GA(
                pop_size=200,
                eliminate_duplicates=True,
                mutation=adaptive_mutation,
                crossover=adaptive_crossover,
                sampling=sampling
            )

            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', num_new_generations),
                callback=callback_obj
            )

            self.get_logger().info(f"Optimized speeds in knots: {res.X}")
            self.get_logger().info(f"Total dist covered after optimization (NM): {np.sum(np.array(res.X) * self.timestep)}")

            min_f_values = callback_obj.min_f_values

            plt.plot(min_f_values)
            plt.xlabel('Generation')
            plt.ylabel('Min F')
            plt.title('Convergence Plot')
            filename = "speed_opt_convergence.png"
            plt.savefig(f"imgs/{filename}")
            plt.close()

            self.optimized_speed = res.X
            self.previous_solution_num_gen = total_gen
            self.plot_speeds(problem, res.X)

    def plot_speeds(self, problem, optimized_speed: list):

        cost = problem.speed_objective.cost_function_noise(np.array(optimized_speed).reshape(1, len(optimized_speed)))

        time_vec = np.ones((1,int(self.remaining_time/self.timestep)))*self.timestep
        ship_vec = problem.speed_objective.compute_ship_traj(time_vec=time_vec,speed_profile= np.array(optimized_speed).reshape(1, len(optimized_speed))) 
        mammal_poss = problem.speed_objective.compute_mammal_traj(time_vec=time_vec, mammal_states=self.mammal_current_states)

        time_axis = [(self.timestep * i + self.timestep / 2) for i in range(len(optimized_speed))]

        # Compute L_s_JE_decidecade
        L_s_JE_decidecade, f_decidecade = jomopans_model_matrix(self.freq, self.ship_type, np.array(optimized_speed).reshape(1, len(optimized_speed)), self.ship_length)

        # Reshape L_s_JE_decidecade for plotting
        reshaped_L_s_JE = L_s_JE_decidecade.reshape((-1, f_decidecade.shape[0]))
        
        # Create the figure and axes
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 5))
        
        # Plot the bar chart in the first column
        axs[0].bar(time_axis, optimized_speed, width=self.timestep, color='blue', edgecolor='black')
        axs[0].set_title('Optimized Speed Over Time')
        axs[0].set_xlabel('Time (hrs)')
        axs[0].set_ylabel('Speed (knots)')
        axs[0].grid(True)
        axs[0].set_ylim(self.min_v, self.max_v)
        axs[0].set_xlim(0, self.remaining_time)
        
        # Plot ship_vec and mammal_poss in the second column
        ship_latitudes = ship_vec[0, :, 0]
        ship_longitudes = ship_vec[0, :, 1]
        
        # Define alpha values for ship and mammal positions
        ship_alpha_values = np.linspace(0.2, 1, len(ship_latitudes))
        
        # Plot ship positions with varying alpha values
        for lat, lon, alpha in zip(ship_latitudes, ship_longitudes, ship_alpha_values):
            axs[1].plot(lon, lat, 'bo', alpha=alpha)
        
        # Plot mammal positions with varying alpha values
        for mammal in mammal_poss:
            mammal_latitudes = mammal[:, 0]
            mammal_longitudes = mammal[:, 1]
            mammal_alpha_values = np.linspace(0.2, 1, len(mammal_latitudes))
            
            for lat, lon, alpha in zip(mammal_latitudes, mammal_longitudes, mammal_alpha_values):
                axs[1].plot(lon, lat, 'ro', alpha=alpha)

        for i in range(len(self.waypoints)):
             axs[1].plot(self.waypoints[i].longitude,self.waypoints[i].latitude,'ko',fillstyle='none', markersize=5)
        
        for geom in self.shoreline_polygons['geometry']:
            if geom.geom_type == 'Polygon':
                xs, ys = geom.exterior.xy
                axs[1].fill(xs, ys, alpha=0.5, fc='r', ec='none')
            elif geom.geom_type == 'MultiPolygon':
                for polygon in geom.geoms:  # Access individual Polygons in the MultiPolygon
                    xs, ys = polygon.exterior.xy
                    axs[1].fill(xs, ys, alpha=0.5, fc='r', ec='none')

        # Plot settings
        axs[1].set_title('Ship and Mammal Positions')
        axs[1].set_xlabel('Longitude')
        axs[1].set_ylabel('Latitude')
        axs[1].set_aspect('equal')

        for i in range(reshaped_L_s_JE.shape[0]):
            axs[2].plot(f_decidecade, reshaped_L_s_JE[i, :], color='black')
        
        axs[2].set_xscale('log')
        axs[2].set_xlabel('Frequency (Hz)')
        axs[2].set_ylabel('L_{s,JE} (dB)')
        axs[2].set_title(f'Overlay of L_s_JE_decidecade Across Different Velocity Matrices\nCost = {cost}')
        axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Save the figure
        filename = "optimized_speed_and_positions.png"
        plt.savefig(f"imgs/{filename}")
        plt.close()

    def mammal_state_callback(self, msg, index):
        self.mammal_current_states[index] = msg
        if self.verbose_logging:
            self.get_logger().info(
                f'I heard Mammal State: [{msg.position.latitude}, {msg.position.longitude}, {msg.position.depth}], '
                f'Angle: {msg.angle.heading}, Speed: {msg.speed} m/s')

def main(args=None):
    rclpy.init(args=args)
    speed_optimizer = SpeedOptimizer()
    try:
        rclpy.spin(speed_optimizer)
    except KeyboardInterrupt:
        pass
    finally:
        speed_optimizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
