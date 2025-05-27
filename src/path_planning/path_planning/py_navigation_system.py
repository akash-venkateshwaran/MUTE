import rclpy
from rclpy.node import Node
from custom_interfaces.msg import State, HelperPosition, HeadingAngle, Path
from custom_interfaces.srv import BathyService, PolygonService, BoundsService, StateService
import os
from pyproj import Geod
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from threading import Thread
import numpy as np
import re
import yaml
from shapely.geometry import Point
import sys
import pandas as pd
from rosgraph_msgs.msg import Clock

from path_planning.ompl_path2 import OMPLPath, OMPLPathState

from pathlib import Path as Path2

def find_src_dir(start_path: Path2) -> Path2:
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate a 'src' directory.")

GEODESIC = Geod(ellps="WGS84")
base_directory = find_src_dir(Path2(__file__))


import random
from ompl.util import RNG

import matplotlib
matplotlib.use('Agg')

import gc

class NavigateSystem(Node):
    def __init__(self):
        super().__init__('navigate_system')
        self.setup_logger()

        self.case_study = base_directory / 'path_planning' / 'case_studies' / 'Case_1' / 'case_study.csv'
        self.export_path = base_directory / 'ship' / 'case_studies' / 'Case_1' # NOTE Make sure both the path's case number matches
        self.TL_model_path = base_directory / 'noise_propagation_model' / 'noise_propagation_model' / 'models' / 'RBF' / 'model'

        self.verbose_logging = False
        self.ompl_path_state = None
        self.ompl_path = None
        RNG.setSeed(42) # NOTE Setting the seeds
        random.seed(42)

        self.ship_current_state = State()
        self.ship_start_state = State()
        self.ship_end_state = State()

        self.mammal_current_states = np.empty((0,), dtype=State)
        self.ship_optimized_waypoints = Path()
        self.ship_current_waypoint_index = 0
        self.ship_waypoint_proximity = 100.0 # In meters
        self.num_ompl_sol = 0
        self.simulation_time = 0.0 # Subscribe to simulation time from ship node
        self.ompl_runtime = 500.0 # TODO Change to 500
        self.solving_period = 1.0 # To run the solver, it will run right after the previous run is complete. Keep it 1s
        self.solve_ompl_thread = None
        self.skip_process = None
        self.optimize_once = False

        self.ETA = None # mins

        self.shoreline_polygons = []  # List of GeoSeries geometric objects in lat/lon
        self.shiplane_polygons = []  # List of GeoSeries geometric objects in lat/lon

        self.bb_buffer = 1
        self.bb_max_lat = self.bb_min_lat = self.bb_max_long = self.bb_min_long = 0.0


        self.load_voyage_definition(self.case_study)
        self.geting_ompl_boundingbox(self.bb_buffer) # Updates bb_max_lat, bb_max_long, bb_min_lat and bb_min_long

        self.ship_start_state_service = self.create_service(StateService, 'ship_start_state_service', self.handle_ship_start_state_service)
        self.ship_end_state_service = self.create_service(StateService, 'ship_end_state_service', self.handle_ship_end_state_service)

        ######### Publishers:
        self.ship_optimized_waypoints_publisher = self.create_publisher(
            Path,
            'ship_optimized_waypoints',
            10
        )
        self.ETA_publisher = self.create_publisher(Clock, 'ETA', 10)

        ######### Services:
        self.bounds_service = self.create_service(BoundsService, 'bounds_service', self.handle_bounds_service)

        self.shoreline_client = self.create_client(PolygonService, 'shoreline_service')
        while not self.shoreline_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('ShorelineService not available, waiting again...')
        self.get_polygon_service_response(self.shoreline_client, 'shoreline', 'shoreline_polygons') # Note that 'shoreline_polygons' should match self.shoreline_polygons 

        self.shiplane_client = self.create_client(PolygonService, 'shiplane_service')
        while not self.shiplane_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('ShiplaneService not available, waiting again...')
        self.get_polygon_service_response(self.shiplane_client, 'shiplane', 'shiplane_polygons')

        # Checks if start state or end state is outisde shiplane poly
        self.check_ship_states_in_lane()



        ######### Subscriptions:
        if self.skip_process == False:
            self.subscribe_to_mammal_topics()

            self.ship_state_subscription = self.create_subscription(
                State,
                'ship_state',
                self.ship_state_callback,
                10
            )

            self.subscription_clock = self.create_subscription(
                Clock,
                '/clock',
                self.plotting,
                10
            )

        ######### Timer:
        self.solving_timer = self.create_timer(self.solving_period, self.all_publishers)

    
    def setup_logger(self):
        self.get_logger().info("Navigation Node is started!")

    def generate_state(self, latitude, longitude, depth, heading, speed):
        return State(
            position=HelperPosition(latitude=latitude, longitude=longitude, depth=depth),
            angle=HeadingAngle(heading=heading),
            speed=speed
        )

    def ship_state_callback(self, msg):
        # Update ship_current_state attribute when a new message is received
        self.ship_current_state = msg
        if self.verbose_logging:
            self.get_logger().info(
                f'I heard Ship State: [{msg.position.latitude}, {msg.position.longitude}, {msg.position.depth}], '
                f'Angle: {msg.angle.heading}, Speed: {msg.speed} m/s')
            
    def mammal_state_callback(self, msg, index):
        self.mammal_current_states[index] = msg
        if self.verbose_logging:
            self.get_logger().info(
                f'I heard Mammal State: [{msg.position.latitude}, {msg.position.longitude}, {msg.position.depth}], '
                f'Angle: {msg.angle.heading}, Speed: {msg.speed} m/s')

    def handle_ship_start_state_service(self, request, response):
        response.state = self.ship_start_state
        return response

    def handle_ship_end_state_service(self, request, response):
        response.state = self.ship_end_state
        return response

    def handle_bounds_service(self, request, response):
        response.max_lat = self.bb_max_lat
        response.min_lat = self.bb_min_lat
        response.max_lon = self.bb_max_long
        response.min_lon = self.bb_min_long
        return response
    
    def subscribe_to_mammal_topics(self):
        pattern = re.compile(r'^/m\d')
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

    def compute_lag_time_case_study(self, df):
        """
        This function is to calc lag time in case studies btw dist based on vel and path:
        - Extracting velocity, longitude, and latitude columns.
        - Calculating the distance between consecutive latitude/longitude points.
        -  Summing all distances to get the total distance in meters.
        - Summing all velocities to get the total velocity in m/s.
        - Computing the total time based on total distance and average velocity.
        - Returning the total time in minutes as an integer.
        """
        longitudes = df.iloc[:, 2]
        latitudes = df.iloc[:, 3]
        velocities_knots = df.iloc[:, 1]
        
        velocities_mps = velocities_knots * 0.514444
        
        distances_m = []
        for i in range(1, len(longitudes)):
            lon1, lat1 = longitudes[i - 1], latitudes[i - 1]
            lon2, lat2 = longitudes[i], latitudes[i]
            _, _, distance_m = GEODESIC.inv(lon1, lat1, lon2, lat2)
            distances_m.append(distance_m)
        
        distances_m.insert(0, 0)
        df['distance_between_points_m'] = distances_m

        
        total_distance_m = sum(distances_m)
        total_velocity_mps = sum(velocities_mps)

       
        if total_velocity_mps > 0:
            total_time_seconds = total_distance_m / (total_velocity_mps / len(velocities_mps))
        else:
            total_time_seconds = 0
        
        total_time_minutes = total_time_seconds / 60.0
        
        return int(total_time_minutes)
        

    def load_voyage_definition(self, file_path):
        try:
            voyage_data = pd.read_csv(file_path, header=None) 
            if len(voyage_data.columns) < 4:
                self.get_logger().error("CSV file has insufficient columns. Expected at least 4 columns: time, longitude, latitude, etc.")
                return

            start_state_values = voyage_data.iloc[0]
            end_state_values = voyage_data.iloc[-1]

            self.ship_start_state = self.generate_state(
                start_state_values[3],  
                start_state_values[2],  
                0.0,  
                0.0,  
                0.0   
            )

            self.ship_end_state = self.generate_state(
                end_state_values[3], 
                end_state_values[2], 
                0.0,  
                0.0,  
                0.0   
            )

            self.ETA = self.compute_lag_time_case_study(df = voyage_data)

            self.get_logger().info("Voyage definition loaded from CSV.")

            print("\nChoose an option: ")
            print("(1) Run the case study")
            print("(2) Optimize the case study")
            print("(3) Optimize the case study only ONCE")

            user_choice = input("Enter your choice (1/2/3): ").strip()
            if user_choice == '1':
                self.skip_process = True
                waypoints = []
                for _, row in voyage_data.iterrows():
                    waypoint = HelperPosition(latitude=row[3], longitude=row[2], depth=0.0)
                    waypoints.append(waypoint)
                self.ship_optimized_waypoints.waypoints = waypoints
                self.get_logger().info(f"Optimized waypoints updated with {len(waypoints)} waypoints.")
            elif user_choice == '2':
                self.skip_process = False
                self.get_logger().info("User chose to optimize the case study.")
            elif user_choice == '3':
                self.skip_process = False
                self.optimize_once = True
                self.get_logger().info("User chose to optimize the case study.")
            else:
                self.get_logger().warn("Invalid choice entered. Defaulting to running the case study.")
                self.skip_process = True

        except FileNotFoundError:
            self.get_logger().error(f"Voyage Definition CSV file not found at {file_path}")
        except pd.errors.EmptyDataError:
            self.get_logger().error("CSV file is empty or invalid.")
        except Exception as e:
            self.get_logger().error(f"Error parsing Voyage Definition CSV data: {e}")


            

    def geting_ompl_boundingbox(self, buffer=1.0):
        # Extract latitude and longitude values from start and end states
        start_lat = self.ship_start_state.position.latitude
        start_long = self.ship_start_state.position.longitude
        end_lat = self.ship_end_state.position.latitude
        end_long = self.ship_end_state.position.longitude
        
        self.bb_max_lat = max(start_lat, end_lat) + buffer  
        self.bb_min_lat = min(start_lat, end_lat) - buffer 
        self.bb_max_long = max(start_long, end_long) + buffer 
        self.bb_min_long = min(start_long, end_long) - buffer

        self.get_logger().info(f"Bounding Box Updated: \n Max Latitude: {self.bb_max_lat}, Min Latitude: {self.bb_min_lat}, Max Longitude: {self.bb_max_long}, Min Longitude: {self.bb_min_long}")


    def check_ship_states_in_lane(self):
        """Checks if ship_start_state or ship_end_state lie inside one of the shiplane polygons.
        
        Terminates execution and prints an error message if either state is outside the polygons.
        """
        start_point = Point(self.ship_start_state.position.longitude, self.ship_start_state.position.latitude)
        end_point = Point(self.ship_end_state.position.longitude, self.ship_end_state.position.latitude)

        start_in_polygon = any(polygon.contains(start_point) for polygon in self.shiplane_polygons)
        end_in_polygon = any(polygon.contains(end_point) for polygon in self.shiplane_polygons)

        # If either point is not in a polygon, print error and terminate
        if not start_in_polygon:
            self.get_logger().error("Start state is outside the shipping lane polygons.")
            sys.exit("Terminating: Start state is outside the shipping lane polygons.")
        
        if not end_in_polygon:
            self.get_logger().error("End state is outside the shipping lane polygons.")
            sys.exit("Terminating: End state is outside the shipping lane polygons.")



    def plotting(self, msg):
        temp = msg.clock.sec # Note This is NOT in sec but in mins
        if temp > self.simulation_time:
            self.simulation_time = temp
            if self.verbose_logging:
                self.get_logger().info(f"Plotted and saved at {temp} seconds since first ompl solution.")


    def all_publishers(self):
        msg = Clock()
        msg.clock.sec = self.ETA
        self.ETA_publisher.publish(msg) # Note all Clock msgs are published always in mins
        self.ship_optimized_waypoints_publisher.publish(self.ship_optimized_waypoints)


        if self.skip_process:
            self.get_logger().info(f"Skipping OMPL and running the case study. Publishing {len(self.ship_optimized_waypoints.waypoints)}")
        else:
            if self.solve_ompl_thread is None or not self.solve_ompl_thread.is_alive():
                try:
                    self.solve_ompl_thread = Thread(target=self.solve_ompl)
                    self.solve_ompl_thread.start()
                except Exception as e:
                    self.get_logger().error(f"Error starting solve_ompl_thread: {e}")
            


    def solve_ompl(self):
        is_ship_state_changed = (self.ship_current_state.position.latitude != 0.0 or
                                self.ship_current_state.position.longitude != 0.0 or
                                self.ship_current_state.position.depth != 0.0 or
                                self.ship_current_state.angle.heading != 0.0 or 
                                self.ship_current_state.speed != 0.0)

        is_mammal_states_changed = self.mammal_current_states[0] != None


        
        if is_ship_state_changed and is_mammal_states_changed and (self.bb_max_lat != 0.0 or self.bb_min_lat != 0.0 or 
    self.bb_max_long != 0.0 or self.bb_min_long != 0.0) and len(self.shoreline_polygons) != 0:
            # TODO Add shiplanes as attribute to OMPLState
            self.ompl_path_state = OMPLPathState(
                ship_current_state=self.ship_current_state,
                mammal_current_states=self.mammal_current_states,
                ship_start_state=self.ship_start_state,
                ship_end_state=self.ship_end_state, bounding_box = [self.bb_min_lat, self.bb_min_long, self.bb_max_lat, self.bb_max_long],count = self.num_ompl_sol,
                shipplanes_polygons=self.shiplane_polygons, GEODESIC=GEODESIC, TL_model_path = self.TL_model_path,
                previous_waypoints=self.ship_optimized_waypoints.waypoints, 
            )
            self.ompl_path = OMPLPath(max_runtime=self.ompl_runtime, ompl_state=self.ompl_path_state)


            if self.ompl_path.solved:
                self.num_ompl_sol+=1
                if self.num_ompl_sol == 1:
                    self.ompl_runtime = self.ompl_runtime/2
                distance = self.ompl_path.distance_from_goal()
                total_cost = self.ompl_path._simple_setup.getSolutionPath().cost(self.ompl_path._simple_setup.getOptimizationObjective()).value()

                self.get_logger().info(f"Solved: Final Total cost of the path: {total_cost} Distance btw the last waypoint and goal state: {distance} km. Published the new optimized waypoints")
                self.ship_optimized_waypoints._waypoints =  self.ompl_path.get_waypoints()

                self.ompl_path.plot_ompl_solution(1.0 ,self.ship_current_state,self.mammal_current_states, waypoints = self.ompl_path.get_waypoints()) # TODO Remove this func. Only for testing
                self.ompl_path.plot_convergence(filename = "OMPL_convergence.png" )
                self.ompl_path.export_waypoints_to_csv(self.ship_optimized_waypoints._waypoints, self.export_path)

                if self.optimize_once: # Making sure OMPL is solved only ONCE
                    self.skip_process = True

                # To clear RAM memory
                self.ompl_path._simple_setup.clear()
                del self.ompl_path
                del self.ompl_path_state
                gc.collect()

            else:
                self.get_logger().error("Not solved within the given runtime")
        else:
            self.get_logger().info("Ship/mammal states have not changed from initialization/bathy or shoreline not received; skipping OMPLPath execution.")


    def get_polygon_service_response(self, client, service_name, polygons_attribute):
        """Method to obtain polygon data from a given service and convert to polygons."""
        request = PolygonService.Request()
        request.min_box_edge.latitude = self.bb_min_lat
        request.min_box_edge.longitude = self.bb_min_long
        request.max_box_edge.latitude = self.bb_max_lat
        request.max_box_edge.longitude = self.bb_max_long

        future = client.call_async(request)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    result = future.result()
                    polygons = []
                    for start_index in result.polygon_starts: # Polygon_starts are the starting indices of each polygon in the lat/lon list
                        end_index = result.polygon_starts.index(start_index) + 1
                        end_index = result.polygon_starts[end_index] if end_index < len(result.polygon_starts) else len(result.latitudes)

                        polygon_points = [(result.longitudes[i], result.latitudes[i]) for i in range(start_index, end_index)]

                        if len(polygon_points) > 2:
                            polygons.append(Polygon(polygon_points))

                    # Convert the list of polygons into a GeoSeries
                    setattr(self, polygons_attribute, gpd.GeoSeries(polygons))

                    self.get_logger().info(f"Received {service_name.capitalize()} Service Response. Converted to {len(getattr(self, polygons_attribute))} polygons.")
                else:
                    self.get_logger().error(f'{service_name.capitalize()}Service call failed %r' % (future.exception(),))
                break

def main(args=None):
    rclpy.init(args=args)

    navig_obj = NavigateSystem()

    rclpy.spin(navig_obj)

    navig_obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
