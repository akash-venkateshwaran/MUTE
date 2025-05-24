import rclpy
from rclpy.node import Node
from custom_interfaces.msg import State, HelperPosition, HeadingAngle, Path, ShipInfo, Speed
from custom_interfaces.srv import StateService
from rosgraph_msgs.msg import Clock

from pyproj import Geod
import math
import time
import numpy as np
import yaml
import os 
import subprocess
import re 
import multiprocessing
import sys
import shutil

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


GEODESIC = Geod(ellps="WGS84")


class Ship(Node):
    def __init__(self):
        super().__init__('ship_node')
        self.setup_logger()
        self.case_study = str(src_dir / 'ship' / 'case_studies' / 'Case_3')

        # Clock publisher setup
        self.clock_publisher = self.create_publisher(Clock, '/clock', 10)

        
        # Checking if the bag folder exist
        self.bag_file_path = os.path.join(self.case_study, 'ship_data')
        if os.path.exists(self.bag_file_path):
            user_input = input(f"Bag files already exist at {self.bag_file_path}. Do you want to delete them? (y/n): ").strip().lower()

            if user_input == 'y':
                try:
                    if os.path.exists(self.bag_file_path):
                        shutil.rmtree(self.bag_file_path)
                        self.get_logger().info(f"Deleted bag file at {self.bag_file_path}")

                except Exception as e:
                    self.get_logger().error(f"Failed to delete bag files: {e}")
                    sys.exit(1)
            else:
                self.get_logger().info("Aborting bag file creation.")
                sys.exit(0)

        # Start recording ROS 2 bag file
        self.start_bag_recording_process()

        # Subscribers:
        self.subscription_ship_optimized_waypoints = self.create_subscription(
            Path,
            'ship_optimized_waypoints',
            self.ship_optimized_waypoints_callback,
            10)
        
        
        self.subscription_ship_optimized_speed = self.create_subscription(
            Speed, 
            'optimized_speeds',
            self.ship_optimized_speed_callback,
            10)
        
        self.subscription_ETA = self.create_subscription(
            Clock, 
            'ETA',
            self.ship_ETA_callback,
            10)

        # Publishers:
        self.publisher = self.create_publisher(
            State,
            'ship_state',
            10)
        
        self.publisher_info = self.create_publisher(
            ShipInfo,
            'ship_info',
            10)
        
    

        # Start recording ROS 2 bag file
        self.start_bag_recording_process()
                    
        self.start_state = None
        self.current_state = None
        self.timer_period = 1.0  # seconds
        self.ETA = 0.0 # mins
        self.optimized_waypoints = []
        self.optimized_speed = [] # knots
        self.current_waypoint_index = 0
        self.waypoint_proximity = 100  # meters, "close enough" threshold

        self.ship_type = None
        self.ship_length = None

        while self.ship_type is None or self.ship_length is None:
            self.load_ship_info()

        # Getting start state:
        self.client = self.create_client(StateService, 'ship_start_state_service')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('ship_start_state_service not available, waiting again...')
        self.request_start_state()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.ship_started = False  # Flag to indicate the ship has started moving and simulation time will start
        self.simulation_time = 0.0 # Simulation time is in mins because time_period = 60

        self.update_ready = False  # Boolean flag to track if both waypoints and speed are received

    def setup_logger(self):
        self.get_logger().info("Ship Node is started!")


    def bag_record(self):
        """Wait for topics starting with 'm' followed by a number, collect only state topics, 
        and start recording a ROS 2 bag file once they match with the number of YAML files and other specified topics."""

        static_topics = ['/ship_state', '/clock', '/source_noise_spl']

        while rclpy.ok():
            topics = self.get_topic_names_and_types()
            available_topic_names = [topic_name for topic_name, _ in topics]

            missing_static_topics = [topic for topic in static_topics if topic not in available_topic_names]

            if missing_static_topics:
                self.get_logger().warn(f"Missing static topics: {missing_static_topics}. Retrying...")
                continue

            

            # If the numbers match, proceed with recording
            all_topics = static_topics
            self.get_logger().warn(f"Recording the following topics: {all_topics}")

            try:
                self.get_logger().info(f"Starting ROS 2 bag recording at {self.bag_file_path}")
                subprocess.Popen(['ros2', 'bag', 'record', '-o', self.bag_file_path, *all_topics])
                break
            except Exception as e:
                self.get_logger().error(f"Failed to start ros2 bag recording: {e}")

            rclpy.spin_once(self, timeout_sec=1.0)
            

    def start_bag_recording_process(self):
        """Start the bag recording function in a separate process."""
        process = multiprocessing.Process(target=self.bag_record)
        process.start()
        self.bag_recording_process = process

    def load_ship_info(self):
        """Load ship information from the first YAML file found in the directory."""
        try:
            # Find the YAML file in the specified case study directory
            yaml_file_path = None
            for file in os.listdir(self.case_study):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    yaml_file_path = os.path.join(self.case_study, file)
                    break

            if yaml_file_path is None:
                self.get_logger().error(f"No Ship YAML file found in directory: {self.case_study}")
                return

            self.get_logger().info(f"Loading ship information from {yaml_file_path}")

            # Load the YAML file
            with open(yaml_file_path, 'r') as yamlfile:
                ship_data = yaml.safe_load(yamlfile)

            # Check if 'ship_info' is present in the YAML data
            if 'ship_info' in ship_data:
                ship_info = ship_data['ship_info']
                self.ship_type = ship_info.get('ship_type', 'Unknown')
                self.ship_length = ship_info.get('ship_length', 0.0)

                self.get_logger().info(f"Ship Info loaded: Type={self.ship_type}, Length={self.ship_length} feet")
            else:
                self.get_logger().error("Missing 'ship_info' field in YAML file.")
        except FileNotFoundError:
            self.get_logger().error(f"Ship Info YAML file not found at {yaml_file_path}")
        except (ValueError, KeyError, yaml.YAMLError):
            self.get_logger().error("Error parsing Ship Info YAML data.")

    def generate_state(self, latitude, longitude, depth, heading, speed):
            return State(
                position=HelperPosition(latitude=latitude, longitude=longitude, depth=depth),
                angle=HeadingAngle(heading=heading),
                speed=speed
            )
    
    def request_start_state(self):
        request = StateService.Request()
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_start_state_response)

    def handle_start_state_response(self, future):
        try:
            response = future.result()
            self.start_state = response.state
            self.get_logger().info(
                f'Received Ship Start State: [{self.start_state.position.latitude}, {self.start_state.position.longitude}, {self.start_state.position.depth}], '
                f'Angle: {self.start_state.angle.heading}, Speed: {self.start_state.speed} m/s'
            )

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def ship_ETA_callback(self, msg):
        self.ETA = msg.clock.sec

    def ship_optimized_speed_callback(self, msg):
        # The ship will continue to follow previous path unless something in opt_speed msg changes
        if  np.sum(np.array(self.optimized_speed)) != np.sum(np.array(msg.opt_speeds)) and msg.opt_speeds != []:
            self.update_ready = True
            if len(msg.opt_speeds) > 0:
                self.optimized_speed = msg.opt_speeds
                self.get_logger().info(f"Optimized speed received: {len(list(self.optimized_speed)) } speed waypoints in knots")
            else:
                self.get_logger().warn("No optimized speeds received in the message.")

    def finding_waypoint_index(self):
        distances = []
        
        for waypoint in self.optimized_waypoints:
             _, _, distance_to_target = GEODESIC.inv(
                self.current_state.position.longitude, 
                self.current_state.position.latitude, 
                waypoint.longitude, 
                waypoint.latitude
            )
             distances.append(distance_to_target)
        min_distance_index = distances.index(min(distances))
        return min_distance_index


    def ship_optimized_waypoints_callback(self, msg):
        if self.update_ready:
            self.optimized_waypoints = msg.waypoints
            waypoints_count = len(self.optimized_waypoints)
            self.get_logger().info(f"{waypoints_count} optimized waypoints received.")

            if waypoints_count > 1 and not math.isnan(self.current_state.position.latitude) and not math.isnan(self.current_state.position.longitude):
                if waypoints_count == 2:
                    last_waypoint = self.optimized_waypoints[-1]
                    _, _, distance_to_target = GEODESIC.inv(
                        self.current_state.position.longitude, 
                        self.current_state.position.latitude, 
                        last_waypoint.longitude, 
                        last_waypoint.latitude
                    )

                    if distance_to_target < self.waypoint_proximity:
                        self.current_waypoint_index = 1
                        self.get_logger().info("Reached the destination (within buffer distance).")
                    else:
                        self.current_waypoint_index = 0
                else:
                    self.current_waypoint_index = self.finding_waypoint_index()
            else:
                self.get_logger().info("A single waypoint received.")
                
            self.update_ready = False

    def compute_current_speed(self) -> float:
        """
        Note that this function returns speed in knots because self.optimized_speed is in knots
        """
        num_intervals = len(self.optimized_speed)
        interval_duration = self.ETA / num_intervals
        interval_index = int(self.simulation_time // interval_duration)
        
        if interval_index >= num_intervals:
            interval_index = num_intervals - 1  

        return self.optimized_speed[interval_index]


    def compute_new_heading_speed(self, ship_pos: HelperPosition ) -> tuple:
        if len(self.optimized_waypoints) > 0:
            target_waypoint = self.optimized_waypoints[self.current_waypoint_index]
            new_heading, _, distance_to_target = GEODESIC.inv(ship_pos.longitude, ship_pos.latitude, target_waypoint.longitude, target_waypoint.latitude)
            speed = self.compute_current_speed()
            return (new_heading, speed* (1/1.94384), distance_to_target) # Return speed in m/s
        else:
            self.get_logger().info("No waypoints or current state unavailable to estimate optimized state.")
            return (self.start_state.angle.heading, self.start_state.speed, 0.0)
        

    def update_state(self, time_period):
        if self.ETA > 0.0:
            new_heading, new_speed, distance_to_target = self.compute_new_heading_speed(ship_pos=self.current_state.position)
            if not math.isnan(new_heading) and not math.isnan(new_speed) and new_speed > 0.0:

                travel_distance = new_speed * time_period 
                while travel_distance > distance_to_target:
                    if self.current_waypoint_index + 1 < len(self.optimized_waypoints):
                        residual_distance = travel_distance - distance_to_target
                        self.current_waypoint_index += 1

                        temp = self.optimized_waypoints[self.current_waypoint_index - 1]
                        self.current_state = self.generate_state(temp.latitude, temp.longitude, self.current_state.position.depth, new_heading, new_speed)
                        new_heading, new_speed, distance_to_target = self.compute_new_heading_speed(ship_pos=self.current_state.position)
                        travel_distance = residual_distance

                    else:
                        # Reached the last waypoint, update the state accordingly
                        self.current_waypoint_index = len(self.optimized_waypoints) - 1
                        target_waypoint = self.optimized_waypoints[self.current_waypoint_index]
                        endlon, endlat = target_waypoint.longitude, target_waypoint.latitude
                        remaining_time = distance_to_target / new_speed
                        self.simulation_time += remaining_time / time_period
                        self.get_logger().info(f"Reached the final destination in {self.simulation_time:.2f} minutes.")
                        self.current_state = self.generate_state(endlat, endlon, self.current_state.position.depth, new_heading, new_speed)

                        # Exporting the exact arrival time:
                        output_file_path = os.path.join(self.bag_file_path, "simulation_results.txt")
                        with open(output_file_path, 'w') as file:
                            file.write(f"Arrival Time: {self.simulation_time:.2f} minutes\n")
                            file.write(f"ETA: {self.ETA:.2f} minutes\n")  # Assuming self.ETA is defined
                        return

                    # Travel the distance towards the current target waypoint
                endlon, endlat, _ = GEODESIC.fwd(
                        self.current_state.position.longitude,
                        self.current_state.position.latitude,
                        new_heading,
                        travel_distance
                    )

                # Update the current state with the new position and heading
                self.current_state = self.generate_state(endlat, endlon, self.current_state.position.depth, new_heading, new_speed)
                self.ship_started = True
                self.simulation_time += 1.0
            else:
                self.get_logger().info("Received NaN for new heading or speed not updating state.")

        else:
            self.get_logger().info("ETA not updated")

    def timer_callback(self):
        # PUblishing ship info:
        ship_info_msg = ShipInfo(ship_type = self.ship_type, ship_length = self.ship_length)
        self.publisher_info.publish(ship_info_msg)

        # Publishing simulated time:
        clock_msg = Clock()
        clock_msg.clock.sec = int(self.simulation_time)
        self.clock_publisher.publish(clock_msg)  # Note all Clock msgs are published always in mins

        if self.current_state is None:
            # Start state received, update current state and begin continuous updates
            self.current_state = self.start_state
            self.get_logger().info(f"Getting the start state: {self.start_state}.")
        else:
            # Continuous updates
            self.update_state(time_period=60.0) # Updates the current_state, each step is 1min
            self.publisher.publish(self.current_state)




def main(args=None):
    rclpy.init(args=args)
    ship_obj = Ship()
    rclpy.spin(ship_obj)
    ship_obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
