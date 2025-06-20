import rclpy
from rclpy.node import Node
from custom_interfaces.msg import State, HelperPosition, HeadingAngle, SPL
from pyproj import Geod
from mammals.launch_utils import get_bounds
from environment.py_polyshp import Polyshp
import os
from shapely.geometry import Point, Polygon, MultiPolygon
import math
from rosgraph_msgs.msg import Clock
import rosbag2_py
from threading import Thread
import numpy as np


from noise_propagation_model.models.RBF.utils_rbf import RBFInterpolatorCustom
from rosbag2_py import SequentialWriter
from rclpy.serialization import serialize_message
import signal

from pathlib import Path

def find_src_dir(start_path: Path) -> Path:
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

src_dir = find_src_dir(Path(__file__))
base_directory = src_dir / "mammals"


GEODESIC = Geod(ellps="WGS84")
bounds = get_bounds()
lat_max = bounds.max_lat
lat_min = bounds.min_lat
lon_max = bounds.max_lon
lon_min = bounds.min_lon

class Mammal(Node):

    def __init__(self, node_name, *, parameter_overrides=None, shared_assets=None):
        super().__init__(node_name, parameter_overrides=parameter_overrides)
        if shared_assets is None:
            raise RuntimeError("Mammal expects shared_assets")
        self.TL_model          = shared_assets.TL_MODEL
        self.shoreline_polygons = shared_assets.SHORELINE_POLYGONS

        signal.signal(signal.SIGINT, self.handle_shutdown) # For clt + c shutdowns
        self.setup_logger()
        # Parameters
        self.declare_parameter('latitude', 49.2072409066087)
        self.declare_parameter('longitude', -123.17497521916255)
        self.declare_parameter('depth', 1000.0)
        self.declare_parameter('heading', 20.0)
        self.declare_parameter('speed', 2.0)
        self.declare_parameter('topic', 'mammal_state')
        self.declare_parameter('bag_dir', '/workspaces/MUTE_v1/src/ship/case_studies/Case_1/mammal_data')

        self.verbose_logging = False
        self.bag_exported = False

        latitude = self.get_parameter('latitude').get_parameter_value().double_value
        longitude = self.get_parameter('longitude').get_parameter_value().double_value
        depth = self.get_parameter('depth').get_parameter_value().double_value
        heading = self.get_parameter('heading').get_parameter_value().double_value
        speed = self.get_parameter('speed').get_parameter_value().double_value # m/s
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.bag_dir = self.get_parameter('bag_dir').get_parameter_value().string_value


        self.bag_dir = os.path.join(self.bag_dir, topic[1:4] + '_TL')

        
        # Publishers:
        self.publisher_ = self.create_publisher(State, topic, 10)

        self.simulation_time = 0.0 # Subscribe to simulation time from ship node
        self.TL_freq = [12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
        self.TL_msgs = None


        self.get_logger().info(f"Parameters: latitude={latitude}, longitude={longitude}, depth={depth}, heading={heading}, speed={speed}")


        # Subscribers:
        self.subscription_clock = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

        self.ship_current_state_subscription = self.create_subscription(
            State,
            'ship_state',
            self.ship_state_callback,
            10
        )

        self.subscription_ETA = self.create_subscription(
            Clock,
            'ETA',
            self.ETA_callback,
            10
        )

        self.initial_state = State(
            position=HelperPosition(latitude=latitude, longitude=longitude, depth=depth),
            angle=HeadingAngle(heading=heading),
            speed=speed
        )
        self.current_state = self.initial_state
        self.ship_state = None
        self.timer = self.create_timer(1.0, self.timer_callback)

    def setup_logger(self):
        self.get_logger().info("Mammal Node is started!")
        
    def ETA_callback(self, msg):
        self.ETA = msg.clock.sec
        if self.TL_msgs is None:
            self.TL_msgs = np.array([SPL() for _ in range(self.ETA)])

    def ship_state_callback(self, msg):
        self.ship_state = msg

    def update_state(self, time_period):
        distance = self.current_state.speed * time_period
        endlon, endlat, _ = GEODESIC.fwd(
            self.current_state.position.longitude,
            self.current_state.position.latitude,
            self.current_state.angle.heading,
            distance
        )

        new_state = State(
            position=HelperPosition(latitude=endlat, longitude=endlon, depth=self.current_state.position.depth),
            angle=HeadingAngle(heading=self.current_state.angle.heading),
            speed=self.current_state.speed
        )

        # Check if the new state is within any shoreline polygon
        if self.check_state(new_state):
            new_state.angle.heading = (new_state.angle.heading + 180) % 360

        return new_state
    
    def handle_shutdown(self, signum, frame):
        self.get_logger().info("Received shutdown signal (SIGINT), calling destroy_node.")
        self.destroy_node()  # This will call your export_as_bagfile() before shutting down
        rclpy.shutdown()  # Ensure the ROS 2 context is properly shut down

    def destroy_node(self):
        # Export data to bag file before shutting down
        self.get_logger().info("Node is shutting down, exporting data to bag file.")
        self.export_as_bagfile()
        super().destroy_node()

    def check_state(self, state):
        point = Point(state.position.longitude, state.position.latitude)
        for geom in self.shoreline_polygons['geometry']:
                        if isinstance(geom, MultiPolygon):
                                for polygon in geom.geoms:
                                        if polygon.contains(point):
                                                if self.verbose_logging:
                                                        self.get_logger().info(f"State {point} is inside polygon, reversing direction")
                                                return True

                        elif isinstance(geom, Polygon):
                                if geom.contains(point):
                                        if self.verbose_logging:
                                                self.get_logger().info(f"State {point} is inside polygon, reversing direction")
                                        return True
        return False

    def timer_callback(self):
        self.publisher_.publish(self.current_state)

        if self.verbose_logging:
            state_msg = (
                f"Publishing Mammal State: [{self.current_state.position.latitude}, "
                f"{self.current_state.position.longitude}, {self.current_state.position.depth}], "
                f"Angle: {self.current_state.angle.heading}, Speed: {self.current_state.speed} m/s"
            )
            self.get_logger().info(state_msg)


    def clock_callback(self, msg):
        temp = msg.clock.sec
        if temp > self.simulation_time:
            self.current_state = self.update_state(time_period=60 * (temp - self.simulation_time))  # 60 because simulation_time is in min
            self.simulation_time = temp

            if self.simulation_time <= self.ETA:
                local_simulation_time = self.simulation_time
                local_ship_state = self.ship_state
                local_mammal_state = self.current_state

                thread = Thread(target=self.compute_and_update_TL_msgs, args=(local_simulation_time, local_ship_state, local_mammal_state))
                thread.start()

                if self.simulation_time == self.ETA:
                    thread.join()
                    self.export_as_bagfile()

            if self.verbose_logging:
                self.get_logger().info(f"Updated simulation time: {self.simulation_time} min")
        else:
            self.get_logger().info(f"Not moving as simulation time: msg:{temp} min < mammal_sim:{self.simulation_time} min")

    def compute_simple_transmission_loss(self, ship_state, mammal_state):
    # Use the local copies of ship_state and mammal_state for computation
        src_lat = ship_state.position.latitude
        src_lon = ship_state.position.longitude
        src_z = ship_state.position.depth

        rec_lat = mammal_state.position.latitude
        rec_lon = mammal_state.position.longitude
        rec_z = mammal_state.position.depth

        _, _, horizontal_distance = GEODESIC.inv(src_lon, src_lat, rec_lon, rec_lat)
        vertical_distance = abs(rec_z - src_z)
        distance_3d = (horizontal_distance**2 + vertical_distance**2) ** 0.5

        if distance_3d == 0:
            return [0.0] * len(self.TL_freq)  # Return zero transmission loss if the distance is zero

        transmission_loss = [20 * math.log10(distance_3d)] * len(self.TL_freq)
        return transmission_loss
    
    def compute_transmission_loss(self, ship_state, mammal_state):
        ship_loc = [ship_state.position.latitude, ship_state.position.longitude]
        mammal_loc = [mammal_state.position.latitude, mammal_state.position.longitude]
        model_input = np.hstack((ship_loc, mammal_loc)).reshape(1, -1)  # Shape: (1, 4)

        model_output = self.TL_model(model_input)  # Output: (1, num_freq)
        last_column = model_output[:, -1].reshape(-1, 1)
        model_output = np.hstack((model_output, last_column))


        # Assert that the number of frequencies matches the expected length
        assert model_output.shape[1] == len(self.TL_freq), "Number of frequencies in model output does not match self.TL_freq"

        model_output = model_output.flatten().tolist()

        return model_output

    def compute_and_update_TL_msgs(self, local_simulation_time, local_ship_state, local_mammal_state):

        if local_simulation_time < self.ETA:
            tl_msg = SPL()
            tl_msg.freq = [float(f) for f in self.TL_freq]
            tl_msg.spl = self.compute_transmission_loss(local_ship_state, local_mammal_state)
            self.TL_msgs[local_simulation_time] = tl_msg

    def export_as_bagfile(self):
        if self.bag_exported is False: # IF the bag file is not exported yet
            bag = SequentialWriter()
            storage_options = rosbag2_py._storage.StorageOptions(uri=self.bag_dir, storage_id='sqlite3')
            converter_options = rosbag2_py._storage.ConverterOptions('', '')
            bag.open(storage_options, converter_options)

            topic_info = rosbag2_py._storage.TopicMetadata(
                name='/mammal_TL_msg',  # Use a single, static topic name
                type='custom_interfaces/msg/SPL',  # The message type
                serialization_format='cdr'  # Standard ROS 2 serialization format
            )
            bag.create_topic(topic_info)

            for t, msg in enumerate(self.TL_msgs):
                current_time = rclpy.clock.Clock().now().nanoseconds
                
                if isinstance(msg, SPL):
                    serialized_msg = serialize_message(msg)
                    
                    bag.write('/mammal_TL_msg', serialized_msg, current_time)
                else:
                    self.get_logger().warn(f"Invalid message at index {t}: {msg}. Stopping export.")
                    return

            self.get_logger().info(f"Transmission Loss data exported as bag file at {self.bag_dir}")
            text_file_path = os.path.join(self.bag_dir, "simulation_info.txt")
            with open(text_file_path, 'w') as file:
                file.write(f"ETA: {self.ETA}\n")
                file.write(f"Simulation Time: {self.simulation_time:.2f} minutes\n")

            self.bag_exported = True
    

def main(args=None):
    rclpy.init(args=args)
    node = Mammal()
    try:
        rclpy.spin(node)   # <-- this will return when shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
