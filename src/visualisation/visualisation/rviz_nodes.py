#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from ament_index_python.packages import get_package_share_directory
import re
import numpy as np
from custom_interfaces.msg import State
from custom_interfaces.msg import Path as Path_custom
from custom_interfaces.srv import StateService
from pyproj import Proj

from visualisation.utils import RVizMarker, create_bathy_stl
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped

from builtin_interfaces.msg import Time
from visualization_msgs.msg import Marker
from rosgraph_msgs.msg import Clock
import math
from nav_msgs.msg import OccupancyGrid
from shapely.geometry import Point

from custom_interfaces.srv import BoundsService
import threading
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from std_msgs.msg import Header

from pathlib import Path as Path2

from noise_propagation_model.models.RBF.utils2 import RBFInterpolatorCustom
from environment.py_polyshp import Polyshp


from sensor_msgs.msg import PointCloud2, PointField
from pyproj import Geod

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

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')

        self.setup_logger()
        self.verbose_logging = False
        self.rviz_scale = 'km' # Everything is in meters by default
        self.ref_lat = None
        self.ref_lon = None
        self.proj = None
        self.ETA = 0.0 # mins
        self.mammal_topics = []
        self.rviz_mammals = []
        self.rviz_mammals_icon = []
        self.rviz_mammals_odom = []
        self.waypoints = []

        # TL computation:
        self.TL_contour = None
        self.tl_thread = None
        self.tl_lock = threading.Lock()
        self.rbf_interpolator = RBFInterpolatorCustom.import_model(str(
                src_dir / 'noise_propagation_model' / 'noise_propagation_model' / 'models' / 'RBF' / 'model_single_freq'))

        self.rviz_ship = self.create_publisher(Marker, 'rviz_ship', 10)
        self.rviz_ship_icon = self.create_publisher(Marker, 'rviz_ship_i', 10)
        self.rviz_odom = self.create_publisher(Odometry, 'rviz_ship_odom', 10)
        self.rviz_bathy = self.create_publisher(Marker, 'rviz_bathy', 10)
        self.rviz_path = self.create_publisher(Path, 'rviz_path', 10)
        self.rviz_clock_marker = self.create_publisher(Marker, 'rviz_clock_text', 10)
        self.rviz_occupancy_grid_publisher = self.create_publisher(OccupancyGrid, 'rviz_occupancy_grid', 10)
        self.rviz_pointcloud_publisher = self.create_publisher(PointCloud2, 'rviz_contour', 10)


        # Get the reference point for proj
        self.get_reference_point()

        bathy_file_name = f'bathy_lat{self.ref_lat}_lon{self.ref_lon}.stl'
        bathy_file_path = src_dir / 'visualisation' / 'mesh' / bathy_file_name

        if not os.path.exists(bathy_file_path):
            create_bathy_stl(proj=self.proj, output_path=bathy_file_path)
        else:
            self.get_logger().info(f"Bathymetry file already exists at {bathy_file_path}")

        self.ship_state_subscription = self.create_subscription(
            State,
            'ship_state',
            self.ship_state_callback,
            10
        )

        mesh_path = os.path.join(get_package_share_directory('visualisation'), 'mesh', 'ship.obj')
        self.ship_marker = RVizMarker(mesh_path=mesh_path)

        whale_mesh_path = os.path.join(get_package_share_directory('visualisation'), 'mesh', 'Whale.obj')
        self.mammal_marker = RVizMarker(mesh_path=whale_mesh_path)

        icon_mesh_path = os.path.join(get_package_share_directory('visualisation'), 'mesh', 'Icon.obj')
        self.icon_marker = RVizMarker(mesh_path=icon_mesh_path)

        

        self.get_logger().info("Mammal topics are subscribed")

        self.ship_waypoints_subscription = self.create_subscription(
            Path_custom,
            'ship_optimized_waypoints',
            self.waypoints_callback,
            10
        )

        # Bathy and shoreline:
        self.bathy_marker = RVizMarker(mesh_path=bathy_file_path)
        bounds = self.get_bounds()
        if bounds:
            lat_min, lat_max = bounds.min_lat, bounds.max_lat
            lon_min, lon_max = bounds.min_lon, bounds.max_lon
            self.get_logger().info(f"Loaded bounds: lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}")

            # Load shapefile data using these bounds
            polygon_list = self.load_shapefile_data(lat_min, lat_max, lon_min, lon_max)
            # Compute occupancy grid once and store it
            self.occupancy_grid_msg = self.create_occupancy_grid(polygon_list=polygon_list)

            self.get_logger().info("Occupancy grid computation done")

        # Timer:   
        self.fixed_timer = self.create_timer(5.0, self.publish_fixed_data)
        # Create a timer to periodically check for new mammal topics
        self.topic_check_timer = self.create_timer(2.0, self.check_mammal_topics)

        # Subscribers:
        self.subscription_clock = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )
        self.subscription_ETA = self.create_subscription(
            Clock,
            '/ETA',
            self.get_ETA,
            10
        )

    def setup_logger(self):
        self.get_logger().info("RViz Node is started!")
    
    def load_shapefile_data(self, lat_min, lat_max, lon_min, lon_max):

        shp_data_path = str(src_dir / 'environment' / 'shoreline_data')
        shp_class = Polyshp(shp_data_path)
        
        polygon_list = shp_class.get_polygons(lat_min, lat_max, lon_min, lon_max)
        return polygon_list

    def publish_pointcloud(self):
        if self.TL_contour is None:
            print("No TL contour data available.")
            return

        point_x = self.TL_contour[:, 0]
        point_y = self.TL_contour[:, 1]
        TL = self.TL_contour[:, 2]

        min_TL = TL.min()
        max_TL = TL.max()

        # Normalize TL values and map to colors
        norm = mcolors.Normalize(vmin=min_TL, vmax=max_TL)
        colormap = plt.get_cmap('jet')
        colors = (colormap(norm(TL))[:, :3] * 255).astype(np.uint8)

        # Combine RGB components into a single uint32 value
        colors_rgb = (
            (colors[:, 0].astype(np.uint32) << 16) |
            (colors[:, 1].astype(np.uint32) << 8) |
            colors[:, 2].astype(np.uint32)
        )

        # Define a structured array data type matching PointCloud2 fields
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
        ])

        # Create the structured array and populate it
        points = np.zeros(point_x.shape[0], dtype=dtype)
        points['x'] = point_x
        points['y'] = point_y
        points['z'] = -0.1  # Assuming a flat surface (z = 0)
        points['rgb'] = colors_rgb


        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.height = 1
        msg.width = points.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False

        # Define the fields in the point cloud
        msg.fields = [
            PointField(name='x', offset=0,   datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,   datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,   datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16  # 4 fields x 4 bytes per field
        msg.row_step = msg.point_step * points.shape[0]
        msg.data = points.tobytes()

        self.rviz_pointcloud_publisher.publish(msg)
        if self.verbose_logging:
            self.get_logger().info(f"Published {len(point_x)} colored point cloud based on TL contour.")

    def get_bounds(self):
        # Create a client for the BoundsService
        client = self.create_client(BoundsService, 'bounds_service')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('BoundsService not available, waiting again...')
        
        request = BoundsService.Request()
        future = client.call_async(request)
        
        self.get_logger().info("Waiting for bounds service response...")
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(
                f"Received bounds: max_lat={response.max_lat}, min_lat={response.min_lat}, "
                f"max_lon={response.max_lon}, min_lon={response.min_lon}"
            )
            return response
        else:
            self.get_logger().error('BoundsService call failed %r' % (future.exception(),))
            return None
        
    def create_occupancy_grid(self, polygon_list, x_origin=-150.0, y_origin=-125.0, resolution=0.5, num_x=500, num_y=300):
        """
        Create an occupancy grid message where each cell indicates if it falls within any polygon.

        Args:
            x_origin (float): X-coordinate origin in kilometers.
            y_origin (float): Y-coordinate origin in kilometers.
            resolution (float): Resolution in kilometers per cell.
            num_x (int): Number of cells along the x-axis.
            num_y (int): Number of cells along the y-axis.
            polygon_list: List of polygons from shapefile data.

        Returns:
            OccupancyGrid: OccupancyGrid message with populated data.
        """
        import numpy as np
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        import shapely

        x_origin = float(x_origin)
        y_origin = float(y_origin)
        resolution = float(resolution)
        num_x = int(num_x)
        num_y = int(num_y)

        grid_msg = OccupancyGrid()
        grid_msg.info.resolution = resolution
        grid_msg.info.width = num_x
        grid_msg.info.height = num_y
        grid_msg.info.origin.position.x = x_origin
        grid_msg.info.origin.position.y = y_origin
        grid_msg.header.frame_id = "map"

        x_coords = x_origin + np.arange(num_x) * resolution
        y_coords = y_origin + np.arange(num_y) * resolution
        xv, yv = np.meshgrid(x_coords, y_coords, indexing='xy')
        wx_km_flat = xv.ravel()
        wy_km_flat = yv.ravel()
        wx_m = wx_km_flat * 1000.0
        wy_m = wy_km_flat * 1000.0
        lon, lat = self.proj(wx_m, wy_m, inverse=True)
        grid_data = np.zeros(grid_msg.info.height * grid_msg.info.width, dtype=int)

        if polygon_list is not None:
            polygons = []
            for poly in polygon_list['geometry']:
                if poly.geom_type == 'Polygon':
                    polygons.append(poly)
                elif poly.geom_type == 'MultiPolygon':
                    polygons.extend(poly.geoms)

            unified_polygon = unary_union(polygons)

            if hasattr(shapely, 'vectorized'):
                mask = shapely.vectorized.contains(unified_polygon, lon, lat)
            else:
                from shapely.prepared import prep
                prepared_polygon = prep(unified_polygon)
                from shapely.geometry import Point
                points = [Point(lon[i], lat[i]) for i in range(len(lon))]
                mask = np.array([prepared_polygon.contains(point) for point in points])

            grid_data[mask] = 100

        grid_msg.data = grid_data.tolist()
        return grid_msg


    def get_reference_point(self):
        client = self.create_client(StateService, 'ship_start_state_service')
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
            client = self.create_client(StateService, 'ship_start_state_service')
        request = StateService.Request()
        future = client.call_async(request)
        self.get_logger().info("Spinning to wait for the service response...")
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("Finished spinning.")
        if future.result() is not None:
            self.get_logger().info("Received start state from service.")
            start_state = future.result().state
            self.ref_lat = start_state.position.latitude
            self.ref_lon = start_state.position.longitude
            if self.verbose_logging:
                self.get_logger().info(f"Reference Latitude: {self.ref_lat}, Reference Longitude: {self.ref_lon}")
            
            proj_string = f"+proj=tmerc +lat_0={self.ref_lat} +lon_0={self.ref_lon} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m"
            self.proj = Proj(proj_string)
        else:
            self.get_logger().error('Failed to call service')


    def check_mammal_topics(self):
        """Periodically check for new mammal topics and subscribe to them"""
        pattern = re.compile(r'^/m\d')
        topics = self.get_topic_names_and_types()
        available_mammal_topics = [topic_name for topic_name, _ in topics if pattern.match(topic_name)]
        
        # Find new topics that we haven't subscribed to yet
        new_topics = [topic for topic in available_mammal_topics if topic not in self.mammal_topics]
        
        if new_topics:
            # Print information about the new topics found
            self.get_logger().info(f"Found {len(new_topics)} new mammal topics: {new_topics}")
            
            # Subscribe to each new topic
            for new_topic in new_topics:
                index = len(self.mammal_topics)
                self.create_subscription(
                    State,
                    new_topic,
                    lambda msg, idx=index: self.mammal_state_callback(msg, idx),
                    10
                )
                
                # Create publishers for the new mammal
                rviz_mammal_pub = self.create_publisher(Marker, f'rviz_mammal_{index}', 10)
                self.rviz_mammals.append(rviz_mammal_pub)
                
                rviz_mammal_icon_pub = self.create_publisher(Marker, f'rviz_mammal_{index}_i', 10)
                self.rviz_mammals_icon.append(rviz_mammal_icon_pub)
                
                rviz_mammal_odom_pub = self.create_publisher(Odometry, f'rviz_mammal_{index}_odom', 10)
                self.rviz_mammals_odom.append(rviz_mammal_odom_pub)
                
                self.mammal_topics.append(new_topic)
            
            total_mammals = len(self.mammal_topics)
            self.get_logger().info(f"Now publishing {total_mammals} mammal topics")
        elif not available_mammal_topics and not self.mammal_topics:
            self.get_logger().warn("No mammal topics found. Will continue checking...")

    def create_icon_pose(self,marker_pose):
        rotation = R.from_euler('xyz', [90.0, 0.0, 0.0], degrees=True)
        quaternion = rotation.as_quat()
        icon_pose = marker_pose.copy()
        icon_pose[2] += 1.0
        icon_pose[3:] = [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        return icon_pose
    
    def convert_angle(self, angle):
        return angle % 360 if angle >= 0 else (angle + 360) % 360
    
    def compute_heading_angle(self, current_x, current_y, previous_x, previous_y):
        delta_x = current_x - previous_x
        delta_y = current_y - previous_y
        heading_angle = math.atan2(delta_y, delta_x)
        heading_angle_deg = math.degrees(heading_angle)

        return heading_angle_deg

    def compute_TL_contour(self, src_lat: float, src_long: float):
        """ The output is a 2D matrix where the row are individual cloud points and cols are location and TL"""

        angles = np.linspace(-180, 180.0, num=200, endpoint=False)
        radius_array = np.linspace(0.01, 50.0, num=100) # 0.01 to 10.0 kms of span
        radius_grid, angle_grid = np.meshgrid(radius_array, angles, indexing='ij')
        src_lat_grid = np.full_like(radius_grid, src_lat)
        src_long_grid = np.full_like(radius_grid, src_long)

        radius_flat = radius_grid.ravel()
        angle_flat = angle_grid.ravel()
        src_lat_flat = src_lat_grid.ravel()
        src_long_flat = src_long_grid.ravel()

        dist_flat = radius_flat*1000
        rec_long_flat, rec_lat_flat, back_azimuth = GEODESIC.fwd(src_long_flat, src_lat_flat, angle_flat, dist_flat)
        

        model_input = np.column_stack((rec_lat_flat, rec_long_flat, src_lat_flat, src_long_flat))
        tl = self.rbf_interpolator(model_input).ravel()

        rec_x, rec_y = self.proj(rec_long_flat, rec_lat_flat)
        rec_x = rec_x/ 1000.0
        rec_y = rec_y/ 1000.0

        self.TL_contour = np.column_stack((rec_x, rec_y, -tl))
    
    def get_ETA(self, msg):
        self.ETA = msg.clock.sec

    def ship_state_callback(self, msg):
        if self.verbose_logging:
            self.get_logger().info(
                f'I heard Ship State: [{msg.position.latitude}, {msg.position.longitude}, {msg.position.depth}], '
                f'Angle: {msg.angle.heading}, Speed: {msg.speed} m/s')

        x, y = self.proj(msg.position.longitude, msg.position.latitude)
        depth = msg.position.depth

        # Getting the Heading Angle:
        previous_x = self.previous_x if hasattr(self, 'previous_x') else x
        previous_y = self.previous_y if hasattr(self, 'previous_y') else y
        heading_angle = self.compute_heading_angle(x, y, previous_x, previous_y)


        self.previous_x = x
        self.previous_y = y

        rotation = R.from_euler('xyz', [0.0, 0.0,heading_angle + 180], degrees=True)
        odom_rotation = R.from_euler('xyz', [0.0, 0.0, heading_angle], degrees=True)
        quaternion = rotation.as_quat()


        marker_pose = [x, y, -depth, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        # marker_scale = [0.02, 0.02, 0.02] # TODO Keep this scale if you want original scale of ship
        marker_scale = [0.1, 0.1, 0.1]
        marker_pose, marker_scale = self.unit_conversion(marker_pose, marker_scale)


        icon_pose = self.create_icon_pose(marker_pose)

        # Publish the ship's state and icon
        self.ship_marker.publish_state(self.rviz_ship, self, marker_pose, marker_scale)
        self.icon_marker.publish_state(self.rviz_ship_icon, self, icon_pose, marker_scale=[1.0, 1.0 , 1.0], color=[0.0, 1.0, 1.0]) # Blue color
        with self.tl_lock:
            if self.tl_thread is None or not self.tl_thread.is_alive():
                self.tl_thread = threading.Thread(target=self.compute_TL_contour, args=(msg.position.latitude, msg.position.longitude))
                self.tl_thread.start()

        # PUblish ship's odom
        if previous_x != x: # Publish odom only when node is moving
            self.rviz_odometry(position = marker_pose[0:3], odom_rotation = odom_rotation, speed = msg.speed, publisher = self.rviz_odom)


    def rviz_odometry(self, position, odom_rotation, speed, publisher):
        offset = -1.32

        euler_angles = odom_rotation.as_euler('xyz', degrees=False)
        theta = euler_angles[2]

        offset_x = offset * np.cos(theta)
        offset_y = offset * np.sin(theta)

        adjusted_position_x = position[0] + offset_x
        adjusted_position_y = position[1] + offset_y
        adjusted_position_z = position[2]

        odom_quaternion = odom_rotation.as_quat()

        odom_msg = Odometry()
        odom_msg.header.frame_id = "map"
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.pose.pose.position.x = adjusted_position_x
        odom_msg.pose.pose.position.y = adjusted_position_y
        odom_msg.pose.pose.position.z = adjusted_position_z
        odom_msg.pose.pose.orientation.x = odom_quaternion[0]
        odom_msg.pose.pose.orientation.y = odom_quaternion[1]
        odom_msg.pose.pose.orientation.z = odom_quaternion[2]
        odom_msg.pose.pose.orientation.w = odom_quaternion[3]
        odom_msg.twist.twist.linear.x = speed
        odom_msg.twist.twist.angular.z = 0.0

        publisher.publish(odom_msg)


    def clock_callback(self, msg):
        clock_marker = Marker()
        clock_marker.header.frame_id = "map"
        clock_marker.header.stamp = self.get_clock().now().to_msg()
        clock_marker.type = Marker.TEXT_VIEW_FACING
        clock_marker.text = f"Time: {msg.clock.sec} mins\nETA: {self.ETA} mins"
        clock_marker.scale.z = 1.0  # Adjust the size of the text
        clock_marker.color.r = 1.0
        clock_marker.color.g = 1.0
        clock_marker.color.b = 1.0
        clock_marker.color.a = 1.0  # Fully opaque
        clock_marker.pose.position.x = 0.0
        clock_marker.pose.position.y = 0.0
        clock_marker.pose.position.z = 10.0  # Adjust the height where the text is displayed

        self.rviz_clock_marker.publish(clock_marker)

    def mammal_state_callback(self, msg, index):
        if self.verbose_logging:
            self.get_logger().info(
                f'I heard Mammal State: [{msg.position.latitude}, {msg.position.longitude}, {msg.position.depth}], '
                f'Angle: {msg.angle.heading}, Speed: {msg.speed} m/s')
        
        x, y = self.proj(msg.position.longitude, msg.position.latitude)
        depth = msg.position.depth

        # Getting the Heading Angle:
        previous_x = getattr(self, f"previous_x_{index}", x)
        previous_y = getattr(self, f"previous_y_{index}", y)
        heading_angle = self.compute_heading_angle(x, y, previous_x, previous_y)
        setattr(self, f"previous_x_{index}", x)
        setattr(self, f"previous_y_{index}", y)

        rotation = R.from_euler('xyz', [0.0, 0.0,heading_angle + 180], degrees=True)
        odom_rotation = R.from_euler('xyz', [0.0, 0.0, heading_angle], degrees=True)
        quaternion = rotation.as_quat()

        marker_pose = [x, y, -depth, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        marker_scale = [3.0,3.0,3.0]
        marker_pose, marker_scale = self.unit_conversion(marker_pose, marker_scale)

        icon_pose = self.create_icon_pose(marker_pose)

        # Publish the mammal's state and icons
        self.mammal_marker.publish_state(self.rviz_mammals[index], self, icon_pose, marker_scale, color=[1.0, 0.0, 0.0])
        self.icon_marker.publish_state(self.rviz_mammals_icon[index], self, icon_pose, marker_scale=[1.0, 1.0 , 1.0], color=[1.0, 0.0, 0.0])  # Red color

        if previous_x != x: # Publish odom only when node is moving
            self.rviz_odometry(position=marker_pose[0:3], odom_rotation=odom_rotation, speed=msg.speed, publisher=self.rviz_mammals_odom[index])

    def bathy_state_callback(self):
        rotation = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True)
        quaternion = rotation.as_quat()
        marker_pose = [0.0, 0.0, 0.0, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        marker_scale = [1.0, 1.0, 1.0]
        marker_pose, marker_scale = self.unit_conversion(marker_pose, marker_scale)

        self.bathy_marker.publish_state(self.rviz_bathy, self, marker_pose, marker_scale)

    def publish_fixed_data(self):
        self.bathy_state_callback()
        self.publish_pointcloud()
        self.rviz_occupancy_grid_publisher.publish(self.occupancy_grid_msg)

    def waypoints_callback(self, msg):
        waypoints_list = []
        for waypoint in msg.waypoints:
            x, y = self.proj(waypoint.longitude, waypoint.latitude)
            waypoints_list.append([x, y])

        # Convert the list to a 2D NumPy array
        self.waypoints = np.array(waypoints_list)

        if self.rviz_scale == 'km':
            self.waypoints = self.waypoints/1000

        path_msg = Path()
        path_msg.header.frame_id = "map"  # Set the appropriate frame ID
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(self.waypoints.shape[0]):
            x, y = self.waypoints[i]
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0  # Neutral orientation
            path_msg.poses.append(pose_stamped)

        self.rviz_path.publish(path_msg)

    def unit_conversion(self, marker_pose, marker_scale):

        if self.rviz_scale == 'km':
        # Only divide the first three elements of marker_pose by 1000
            marker_pose = [coord / 1000 if i < 3 else coord for i, coord in enumerate(marker_pose)]
            marker_scale = [scale / 1000 for scale in marker_scale]
        elif self.rviz_scale == 'm':
                pass
        else:
                raise ValueError("Invalid unit. Use 'km' for kilometers or 'm' for meters.")

        return marker_pose, marker_scale



def main(args=None):
    rclpy.init(args=args)
    node = Visualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
