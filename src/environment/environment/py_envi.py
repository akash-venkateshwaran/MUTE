import rclpy
from rclpy.node import Node
from custom_interfaces.srv import BathyService, PolygonService
from environment.py_bathy import Bathymetry
from environment.py_polyshp import Polyshp
import os
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
base_dir = src_dir / "environment"

class Envi(Node):
    def __init__(self):
        super().__init__('envi_node')
        self.setup_logger()
        
        self.bathy_data_loc = os.path.join(base_dir, 'bathy_data')
        self.bathy_obj = Bathymetry(self.bathy_data_loc)
        self.service = self.create_service(BathyService, 'bathy_service', self.handle_bathy_request)

        self.shoreline_data_loc = os.path.join(base_dir, 'shoreline_data')
        self.shoreline_obj = Polyshp(self.shoreline_data_loc)
        self.shoreline_service = self.create_service(PolygonService, 'shoreline_service', self.handle_shoreline_request)

        self.shiplane_data_loc = os.path.join(base_dir, 'ship_lanes_data')
        self.shiplane_obj = Polyshp(self.shiplane_data_loc)
        self.shiplane_service = self.create_service(PolygonService, 'shiplane_service', self.handle_shiplane_request)


    def setup_logger(self):
        self.get_logger().info("Envi Node is started!")

    def handle_bathy_request(self, request, response):
        response.bathy_points = self.bathy_obj.get_bathy(request)
        self.get_logger().info(f"Responded to Bathy Service Request. {len(response.bathy_points)} points. \n\n")
        return response

    def handle_shoreline_request(self, request, response):
        # Convert HelperPosition to bounding box edges
        min_lat, min_lon = request.min_box_edge.latitude, request.min_box_edge.longitude
        max_lat, max_lon = request.max_box_edge.latitude, request.max_box_edge.longitude

        polygon_list = self.shoreline_obj.get_polygons(min_lat, max_lat, min_lon, max_lon)

        # Format the polygons for the response
        formatted_data = self.shoreline_obj.format_polygons_for_service(polygon_list)
        response.latitudes, response.longitudes, response.polygon_starts = formatted_data

        self.get_logger().info(f"Responded to Shoreline Service Request with {len(response.polygon_starts)} polygons.")
        return response

    def handle_shiplane_request(self, request, response):
        # Convert HelperPosition to bounding box edges
        min_lat, min_lon = request.min_box_edge.latitude, request.min_box_edge.longitude
        max_lat, max_lon = request.max_box_edge.latitude, request.max_box_edge.longitude

        polygon_list = self.shiplane_obj.get_polygons(min_lat, max_lat, min_lon, max_lon)

        # Format the polygons for the response
        formatted_data = self.shiplane_obj.format_polygons_for_service(polygon_list)
        response.latitudes, response.longitudes, response.polygon_starts = formatted_data

        self.get_logger().info(f"Responded to Shiplane Service Request with {len(response.polygon_starts)} polygons.")
        return response

def main(args=None):
    rclpy.init(args=args)
    envi_obj = Envi()
    rclpy.spin(envi_obj)
    envi_obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
