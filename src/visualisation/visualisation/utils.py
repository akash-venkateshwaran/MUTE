import os
from visualization_msgs.msg import Marker

import numpy as np
from netCDF4 import Dataset
from pyproj import Proj
from scipy.ndimage import gaussian_filter
from visualisation import surf2stl

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


def create_bathy_stl(proj: Proj, output_path, unit='m'):
    print('Creating bathymetry STL')

    gebco_file = str(src_dir / 'environment' / 'bathy_data' / 'gebco_2023_n49.9814_s46.8063_w-125.2002_e-121.5601.nc')

    nc = Dataset(gebco_file)
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    lat_coord = np.repeat(lat, len(lon))
    lon_coord = np.tile(lon, len(lat))

    ele = nc.variables['elevation'][:].astype(float)
    
    if unit == 'km':
        ele = ele / 1000  # Convert to kilometers
    elif unit == 'm':
        pass  # Already in meters, no conversion needed
    else:
        raise ValueError("Invalid unit. Use 'km' for kilometers or 'm' for meters.")

    x, y = proj(lon_coord, lat_coord)

    x_mesh = np.reshape(x, (len(lat), len(lon)))
    y_mesh = np.reshape(y, (len(lat), len(lon)))

    ele_smoothed = gaussian_filter(ele, sigma=1)

    output_file_stl = output_path
    surf2stl.write(output_file_stl, x_mesh, y_mesh, ele_smoothed)

    print(f'STL file saved to {output_file_stl}')


class RVizMarker:
    def __init__(self, mesh_path, marker_id = 0):
        self.mesh_path = mesh_path
        self.marker_id = marker_id

    def publish_state(self, publisher, node, marker_pose, marker_scale,color=[1.0, 1.0, 1.0]):
        if os.path.exists(self.mesh_path):
            # node.get_logger().info(f'Loading mesh file: {self.mesh_path}')

            # Create a Marker
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = node.get_clock().now().to_msg()
            marker.ns = "mesh"
            marker.id = self.marker_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.mesh_resource = f'file://{self.mesh_path}'  # Use file URI for mesh_path
            marker.mesh_use_embedded_materials = True

            # Set the position and orientation
            marker.pose.position.x = marker_pose[0]
            marker.pose.position.y = marker_pose[1]
            marker.pose.position.z = marker_pose[2]
            marker.pose.orientation.x = marker_pose[3]
            marker.pose.orientation.y = marker_pose[4]
            marker.pose.orientation.z = marker_pose[5]
            marker.pose.orientation.w = marker_pose[6]

            # Set the scale of the mesh
            marker.scale.x = marker_scale[0]
            marker.scale.y = marker_scale[1]
            marker.scale.z = marker_scale[2]

            # Set a color (optional, if you want to override the mesh materials)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0  # Don't forget to set the alpha channel to a non-zero value

            # Publish the marker
            publisher.publish(marker)
        else:
            node.get_logger().warn(f'Mesh file does not exist: {self.mesh_path}')