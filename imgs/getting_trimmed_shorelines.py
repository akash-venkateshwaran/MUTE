import os
import sys

# Add path to the audiogram function
sys.path.append('/workspaces/MELO_workspace/src/environment/environment')
from py_polyshp import Polyshp  # Replace with the actual module name where Polyshp is defined

# Define the directory containing the shoreline data
directory = '/workspaces/MELO_workspace'
shoreline_paths = os.path.join(directory, "src", "mammals", "data", "shoreline")

# Initialize the Polyshp class
shoreline = Polyshp(shp_data_loc=shoreline_paths)

# Define the latitude and longitude limits
lat_min = 47.12334709947469
lat_max = 50.23939478923854
lon_min = -126.57708629145905
lon_max = -122.33013116632014

# Get the polygons within the specified limits
polygon_list = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)

# Define the export path
export_path = '/workspaces/MELO_workspace/imgs/shoreline_filtered.shp'

# Export the polygons to the specified path
shoreline.export_polygons_as_shp(polygon_list, export_path)
