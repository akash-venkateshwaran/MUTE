import numpy as np
import netCDF4 as nc
import os
from custom_interfaces.msg import HelperPosition
from custom_interfaces.srv import BathyService

class Bathymetry:
    def __init__(self, bathy_data_loc: str):
        self.data_loc = bathy_data_loc

    def create_random_bathy_points(self, request: BathyService.Request, num_points: int):
        latitudes = np.linspace(request.bathy_box_edge1.latitude, request.bathy_box_edge2.latitude, num_points)
        longitudes = np.linspace(request.bathy_box_edge1.longitude, request.bathy_box_edge2.longitude, num_points)

        depths = np.random.uniform(1000, 2000, size=(num_points, num_points))

        bathy_points = []
        for lat in latitudes:
            for lon in longitudes:
                point = HelperPosition()
                point.latitude = lat
                point.longitude = lon
                point.depth = depths.pop(0)
                bathy_points.append(point)

        return bathy_points

    def get_bathy(self, request: BathyService.Request):
        bathy_points = []
        min_lat, min_lon = request.bathy_box_edge1.latitude, request.bathy_box_edge1.longitude
        max_lat, max_lon = request.bathy_box_edge2.latitude, request.bathy_box_edge2.longitude

        if os.path.exists(self.data_loc):
            for file_name in os.listdir(self.data_loc):
                if file_name.endswith('.nc'):
                    file_path = os.path.join(self.data_loc, file_name)
                    with nc.Dataset(file_path, 'r') as nc_file:
                        latitudes = nc_file.variables['lat'][:]
                        longitudes = nc_file.variables['lon'][:]
                        depths = nc_file.variables['elevation'][:]

                    len_lat = len(latitudes)
                    len_long = len(longitudes)

                    depths = np.array(depths).flatten()
                    latitudes = np.repeat(latitudes, len_long)
                    longitudes = np.tile(longitudes, len_lat)

                    for lat, lon, depth in zip(latitudes, longitudes, depths):
                        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                            point = HelperPosition()
                            point.latitude = lat
                            point.longitude = lon
                            point.depth = float(depth)
                            bathy_points.append(point)

                else:
                    print(f"No bathymetry netCDF data found in the dir: {self.data_loc}")
        else:
            print(f"Directory not found: {self.data_loc}")

        return bathy_points
