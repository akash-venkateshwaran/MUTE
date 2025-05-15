import os
import geopandas as gpd
import shapely
from shapely.geometry import box
import matplotlib.pyplot as plt


class Polyshp:
    """A class used to represent and manipulate shp polygon data.

    Attributes:
        data_loc (str): The file location of the shp data.
        data_shp (GeoDataFrame): A GeoDataFrame containing the loaded shapefile data.
    """

    def __init__(self, shp_data_loc: str):
        """Initializes the Polyshp class with a specific location for shp data.

        Args:
            shp_data_loc (str): The file location of the shp data.

        Notes:
            The constructor attempts to load the shp data from the specified location.
        """
        self.data_loc = shp_data_loc
        self.data_shp = self.load_shp_data()

    def load_shp_data(self):
        """Loads shp data from a shapefile.

        Returns:
            GeoDataFrame or None: The loaded shp data as a GeoDataFrame, or None if the data could not be loaded.

        Notes:
            The function searches for a '.shp' file in the specified directory.
        """
        if os.path.exists(self.data_loc):
            for file_name in os.listdir(self.data_loc):
                file_path = os.path.join(self.data_loc, file_name)

                if os.path.isfile(file_path) and file_name.endswith('.shp'):
                    shp_data = gpd.read_file(file_path)
                    return shp_data
            print(f"No shapefile data found in the directory: {self.data_loc}")
        else:
            print(f"Directory not found: {self.data_loc}")
            return None

    def get_polygons(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list:
        """Retrieves polygons from the shp data within specified latitude and longitude bounds.

        Args:
            lat_min (float): The minimum latitude.
            lat_max (float): The maximum latitude.
            lon_min (float): The minimum longitude.
            lon_max (float): The maximum longitude.

        Returns:
            GeoSeries: A GeoSeries of polygons that lie within the specified bounds.

        Notes:
            The function intersects the shp data with a bounding box created from the provided bounds using Geopandas.
        """
        if self.data_shp is not None:
            # bounding_box = box(lon_min, lat_min, lon_max, lat_max)
            # bounding_box_series = gpd.GeoSeries([bounding_box])
            # intersected_geometries = self.data_shp.geometry.intersection(bounding_box_series.iloc[0])
            # polygon_list = intersected_geometries[intersected_geometries.is_empty == False]

            bounding_box = box(lon_min, lat_min, lon_max, lat_max)
            bounding_box_gdf = gpd.GeoDataFrame(geometry=[bounding_box], crs=self.data_shp.crs)
            polygon_list = gpd.clip(self.data_shp, bounding_box_gdf)

            if polygon_list.empty:
                print('No polygons found within the specified range')
            else:
                return polygon_list
        else:
            print('shp data is empty')

    def plot_polygons(self, polygon_list):
        """Plots the given list of polygons.

        Args:
            polygon_list (GeoSeries): A GeoSeries of polygons to be plotted.

        Notes:
            The function creates a plot of the provided polygons, each with a different color and fills them.
        """
        if polygon_list is not None:
            gdf = gpd.GeoDataFrame(geometry=polygon_list)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.plot(ax=ax, edgecolor='black', cmap='tab20', linewidth=1, alpha=0.5)
            ax.set_title('Polygons')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            minx, miny, maxx, maxy = gdf.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            plt.savefig('/workspaces/MELO_workspace/src/environment/plot/polygons.png', bbox_inches='tight')
        else:
            print('No polygons to plot or polygon_list is not defined.')


    def format_polygons_for_service(self, polygon_list):
        """Formats a list of polygons for output via a ROS service: PolygonService

        Args:
            polygon_list (GeoSeries): A GeoSeries of polygons to be formatted.

        Returns:
            tuple: Three lists representing the latitudes, longitudes, and starting indices of polygons.

        Notes:
            The function formats the provided polygons into separate latitude and longitude lists, 
            along with a list of starting indices for each polygon.
        """
        latitudes = []
        longitudes = []
        polygon_starts = []  # Marks start index of every polygon
        current_index = 0  # Track the starting index of the polygon
        
        for idx, row in polygon_list.iterrows():
            polygon = row['geometry']

            # Check if the geometry is a MultiPolygon
            if isinstance(polygon, shapely.geometry.MultiPolygon):
                # Iterate through each Polygon in the MultiPolygon
                for single_polygon in polygon.geoms:
                    # Extract the exterior coordinates of the Polygon
                    coords = list(single_polygon.exterior.coords)
                    latitudes.extend([lat for lon, lat in coords]) 
                    longitudes.extend([lon for lon, lat in coords])
                    current_index += len(coords)
                    polygon_starts.append(current_index)
            elif isinstance(polygon, shapely.geometry.Polygon):
                # If the geometry is a single Polygon, handle it directly
                coords = list(polygon.exterior.coords)
                latitudes.extend([lat for lon, lat in coords])
                longitudes.extend([lon for lon, lat in coords])
                current_index += len(coords)
                polygon_starts.append(current_index)

        # Adjust polygon_starts to have the correct start indices
        polygon_starts = [0] + polygon_starts[:-1]
        return latitudes, longitudes, polygon_starts
    
    def export_polygons_as_shp(self, polygon_list, export_path: str):
        """Exports the given list of polygons to a shapefile at the specified path.

        Args:
            polygon_list (GeoDataFrame): A GeoDataFrame of polygons to be exported.
            export_path (str): The path where the shapefile will be saved.

        Notes:
            The function exports the provided polygons as a new shapefile to the specified location.
        """
        if polygon_list is not None and not polygon_list.empty:
            polygon_list.to_file(export_path, driver='ESRI Shapefile')
            print(f'Shapefile successfully exported to {export_path}')
        else:
            print('No polygons to export or polygon_list is empty.')