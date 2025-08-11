import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import sys
sys.path.append('/workspaces/MUTE_v1/src/noise_propagation_model/interpolator/RBF')
sys.path.append('/workspaces/MUTE_v1/src/environment/environment')

from noise_propagation_model.models.RBF.utils_rbf import RBFInterpolatorCustom
from environment.py_polyshp import Polyshp

# Define an array of source locations (latitude, longitude)
source_locations = [
    (48.965, -123.224),
    (48.7835, -123.031),
    (48.7732, -123.059),
    (48.4132,-123.233)
]

# Define an array of receiver locations (latitude, longitude)
receiver_locations = [
    (48.64634312079397, -123.31305400775665)
]

# Define latitude and longitude limits
lat_min, lat_max = 48.0, 49.5
lon_min, lon_max = -125.0, -122.5

# Create a meshgrid of latitudes and longitudes
latitudes = np.linspace(lat_min, lat_max, 300)
longitudes = np.linspace(lon_min, lon_max, 300)
lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes)

# Flatten the meshgrid for model input
lat_mesh_flat = lat_mesh.ravel()
lon_mesh_flat = lon_mesh.ravel()

def filter_points_outside_polygons(lat_mesh_flat, lon_mesh_flat, polygon_list):
    """
    Remove points in the mesh that lie inside any polygon in the list.

    Args:
        lat_mesh_flat: Flattened array of latitudes from the mesh.
        lon_mesh_flat: Flattened array of longitudes from the mesh.
        polygon_list: DataFrame containing shapely polygons in the 'geometry' column.

    Returns:
        lat_filtered, lon_filtered: Filtered latitude and longitude arrays.
    """
    points = np.column_stack((lon_mesh_flat, lat_mesh_flat))
    filtered_indices = []

    # Extract the list of polygons from the 'geometry' column
    polygons = polygon_list['geometry']

    for i, (lon, lat) in enumerate(points):
        point = Point(lon, lat)
        if not any(poly.contains(point) for poly in polygons):
            filtered_indices.append(i)

    return lat_mesh_flat[filtered_indices], lon_mesh_flat[filtered_indices]

def overlay_polygons(ax, polygon_list):
    """
    Overlay polygons on a plot.

    Args:
        ax: Matplotlib axis.
        polygon_list: DataFrame containing shapely polygons in the 'geometry' column.
    """
    if polygon_list is not None:
        for poly in polygon_list['geometry']:
            if poly.geom_type == 'Polygon':
                x, y = poly.exterior.xy
                ax.plot(x, y, color='black', linewidth=1)
                ax.fill(x, y, color='grey', alpha=1.0)
            elif poly.geom_type == 'MultiPolygon':
                for sub_poly in poly.geoms:
                    x, y = sub_poly.exterior.xy
                    ax.plot(x, y, color='black', linewidth=1)
                    ax.fill(x, y, color='grey', alpha=1.0)

def create_model_input_and_predict(lat_mesh, lon_mesh, src_lat, src_lon, interpolator):
    """
    Create model input and predict transmission loss.

    Args:
        lat_mesh: Flattened array of latitudes from the mesh.
        lon_mesh: Flattened array of longitudes from the mesh.
        src_lat: Source latitude.
        src_lon: Source longitude.
        interpolator: RBF interpolator for prediction.

    Returns:
        predicted_tl: Predicted transmission loss.
    """
    # Create source latitude and longitude arrays repeated for each point in the mesh
    src_lat_col = np.full(lat_mesh.shape, src_lat)
    src_lon_col = np.full(lon_mesh.shape, src_lon)

    # Create the model input array
    model_input = np.column_stack((lat_mesh, lon_mesh, src_lat_col, src_lon_col))

    # Get the predicted transmission loss and return only the first column
    predicted_tl = interpolator(model_input)[:, 0]

    return predicted_tl


def print_tl_for_every_pair(source_locations, receiver_locations, interpolator):
    """
    Create model input and predict transmission loss.

    Args:
        source_locations: List of tuples containing source latitudes and longitudes.
        receiver_locations: List of tuples containing receiver latitudes and longitudes.
        interpolator: RBF interpolator for prediction.

    Returns:
        predicted_tl: Predicted transmission loss.
    """
    model_input = []
    src_rec_pairs = []

    # Pair each receiver with every source
    for src_lat, src_lon in source_locations:
        for rec_lat, rec_lon in receiver_locations:
            model_input.append([rec_lat, rec_lon, src_lat, src_lon])
            src_rec_pairs.append(((src_lat, src_lon), (rec_lat, rec_lon)))

    # Convert model input to a NumPy array
    model_input = np.array(model_input)

    # Get the predicted transmission loss
    predicted_tl = interpolator(model_input)[:, 0]  # Assuming only the first column is relevant

    # Print source, receiver, and transmission loss
    for i, ((src_lat, src_lon), (rec_lat, rec_lon)) in enumerate(src_rec_pairs):
        print(f"Source: ({src_lat}, {src_lon}), Receiver: ({rec_lat}, {rec_lon}), TL: {predicted_tl[i]:.2f} dB")

    return predicted_tl



def plot_transmission_loss(lat_mesh_flat, lon_mesh_flat, src_lat, src_lon, receiver_locations, predicted_tl, polygon_list):
    """
    Plot the transmission loss with source and receiver locations, and overlay polygons.

    Args:
        lat_mesh_flat: Flattened array of latitudes from the mesh.
        lon_mesh_flat: Flattened array of longitudes from the mesh.
        src_lat: Source latitude.
        src_lon: Source longitude.
        receiver_locations: List of receiver locations (latitude, longitude).
        predicted_tl: Predicted transmission loss.
        polygon_list: List of shapely polygons.
    """
    # Clip the predicted transmission loss for better visualization
    predicted_tl = np.clip(predicted_tl, 0, 200)

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    scatter = ax.scatter(lon_mesh_flat, lat_mesh_flat, c=predicted_tl, cmap='jet', s=10, label='Predicted TL')
    plt.colorbar(scatter, label='Transmission Loss (dB)')

    # Overlay polygons
    overlay_polygons(ax, polygon_list)

    # Plot the source location
    plt.scatter(src_lon, src_lat, color='green', marker='*', s=200, label='Source Location')

    # Plot the receiver locations
    for receiver_lat, receiver_lon in receiver_locations:
        plt.scatter(receiver_lon, receiver_lat, color='red', marker='o', s=100, label='Receiver Location')

    # Add labels and legend
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Transmission Loss and Locations')
    plt.legend()
    plt.show()


# Load the RBF interpolator model
model_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/RBF/model'
rbf_interpolator = RBFInterpolatorCustom.import_model(model_path)

# Load shoreline data and get polygon list
shp_data_path = '/workspaces/MUTE_v1/src/environment/shoreline_data'
shp_class = Polyshp(shp_data_path)
polygon_list = shp_class.get_polygons(lat_min, lat_max, lon_min, lon_max)

import time

start_time = time.time()

print("done1")
# Filter points outside polygons
lat_mesh_flat, lon_mesh_flat = filter_points_outside_polygons(lat_mesh_flat, lon_mesh_flat, polygon_list)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# Iterate over each source location and predict transmission loss
# for src_lat, src_lon in source_locations:
#     predicted_tl = create_model_input_and_predict(
#         lat_mesh_flat, lon_mesh_flat, src_lat, src_lon, rbf_interpolator
#     )
#     plot_transmission_loss(lat_mesh_flat, lon_mesh_flat, src_lat, src_lon, receiver_locations, predicted_tl, polygon_list)

print_tl_for_every_pair(source_locations, receiver_locations,interpolator=rbf_interpolator)

