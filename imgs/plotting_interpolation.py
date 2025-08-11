import h5py
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspaces/MUTE_v1/src/noise_propagation_model/interpolator/RBF')
sys.path.append('/workspaces/MUTE_v1/src/environment/environment')


from noise_propagation_model.models.RBF.utils2 import RBFInterpolatorCustom
from environment.py_polyshp import Polyshp
plt.rcParams.update({'font.size': 14})  # Adjust font size as needed

def overlay_polygons(ax, polygon_list):
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

# Function to read .h5 files and extract data
def read_h5_file(file_path, dataset_name):
    with h5py.File(file_path, 'r') as h5_file:
        data = h5_file[dataset_name][:]
    return data

# Function to extract latitude and longitude from the filename
def extract_src_lat_lon(filename):
    match = re.search(r'_lat([-+]?[0-9]*\.?[0-9]+)_lon([-+]?[0-9]*\.?[0-9]+)', filename)
    if match:
        src_lat = float(match.group(1))
        src_lon = float(match.group(2))
        return src_lat, src_lon
    else:
        raise ValueError(f"Could not extract latitude and longitude from filename: {filename}")

# Function to plot results with predicted TL and absolute error
def plot_interpolation_results(coord_path, tl_path, filename, interpolator, shp_class):
    # Read lat, lon, and transmission loss data
    coord_file_path = os.path.join(coord_path, filename)
    tl_file_path = os.path.join(tl_path, filename)

    x_feature = read_h5_file(coord_file_path, 'x_feature')
    transmission_loss = read_h5_file(tl_file_path, 'TransmissionLoss')[:, 0]  # First column of transmission loss
    latitudes = x_feature[:, 3]
    longitudes = x_feature[:, 4]
    polygon_list = shp_class.get_polygons(np.min(latitudes), np.max(latitudes), np.min(longitudes), np.max(longitudes))

    # Extract source latitude and longitude from the filename and create column vectors
    src_lat, src_lon = extract_src_lat_lon(filename)
    src_lat_col = np.full(latitudes.shape, src_lat)
    src_lon_col = np.full(longitudes.shape, src_lon)

    # Create model_input for prediction using the actual latitudes and longitudes
    model_input = np.column_stack((latitudes, longitudes, src_lat_col, src_lon_col))
    predicted_tl = interpolator(model_input).ravel()

    # Clip the values for better visualization
    transmission_loss = np.clip(transmission_loss, 0, 200)
    predicted_tl = np.clip(predicted_tl, 0, 200)

    # Calculate absolute error between actual and predicted transmission loss
    abs_error = np.abs(transmission_loss - predicted_tl)
    abs_error = np.clip(abs_error, 0, 140)

    # Plotting the scatter plot, predicted scatter plot, and absolute error plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 30))

    # Set the axis border (spine) and tick mark thickness for each subplot
    for ax in axes:
        # Increase border thickness
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # Adjust border thickness as needed

        # Increase tick mark thickness
        ax.tick_params(width=2)  # Adjust tick mark thickness as needed

    # Scatter plot for actual transmission loss
    scatter_actual = axes[0].scatter(
        longitudes, latitudes, c=transmission_loss, cmap='jet', s=5
    )
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    overlay_polygons(axes[0], polygon_list)

    # Scatter plot for predicted transmission loss
    scatter_predicted = axes[1].scatter(
        longitudes, latitudes, c=predicted_tl, cmap='jet', s=5
    )
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    overlay_polygons(axes[1], polygon_list)

    # Scatter plot for absolute error
    scatter_error = axes[2].scatter(
        longitudes, latitudes, c=abs_error, cmap='Greys', s=5
    )
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    overlay_polygons(axes[2], polygon_list)
    # Add a horizontal color bar below the last subplot
    # cbar = fig.colorbar(scatter_error, ax=axes[2], orientation='vertical', pad=0.2)  # pad adjusts the distance from the plot
    # cbar.set_label('Absolute Error')  # Label for the color bar

    # Adjust layout and save the figure as a high-resolution PNG
    plt.tight_layout()
    plt.savefig('/workspaces/MUTE_v1/imgs/interpolation_comparisons.png', format='png', dpi=300)
    plt.close()

coord_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/bellhop/Coordinate_data'
tl_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/bellhop/TL_data'
shp_data_path = '/workspaces/MUTE_v1/src/environment/shoreline_data'  # Path to your shoreline shapefile directory
model_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq'

# Define 'filtered' folder paths
filtered_coord_path = os.path.join(coord_path, 'filtered','one_z_plain')
filtered_tl_path = os.path.join(tl_path, 'filtered','one_z_plain')

rbf_interpolator = RBFInterpolatorCustom.import_model(model_path)
filename = '334_origin_lat48.821783203_lon-122.931561706.h5'

shp_data_path = '/workspaces/MUTE_v1/src/environment/shoreline_data'
shp_class = Polyshp(shp_data_path)
# Plot results using the interpolator
plot_interpolation_results(filtered_coord_path, filtered_tl_path, filename, rbf_interpolator, shp_class)

