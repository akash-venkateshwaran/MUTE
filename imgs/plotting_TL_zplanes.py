import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspaces/MELO_workspace/src/environment/environment')
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from py_polyshp import Polyshp  # Importing Polyshp from py_polyshd.py


def read_h5(file_path):
    try:
        with h5py.File(file_path, 'r') as h5_file:
            print(f"\nFile: {file_path}")
            print("Keys available in the file:")
            for key in h5_file.keys():
                print(f"- {key}, shape: {h5_file[key].shape}")
            
            print("\nMetadata:")
            for attr in h5_file.attrs:
                print(f"- {attr}: {h5_file.attrs[attr]}")
            
    except OSError:
        print(f"Could not open file: {file_path}")

def extract_tl_column(tl_file, freq_value):
    try:
        with h5py.File(tl_file, 'r') as h5_file:
            print(f"\nExtracting data from file: {tl_file}")
            freq = h5_file['Frequency'][:].flatten()  # Flatten to ensure it's a 1D array
            # Find the index of the provided frequency value
            col = np.where(freq == freq_value)[0][0]

            transmission_loss = h5_file['TransmissionLoss'][:, col]
            selected_freq = freq[col]
            
            print(f"Selected frequency: {selected_freq}")
            print(f"Transmission Loss column for frequency {freq_value} extracted, shape: {transmission_loss.shape}")
            
            return transmission_loss, selected_freq
            
    except OSError:
        print(f"Could not open file: {tl_file}")
    except IndexError:
        print(f"Frequency {freq_value} not found in the file.")

def process_coord_and_tl(coord_file, tl_col):
    try:
        with h5py.File(coord_file, 'r') as h5_file:
            print(f"\nProcessing data from file: {coord_file}")
            
            dataset_name = list(h5_file.keys())[0]
            coord_data = h5_file[dataset_name][:, 3:]  # Remove first three columns (keeping lat, long, z)

            clubbed_data = np.hstack((coord_data, tl_col.reshape(-1, 1)))
            
            print(f"Clubbed data shape: {clubbed_data.shape}")
            return clubbed_data
            
    except OSError:
        print(f"Could not open file: {coord_file}")
def plot_z_plane(data, z_planes, selected_freq, shp_class, image_export):
    
    first_row = data[0]
    lat, long, z = first_row[0], first_row[1], first_row[2]

    # Calculate sizes for lat, z, and long as given
    lat_count = np.sum(data[:, 0] == lat)
    z_count = np.sum(data[:, 2] == z)

    lat_size = int(data.shape[0] / lat_count)
    z_size = int(data.shape[0] / z_count)
    long_size = int(data.shape[0] / (lat_size * z_size))

    # Create normalization object to set vmin and vmax for all subplots
    norm = colors.Normalize(vmin=20, vmax=200)

    fig, axes = plt.subplots(1, len(z_planes), figsize=(15, 5))
    if len(z_planes) == 1:
        axes = [axes]

    # Loop over z_planes to generate the contour plots
    for i, z_value in enumerate(z_planes):
        
        z_index = np.where(data[:, 2] == z_value)[0]
        z_data = data[z_index, :]

        latitudes = z_data[:, 0].reshape(lat_size, long_size)
        longitudes = z_data[:, 1].reshape(lat_size, long_size)
        transmission_loss = z_data[:, 3].reshape(lat_size, long_size)
        transmission_loss = np.clip(transmission_loss, 0, 200)

        ax = axes[i]
        # Use the norm object in contourf to apply the same color scale
        contour = ax.contourf(longitudes, latitudes, transmission_loss, cmap='jet', norm=norm)
        ax.set_title(f"Z-plane {i + 1} (z = {z_value} m, Freq = {selected_freq} Hz)")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Retrieve polygons (shorelines)
        polygon_list = shp_class.get_polygons(np.min(latitudes), np.max(latitudes), np.min(longitudes), np.max(longitudes))

        if polygon_list is not None:
            for poly in polygon_list['geometry']:
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color='black', linewidth=1)
                    # Fill the polygon with light grey
                    ax.fill(x, y, color='grey', alpha=0.5)
                elif poly.geom_type == 'MultiPolygon':
                    # Iterate over each polygon in the MultiPolygon
                    for sub_poly in poly.geoms:
                        x, y = sub_poly.exterior.xy
                        ax.plot(x, y, color='black', linewidth=1)
                        # Fill the MultiPolygon with light grey
                        ax.fill(x, y, color='grey', alpha=0.5)

    # Create a shared colorbar for all subplots
    # cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', fraction=0.2, pad=0.04)
    # cbar.set_label('Transmission Loss (dB)', fontsize=12)

    # Export the plot as a PDF to the specified path
    plt.savefig(f"{image_export}/temp.png", format='png', dpi=300)
    plt.show()


def search_and_analyze_h5(filename, coord_path, tl_path, freq_value, z_planes, shp_class, image_export):
    if not filename.endswith('.h5'):
        filename += '.h5'
    
    coord_file = os.path.join(coord_path, filename)
    tl_file = os.path.join(tl_path, filename)
    
    if os.path.exists(coord_file):
        read_h5(coord_file)
    else:
        print(f"File not found in {coord_path}")

    if os.path.exists(tl_file):
        read_h5(tl_file)
    else:
        print(f"File not found in {tl_path}")

    if os.path.exists(tl_file):
        tl_col, selected_freq = extract_tl_column(tl_file, freq_value)
    else:
        print(f"TL file not found in {tl_path}")

    if os.path.exists(coord_file):
        clubbed_data = process_coord_and_tl(coord_file, tl_col)
        plot_z_plane(clubbed_data, z_planes, selected_freq, shp_class, image_export)
    else:
        print(f"Coord file not found in {coord_path}")

# Example usage
filename = '354_origin_lat49.080969519_lon-123.344334980'
coord_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/Coordinate_data'
tl_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/TL_data'
image_export = '/workspaces/MELO_workspace/imgs'
freq_value = 100.0  # The column index to pick from TransmissionLoss
z_planes = [0.0, 50.0, 100.0]  # Number of z planes to plot

shp_data_path = '/workspaces/MELO_workspace/src/environment/shoreline_data'  # Path to your shoreline shapefile directory
shp_class = Polyshp(shp_data_path)

search_and_analyze_h5(filename, coord_path, tl_path, freq_value, z_planes, shp_class, image_export)
