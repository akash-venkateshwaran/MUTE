import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspaces/MELO_workspace/src/environment/environment')
import matplotlib as mpl
import matplotlib.colors as colors
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

def extract_tl_columns(tl_file, freq_values):
    try:
        with h5py.File(tl_file, 'r') as h5_file:
            print(f"\nExtracting data from file: {tl_file}")
            freq = h5_file['Frequency'][:].flatten()  # Flatten to ensure it's a 1D array

            # Initialize list to hold the transmission loss columns
            tl_columns = []

            # Find the indices for the given freq_values and stack them column-wise
            for freq_value in freq_values:
                col = np.where(freq == freq_value)[0][0]
                transmission_loss = h5_file['TransmissionLoss'][:, col]
                tl_columns.append(transmission_loss)

                print(f"Selected frequency: {freq[col]}")
                print(f"Transmission Loss column for frequency {freq_value} extracted, shape: {transmission_loss.shape}")

            # Stack the columns to form a matrix (rows: data points, columns: frequencies)
            stacked_tl = np.column_stack(tl_columns)
            print(f"Stacked Transmission Loss shape: {stacked_tl.shape}")

            return stacked_tl, freq_values
            
    except OSError:
        print(f"Could not open file: {tl_file}")
    except IndexError:
        print(f"Frequency value not found in the file.")

def process_coord_and_tl(coord_file, tl_data):
    try:
        with h5py.File(coord_file, 'r') as h5_file:
            print(f"\nProcessing data from file: {coord_file}")
            
            dataset_name = list(h5_file.keys())[0]
            coord_data = h5_file[dataset_name][:, 3:]  # Remove first three columns (keeping lat, long, z)

            # Append the stacked transmission loss data as additional columns
            clubbed_data = np.hstack((coord_data, tl_data))
            
            print(f"Clubbed data shape: {clubbed_data.shape}")
            return clubbed_data
            
    except OSError:
        print(f"Could not open file: {coord_file}")

def plot_multiple_freq(data, freq_values, z_value, shp_class, image_export):
    # Set font to 'serif'
    mpl.rcParams['font.family'] = 'serif'

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

    fig, axes = plt.subplots(1, len(freq_values), figsize=(18, 5))
    if len(freq_values) == 1:
        axes = [axes]

    # Find all rows that match the given z_value
    z_index = np.where(data[:, 2] == z_value)[0]
    z_data = data[z_index, :]

    # Loop over freq_values to generate the contour plots for each column
    for i, freq_value in enumerate(freq_values):
        latitudes = z_data[:, 0].reshape(lat_size, long_size)
        longitudes = z_data[:, 1].reshape(lat_size, long_size)
        transmission_loss = z_data[:, 3 + i].reshape(lat_size, long_size)  # Select the i-th frequency column
        transmission_loss = np.clip(transmission_loss, 0, 200)

        ax = axes[i]
        # Use the norm object in contourf to apply the same color scale
        contour = ax.contourf(longitudes, latitudes, transmission_loss, cmap='jet', norm=norm)
        ax.set_title(f"Freq: {freq_value} Hz at Z = {z_value} m")
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

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3)

    # Export the plot as a PDF to the specified path
    plt.savefig(f"{image_export}/TL_contours_multiple_freqs.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()


def search_and_analyze_h5_multiple_freq(filename, coord_path, tl_path, freq_values, z_value, shp_class, image_export):
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
        # Extract the stacked transmission loss data for multiple frequencies
        tl_data, selected_freqs = extract_tl_columns(tl_file, freq_values)
        
        if os.path.exists(coord_file):
            clubbed_data = process_coord_and_tl(coord_file, tl_data)
            plot_multiple_freq(clubbed_data, freq_values, z_value, shp_class, image_export)
        else:
            print(f"Coord file not found in {coord_path}")
    else:
        print(f"TL file not found in {tl_path}")


# Example usage
filename = '354_origin_lat49.080969519_lon-123.344334980'
coord_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/Coordinate_data'
tl_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/TL_data'
image_export = '/workspaces/MELO_workspace/imgs'
freq_values = [12.5, 100.0, 1000.0]  # Frequencies to plot
z_value = 50.0  # Fixed depth for all subplots

shp_data_path = '/workspaces/MELO_workspace/src/environment/shoreline_data'  # Path to your shoreline shapefile directory
shp_class = Polyshp(shp_data_path)

search_and_analyze_h5_multiple_freq(filename, coord_path, tl_path, freq_values, z_value, shp_class, image_export)
