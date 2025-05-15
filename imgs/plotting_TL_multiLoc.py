import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/workspaces/MELO_workspace/src/environment/environment')
import matplotlib as mpl
import matplotlib.colors as colors
from py_polyshp import Polyshp  # Importing Polyshp from py_polyshd.py



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

def plot_files(filenames, coord_path, tl_path, freq_value, z_value, shp_class, image_export):
    # Set font to 'serif'
    mpl.rcParams['font.family'] = 'serif'

    # Create normalization object to set vmin and vmax for all subplots
    norm = colors.Normalize(vmin=20, vmax=200)

    fig, axes = plt.subplots(1, len(filenames), figsize=(18, 5))
    if len(filenames) == 1:
        axes = [axes]

    for i, filename in enumerate(filenames):
        coord_file = os.path.join(coord_path, filename)
        tl_file = os.path.join(tl_path, filename)

        # Read and process the TL data for the given frequency
        if os.path.exists(coord_file) and os.path.exists(tl_file):

            tl_col, selected_freq = extract_tl_column(tl_file, freq_value)
            clubbed_data = process_coord_and_tl(coord_file, tl_col)

            # Find the rows corresponding to the given z_value (depth)
            z_index = np.where(clubbed_data[:, 2] == z_value)[0]
            z_data = clubbed_data[z_index, :]

            # Calculate lat, lon, and transmission loss for this file
            latitudes = z_data[:, 0].reshape(-1, int(np.sqrt(len(z_index))))  # Reshape for plotting
            longitudes = z_data[:, 1].reshape(-1, int(np.sqrt(len(z_index))))
            transmission_loss = z_data[:, 3].reshape(latitudes.shape)  # TL is the last column
            transmission_loss = np.clip(transmission_loss, 0, 200)

            ax = axes[i]
            # Use the norm object in contourf to apply the same color scale
            contour = ax.contourf(longitudes, latitudes, transmission_loss, cmap='jet', norm=norm)
            ax.set_title(f"File: {filename}, Freq: {freq_value} Hz at Z = {z_value} m")
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

        else:
            print(filename,"not found")

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.3)

    # Export the plot as a PDF to the specified path
    plt.savefig(f"{image_export}/TL_contours_filenames.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()


# Example usage
filenames = ['234_origin_lat48.257075545_lon-123.607003305.h5', 
             '264_origin_lat48.708851640_lon-123.244685824.h5',
             '354_origin_lat49.080969519_lon-123.344334980.h5']  # Example filenames

coord_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/Coordinate_data'
tl_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/TL_data'
image_export = '/workspaces/MELO_workspace/imgs'
freq_value = 100.0  # Fixed frequency of 100 Hz
z_value = 50.0  # Fixed depth of 50.0 m

shp_data_path = '/workspaces/MELO_workspace/src/environment/shoreline_data'  # Path to your shoreline shapefile directory
shp_class = Polyshp(shp_data_path)

plot_files(filenames, coord_path, tl_path, freq_value, z_value, shp_class, image_export)
