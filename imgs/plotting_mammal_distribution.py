import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity
import pickle
from environment.py_polyshp import Polyshp
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
font_size = 16
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': font_size})

def check_duplicate_points(csv_data):
    """
    Checks for duplicate latitude and longitude points in the given data and returns the maximum count.

    Parameters:
    csv_data (numpy array): 2D array where the first column is latitude and the second column is longitude.

    Returns:
    max_count (int): Maximum count of duplicate points.
    sizes (numpy array): Array of sizes for scatter plot based on duplicate counts.
    """
    # Round latitudes and longitudes to avoid floating-point precision issues
    rounded_data = np.round(csv_data, decimals=6)
    
    # Find unique rows and their counts
    unique_points, counts = np.unique(rounded_data, axis=0, return_counts=True)
    
    # Determine the size of each point for the scatter plot based on counts
    max_count = np.max(counts)
    point_sizes = np.array([counts[np.where((unique_points == point).all(axis=1))[0][0]] for point in rounded_data])
    point_sizes = (point_sizes / max_count) * 75  # Scale sizes for visualization

    return max_count, point_sizes


def process_depth_data(directory):
    """
    Process depth data for histogram plotting.
    
    Parameters:
    directory (str): Base directory path
    
    Returns:
    tuple: (samples, bin_edges, bin_centers, interpolated_counts, kde)
    """
    # Paths
    csv_file = os.path.join(directory, "src", "mammals", "data", "depth_data", "species_1.csv")
    kde_model_path = os.path.join(directory, "src", "mammals", "kde_models", "depth_data", "kde_model_species_0.pickle")
    
    # Load depth data
    data = pd.read_csv(csv_file)
    
    # Extract depths and counts
    depths = data['Depth (m)'].values
    sample_counts = data['Counts'].values
    
    # Generate samples based on counts for the KDE and histogram
    samples = []
    for depth, count in zip(depths, sample_counts):
        samples.extend([depth] * count)
    samples = np.array(samples)
    
    # Load KDE from pickle file
    try:
        with open(kde_model_path, 'rb') as f:
            kde = pickle.load(f)
        print("Loaded depth KDE model from file.")
    except FileNotFoundError:
        print("Depth KDE model file not found.")
        kde = None
    
    # Define bin edges for the range 0.5 to 100, with 20 bins per log interval
    bin_edges = []
    for i in range(int(np.log10(0.5)), int(np.log10(100))):
        bin_edges.extend(np.logspace(i, i + 1, 21)[:-1])
    bin_edges.append(100)  # Ensure the last bin edge reaches exactly 100
    
    # Convert bin_edges to a NumPy array for numerical operations
    bin_edges = np.array(bin_edges)
    
    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Centers of each bin
    
    # Interpolate count values to match each bin center
    log_depths = np.log10(depths)
    interpolator = interp1d(log_depths, sample_counts, kind='linear', fill_value="extrapolate")
    interpolated_counts = interpolator(np.log10(bin_centers))
    
    return samples, bin_edges, bin_centers, interpolated_counts, kde


# Paths
directory = '/workspaces/MUTE_v1'
file_paths = os.path.join(directory, "src", "mammals", "data", "population_data")
shoreline_paths = os.path.join(directory,"src", "environment",  "shoreline_data")
model_path = os.path.join(directory, "src", "mammals", "kde_models", "population_data", "kde_model_species_0.pickle")
image_export = os.path.join(directory, "imgs")

csv_file = os.path.join(file_paths, 'species_1.csv')
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)  # Skip headers

# Extract latitude and longitude
all_data = data[:, :2]  # First column is latitude, second column is longitude
max_count, point_sizes = check_duplicate_points(all_data)

print(max_count,point_sizes)

# Define buffer and calculate bounding box
buffer = 0.1  # Modify this buffer as needed
lat_min, lon_min = np.min(all_data, axis=0) - buffer
lat_max, lon_max = np.max(all_data, axis=0) + buffer

print(f"Min Lat = {lat_min} Lon = {lon_min}")
print(f"Max Lat = {lat_max} Lon = {lon_max}")

# Load shoreline and retrieve polygons within bounding box
shoreline = Polyshp(shp_data_loc=shoreline_paths)
polygon_list = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)

# Set up mesh grid for KDE evaluation
lat_grid = np.linspace(lat_min, lat_max, 100)
lon_grid = np.linspace(lon_min, lon_max, 100)
Lon, Lat = np.meshgrid(lon_grid, lat_grid)
grid_points = np.vstack([Lat.ravel(), Lon.ravel()]).T * np.pi / 180.0  # Convert to radians

# Load or fit KDE model
try:
    with open(model_path, 'rb') as f:
        kde = pickle.load(f)
    print("Loaded KDE model from file.")
except FileNotFoundError:
    print("KDE model not found, creating a new one.")
    kde = KernelDensity(bandwidth=0.04, metric="haversine", kernel="gaussian", algorithm="ball_tree")
    kde.fit(all_data * np.pi / 180.0)  # Convert lat/long to radians for spherical KDE
    with open(model_path, 'wb') as f:
        pickle.dump(kde, f)

# Evaluate KDE on grid
Z = np.exp(kde.score_samples(grid_points)).reshape(Lat.shape)
Z_normalized = Z / Z.max()  # Normalize to [0, 1] range

# Process depth data for third subplot
samples, bin_edges, bin_centers, interpolated_counts, depth_kde = process_depth_data(directory)

# Plotting
dpi = 300  # Dots per inch
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)  # Three horizontal subplots

# Adjust subplots to make room for colorbar at top and keep them compact
fig.subplots_adjust(left=0.06, right=0.94, top=0.75, bottom=0.15, wspace=0.3)

print(all_data.shape)
print(point_sizes.shape)

# First subplot: Scatter plot of the CSV data points with fixed color and varying sizes
scatter = axes[0].scatter(all_data[:, 1], all_data[:, 0], c='#6f0f16', s=point_sizes, alpha=0.7)
axes[0].set_xlabel("Longitude", fontsize=14)
axes[0].set_ylabel("Latitude", fontsize=14)

# Add shoreline to the first plot
if polygon_list is not None:
    for poly in polygon_list['geometry']:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            axes[0].plot(x, y, color='black', linewidth=1)
            axes[0].fill(x, y, color='lightgrey', alpha=0.5)
        elif poly.geom_type == 'MultiPolygon':
            for sub_poly in poly.geoms:
                x, y = sub_poly.exterior.xy
                axes[0].plot(x, y, color='black', linewidth=1)
                axes[0].fill(x, y, color='lightgrey', alpha=0.5)

# Second subplot: KDE density contour
levels = np.linspace(0, 1, 25)
contour = axes[1].contourf(Lon, Lat, Z_normalized, levels=levels, cmap='gist_heat_r')

# Plot the shoreline as a grey mask in the second subplot
if polygon_list is not None:
    for poly in polygon_list['geometry']:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            axes[1].plot(x, y, color='black', linewidth=1)
            axes[1].fill(x, y, color='grey', alpha=0.5)
        elif poly.geom_type == 'MultiPolygon':
            for sub_poly in poly.geoms:
                x, y = sub_poly.exterior.xy
                axes[1].plot(x, y, color='black', linewidth=1)
                axes[1].fill(x, y, color='grey', alpha=0.5)

# Third subplot: Depth distribution histogram
axes[2].hist(bin_centers, bins=bin_edges, weights=interpolated_counts, color='lightgrey', edgecolor='black', linewidth=1)
axes[2].set_xscale('log')
axes[2].set_xlim(1.0, 100)
axes[2].set_xlabel("Depth (m) (log scale)", fontsize=14)
axes[2].set_ylabel("Interpolated Counts", color='black', fontsize=14)
axes[2].tick_params(axis='y', labelcolor='black')

# Create a secondary y-axis for the KDE in the depth distribution subplot
ax2_kde = axes[2].twinx()
if depth_kde:
    depth_range = np.linspace(1.0, 100, 1000)[:, np.newaxis]
    log_density = depth_kde.score_samples(depth_range)
    density = np.exp(log_density)
    
    ax2_kde.plot(depth_range, density, color='black', lw=2, label='KDE')  # Set line width to 4
    ax2_kde.set_ylabel("Density", color='black', fontsize=14)
    ax2_kde.tick_params(axis='y', labelcolor='black')

# Set the latitude and longitude limits to be the same in both geographical subplots
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)

# Color bar horizontal above the figure, matching width of second subplot
# Calculate the position and width to match the second subplot
subplot_pos = axes[1].get_position()
cbar_ax = fig.add_axes([subplot_pos.x0, 0.8, subplot_pos.width, 0.03])  # Match second subplot width
cbar = plt.colorbar(contour, cax=cbar_ax, orientation="horizontal")
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label("Normalized Density", fontsize=12)
# Set tick resolution to 0.25
cbar.set_ticks(np.arange(0, 1.1, 0.25))

# Remove y-axis text (tick labels) from the second subplot
axes[1].set_yticklabels([])
axes[1].set_xlabel("Longitude", fontsize=14)
axes[1].set_ylabel("")  # Remove y-axis label

# Increase tick size for all subplots
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=12)  # Set font size to 12


# Update axes labels and tick sizes
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    for label in ax.get_xticklabels():
        label.set_fontsize(font_size)
    for label in ax.get_yticklabels():
        label.set_fontsize(font_size)

# Colorbar label and ticks
cbar.set_label("Normalized Density", fontsize=font_size)
cbar.ax.tick_params(labelsize=font_size)

# Secondary y-axis on KDE plot
if depth_kde:
    ax2_kde.yaxis.label.set_size(font_size)
    ax2_kde.tick_params(axis='y', labelsize=font_size)

# Export the figure as PDF
os.makedirs(image_export, exist_ok=True)
plt.savefig(f"{image_export}/Mammal_distribution_comparison.pdf", format='pdf', bbox_inches='tight')
plt.close()