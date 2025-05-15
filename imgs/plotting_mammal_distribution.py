import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pickle
from environment.py_polyshp import Polyshp


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
    point_sizes = (point_sizes / max_count) * 100  # Scale sizes for visualization


    
    return max_count, point_sizes


# Paths
directory = '/workspaces/MELO_workspace'
file_paths = os.path.join(directory, "src", "mammals", "data", "population_data")
shoreline_paths = os.path.join(directory,"src", "mammals",  "data", "shoreline")
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

# Plotting
dpi = 300  # Dots per inch
fig, axes = plt.subplots(1, 2, figsize=(5000 / dpi, 2500 / dpi), dpi=dpi)  # Horizontal layout

fig.subplots_adjust(wspace=0.3)

# First subplot: Scatter plot of the CSV data points with fixed color and varying sizes
scatter = axes[0].scatter(all_data[:, 1], all_data[:, 0], c='#6f0f16', s=point_sizes, alpha=0.7)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

# Add shoreline to the first plot
if polygon_list is not None:
    for poly in polygon_list['geometry']:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            axes[0].plot(x, y, color='black', linewidth=1)
            axes[0].fill(x, y, color='grey', alpha=0.5)
        elif poly.geom_type == 'MultiPolygon':
            for sub_poly in poly.geoms:
                x, y = sub_poly.exterior.xy
                axes[0].plot(x, y, color='black', linewidth=1)
                axes[0].fill(x, y, color='grey', alpha=0.5)

# Second subplot: KDE density contour
levels = np.linspace(0, 1, 25)
contour = axes[1].contourf(Lon, Lat, Z_normalized, levels=levels, cmap=plt.cm.Reds)

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

# Set the latitude and longitude limits to be the same in both subplots
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)

# Color bar and labels for the second subplot
plt.colorbar(contour, ax=axes[1], orientation="vertical", label="Normalized Density")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

for ax in axes:
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Increase border thickness by 2

# Increase tick size and tick width for both subplots
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=ax.xaxis.get_label().get_size() + 2, width=2)  # Increase tick label size and thickness

# Export the figure
os.makedirs(image_export, exist_ok=True)
plt.savefig(f"{image_export}/Mammal_distribution_comparison.png", format='png', bbox_inches='tight')
plt.close()
