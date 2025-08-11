import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Paths to the CSV files based on the directory structure
csv_files = [
    '/workspaces/MUTE_v1/src/path_planning/case_studies/Case_1/case_study.csv',
    '/workspaces/MUTE_v1/src/path_planning/case_studies/Case_3/case_study.csv'
]
case_names = ['Case 1', 'Case 3']
ship_plane_path = '/workspaces/MUTE_v1/src/environment/ship_lanes_data/new_ship_lanes_polys.shp'

# Colors for each case - standard blue and red
colors = ['blue', 'red']

# Load and store the data (only Case 1 and Case 3)
data = [pd.read_csv(file, names=['Voyage Time', 'Speed', 'Longitude', 'Latitude']) for file in csv_files]

# Extract latitude and longitude data for geoplotting
locations = {
    'Victoria': (48.4284, -123.3656),
    'Vancouver': (49.2827, -123.1207),
    'Port Angeles': (48.1181, -123.4307),
    'Boundary Pass': (48.7078, -123.1851),
    'Haro Strait': (48.5669, -123.2344),
    'Port Renfrew': (48.5530, -124.4223)
}

# Create a GeoDataFrame for these fixed locations
locations_df = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in locations.values()], index=locations.keys())

# Set the figure size and create subplots with equal spacing
fig = plt.figure(figsize=(20, 8))

# Create subplots with manual positioning for equal space - adjusted for compact fit
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

# Adjust subplot parameters for better space utilization and compact fit
plt.subplots_adjust(left=0.12, bottom=0.08, right=0.95, top=0.95, hspace=0.25)

# Load shoreline/islands data with black border
shoreline = gpd.read_file('/workspaces/MUTE_v1/src/environment/shoreline_data')

shoreline.plot(ax=ax1, color='lightgrey', edgecolor='black')  # Add black border
ax1.set_aspect('auto')  # Override geopandas default equal aspect ratio

# Plot paths for each case with matching color start and end markers
for i, df in enumerate(data):
    ax1.plot(df['Longitude'], df['Latitude'], linestyle='-', color=colors[i], label=case_names[i])
    # Start (Departure) and End (Destination) markers in the same color as path
    ax1.plot(df['Longitude'].iloc[0], df['Latitude'].iloc[0], 'o', color=colors[i], markersize=10)
    ax1.plot(df['Longitude'].iloc[-1], df['Latitude'].iloc[-1], 's', color=colors[i], markersize=10)

# Plot key locations with labels
locations_df.plot(ax=ax1, color='black', marker='^')
for name, (lat, lon) in locations.items():
    ax1.text(lon + 0.02, lat, name, fontsize=10, fontfamily="Times New Roman")  # Offset for visibility

ship_plane = gpd.read_file(ship_plane_path)
ship_plane_remainder = ship_plane.drop(ship_plane.index[0])
ship_plane_remainder.plot(ax=ax1, color='blue', alpha=0.3, edgecolor='black')

# Ensure aspect ratio allows full use of subplot space
ax1.set_aspect('auto')

# Configure map limits and labels
ax1.set_xlim(-125.72143254685919, -121.78283374169551)
ax1.set_ylim(47.95629176783697, 49.5104276496541)
ax1.set_xlabel('Longitude', fontfamily="Times New Roman", fontsize=18)
ax1.set_ylabel('Latitude', fontfamily="Times New Roman", fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=16)

# Reduce the number of x-axis ticks to avoid cramping
ax1.locator_params(axis='x', nbins=4)

# Custom legend with black markers for Departure and Destination
custom_legend = [
    plt.Line2D([0], [0], color=colors[0], lw=2, label='Case 1'),
    plt.Line2D([0], [0], color=colors[1], lw=2, label='Case 3'),
    plt.Line2D([0], [0], color='black', marker='^', linestyle='None', markersize=8, label='Key Locations'),
    plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Departure'),
    plt.Line2D([0], [0], color='black', marker='s', linestyle='None', markersize=8, label='Destination')
]
ax1.legend(handles=custom_legend, loc='upper right', prop={"family": "Times New Roman", "size": 12})

# Plot velocity profile in the second subplot with specified colors and start/end markers
for i, df in enumerate(data):
    ax2.plot(df['Voyage Time'], df['Speed'], color=colors[i], label=case_names[i], linewidth=3)
    # Start and End markers in the same color as the line
    ax2.plot(df['Voyage Time'].iloc[0], df['Speed'].iloc[0], 'o', color=colors[i], markersize=8)
    ax2.plot(df['Voyage Time'].iloc[-1], df['Speed'].iloc[-1], 's', color=colors[i], markersize=8)

# Configure velocity plot, starting voyage time from 0
ax2.set_xlim(left=0)
ax2.set_ylim(8, 18)  # Set y-axis limits from 8 to 18
ax2.set_xlabel('Voyage Time (minutes)', fontfamily="Times New Roman", fontsize=18)
ax2.set_ylabel('Ship speed (knots)', fontfamily="Times New Roman", fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.legend(case_names, loc='upper right', prop={"family": "Times New Roman", "size": 12})

# Set all fonts to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

# Save the figure as a PNG file with specified size
output_path = '/workspaces/MUTE_v1/imgs/voyage_plots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()