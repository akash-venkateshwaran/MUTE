
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Paths to the CSV files based on the directory structure
csv_files = [
    '/workspaces/MELO_workspace/src/path_planning/case_studies/Case_1/case_study.csv',
    '/workspaces/MELO_workspace/src/path_planning/case_studies/Case_2/case_study.csv',
    '/workspaces/MELO_workspace/src/path_planning/case_studies/Case_3/case_study.csv'
]
case_names = ['Case 1', 'Case 2', 'Case 3']
ship_plane_path = '/workspaces/MELO_workspace/src/environment/ship_lanes_data/new_ship_lanes_polys.shp'


# Colors for each case (RGB format)
colors = [(0.1, 0.2, 0.8), (0.2, 0.7, 0.3), (0.8, 0.1, 0.1)]  # Blue, Green, Red

# Load and store the data
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

# Set the figure size and create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

# Load shoreline/islands data with black border
shoreline = gpd.read_file('/workspaces/MELO_workspace/src/environment/shoreline_data')

shoreline.plot(ax=ax1, color='lightgrey', edgecolor='black')  # Add black border

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

# Configure map limits and labels
ax1.set_xlim(-125.72143254685919, -121.78283374169551)
ax1.set_ylim(47.95629176783697, 49.5104276496541)
ax1.set_title('Voyage Paths with Key Locations', fontfamily="Times New Roman")
ax1.set_xlabel('Longitude', fontfamily="Times New Roman")
ax1.set_ylabel('Latitude', fontfamily="Times New Roman")

# Custom legend with black markers for Departure and Destination
custom_legend = [
    plt.Line2D([0], [0], color=colors[0], lw=2, label='Case 1'),
    plt.Line2D([0], [0], color=colors[1], lw=2, label='Case 2'),
    plt.Line2D([0], [0], color=colors[2], lw=2, label='Case 3'),
    plt.Line2D([0], [0], color='black', marker='^', linestyle='None', markersize=8, label='Key Locations'),
    plt.Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Departure'),
    plt.Line2D([0], [0], color='black', marker='s', linestyle='None', markersize=8, label='Destination')
]
ax1.legend(handles=custom_legend, loc='upper right', prop={"family": "Times New Roman"})

# Plot velocity profile in the second subplot with specified colors and start/end markers
for i, df in enumerate(data):
    ax2.plot(df['Voyage Time'], df['Speed'], color=colors[i], label=case_names[i])
    # Start and End markers in the same color as the line
    ax2.plot(df['Voyage Time'].iloc[0], df['Speed'].iloc[0], 'o', color=colors[i], markersize=8)
    ax2.plot(df['Voyage Time'].iloc[-1], df['Speed'].iloc[-1], 's', color=colors[i], markersize=8)

# Configure velocity plot, starting voyage time from 0
ax2.set_xlim(left=0)
ax2.set_title('Velocity Profile over Voyage Time', fontfamily="Times New Roman")
ax2.set_xlabel('Voyage Time (minutes)', fontfamily="Times New Roman")
ax2.set_ylabel('Vessel Speed (kt)', fontfamily="Times New Roman")
ax2.legend(case_names, loc='upper right', prop={"family": "Times New Roman"})

# Set all fonts to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})
plt.tight_layout()

# Save the figure as a PNG file with specified size
output_path = '/workspaces/MELO_workspace/imgs/voyage_plots.png'
# fig.set_size_inches(10, 25)  # Fixed size before saving
plt.savefig(output_path, dpi=300)
plt.show()