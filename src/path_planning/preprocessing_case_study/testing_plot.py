from environment.py_polyshp import Polyshp
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def extract_lat_lon_from_shd(shd_folder):
    latitudes = []
    longitudes = []
    
    for file_name in os.listdir(shd_folder):
        if file_name.endswith(".shd"):
            file_path = os.path.join(shd_folder, file_name)
            
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            latitudes.append(lat)
                            longitudes.append(lon)
                        except ValueError:
                            continue
    
    return latitudes, longitudes


bathy_max_lat = 49.55530783251801 
bathy_min_lat = 47.619114710530006 
bathy_max_lon = -122
bathy_min_lon = -127

shiplane_data_loc = '/workspaces/MELO_workspace/src/environment/ship_lanes_data'
shiplane_obj = Polyshp(shiplane_data_loc)
poly_list = shiplane_obj.get_polygons(lat_min=bathy_min_lat, lat_max=bathy_max_lat, lon_min=bathy_min_lon, lon_max=bathy_max_lon)

csv_file_path = '/workspaces/MELO_workspace/src/path_planning/preprocessing_case_study/processed_ship_states.csv'
df = pd.read_csv(csv_file_path)

latitude = df['Latitude']
longitude = df['Longitude']

# Print the first and last latitude and longitude
first_lat_lon = (df['Latitude'].iloc[0], df['Longitude'].iloc[0])
last_lat_lon = (df['Latitude'].iloc[-1], df['Longitude'].iloc[-1])

print(f"First Latitude, Longitude: {first_lat_lon}")
print(f"Last Latitude, Longitude: {last_lat_lon}")

# Print the total time of the voyage (last value of the 'Time (mins)' column)
total_time = df['Time (mins)'].iloc[-1]
print(f"Total Time of Voyage: {total_time} minutes")

shd_folder_path = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/source_locs'
latitudes, longitudes = extract_lat_lon_from_shd(shd_folder_path)

print(len(latitudes))


# Plotting the polygons and ship states
if poly_list is not None:
    gdf = poly_list
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor='black', cmap='tab20', linewidth=1, alpha=0.5)
    ax.scatter(longitude, latitude, color='red', marker='o', s=10, label='Ship States', zorder=5)
    ax.set_title('Polygons with Ship States Overlay')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.show()
    plt.savefig('/workspaces/MELO_workspace/src/path_planning/preprocessing_case_study/polygons_with_ship_states.png', bbox_inches='tight', dpi=300)
