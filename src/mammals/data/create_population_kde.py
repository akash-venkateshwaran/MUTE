import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from shapely.geometry import Point
from environment.py_polyshp import Polyshp
import geopandas as gpd
import pickle
import shutil

from pathlib import Path


def create_mask(directory):
    
    file_paths = get_files(os.path.join(directory, "data", "population_data"))

    species_names = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
    data_list = [load_csv_data(path) for path in file_paths]

    shoreline = Polyshp(shp_data_loc=shoreline_paths)

    all_data = np.vstack(data_list)
    buffer = 1.0
    lat_min, lon_min = np.min(all_data, axis=0) - buffer
    lat_max, lon_max = np.max(all_data, axis=0) + buffer
    
    print(f"Min Lat = {lat_min} Lon = {lon_min}")
    print(f"Max Lat = {lat_max} Lon = {lon_max}")

    polygons = shoreline.get_polygons(lat_min, lat_max, lon_min, lon_max)
    if isinstance(polygons, gpd.GeoDataFrame):
        # grab only the Shapely geometries
        geometry_list = polygons.geometry.tolist()
    else:
        # if it's a single Polygon, wrap it; if it's already a list of Polygons, leave it
        if not isinstance(polygons, (list, tuple)):
            geometry_list = [polygons]
        else:
            geometry_list = polygons
    print("Necessary polygon extracted")

    if polygons is not None:
        num_grid_points = 300
        latitude_points = np.linspace(lat_min, lat_max, num_grid_points)
        longitude_points = np.linspace(lon_min, lon_max, num_grid_points)
        grid_lat, grid_lon = np.meshgrid(latitude_points, longitude_points)
        point_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(np.ravel(grid_lon), np.ravel(grid_lat))], crs='EPSG:4326')
        
        polygon_gdf = gpd.GeoDataFrame({'geometry': geometry_list}, crs='EPSG:4326')
        polygon_gdf.sindex
        
        result = gpd.sjoin(point_gdf, polygon_gdf, how="inner", op='intersects')
        mask = np.zeros((num_grid_points, num_grid_points), dtype=int)
        mask[np.unravel_index(result.index, (num_grid_points, num_grid_points))] = 1

        plt.figure(figsize=(10, 10))
        plt.contourf(grid_lon, grid_lat, mask, levels=[-0.5, 0.5, 1.5], cmap='Greys')
        plt.colorbar(label='Land/Water Mask')
        plt.title('Land/Water Mask Based on Shoreline Data')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(os.path.join(directory, "plot", "land_water_mask2.png"))
        plt.close()

        return mask, grid_lat, grid_lon
    else:
        print("No polygons found inside the box")
        return None


def load_csv_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return np.array(data)

def get_files(directory, extension=".csv"):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]

def plot_species_scatter(data_list, species_names, directory):
    plt.figure()
    colors = ['r', 'b', 'g', 'k']
    for i, data in enumerate(data_list):
        data = np.radians(data)
        color = colors[i % len(colors)]
        plt.scatter(data[:, 1], data[:, 0], color=color, label=species_names[i], s=5)
    plt.legend()
    plt.xlabel('Longitude ')
    plt.ylabel('Latitude ')
    plt.title('Species Distribution Scatter Plot')
    filename = "Scatter.png"
    plt.savefig(os.path.join(directory,"plot", filename))
    plt.close()

def main(directory):
    file_paths = get_files(os.path.join(directory, "data", "population_data"))
    species_names = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]
    data_list = [load_csv_data(path) for path in file_paths]

    mask, grid_lat, grid_lon = create_mask(directory)
    mask_bool = (mask == 0.0).ravel() # mask[i][j] == 1 is land / == 0 is water
    grid_lat = np.radians(grid_lat)
    grid_lon = np.radians(grid_lon)

    plt.figure(figsize=(15, 7))
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
    
    # Clear all files in the kde_models folder
    kde_models_directory = os.path.join(directory, "kde_models","population_data")
    if os.path.exists(kde_models_directory):
        shutil.rmtree(kde_models_directory)
    os.makedirs(kde_models_directory)

    for i, data in enumerate(data_list):
        data = np.radians(data)

        xy = np.vstack([grid_lat.ravel(), grid_lon.ravel()]).T
        xy = xy[mask_bool]
        # Bandwdth should be 0.001 ideally (Lower means higher resoln)
        kde = KernelDensity(bandwidth=0.002, metric="haversine", kernel="gaussian", algorithm="ball_tree")
        kde.fit(data)
        Z = np.full(mask_bool.shape[0], 0.0, dtype="int")
        Z[mask_bool] = np.exp(kde.score_samples(xy))
        Z = Z.reshape(grid_lon.shape)

        ax = plt.subplot(1, len(file_paths), i + 1)
        contourf_plot = ax.contourf(grid_lon, grid_lat, Z, cmap=plt.cm.Reds)
        plt.colorbar(contourf_plot, ax=ax, label='Density')
        plt.title(species_names[i])
        plt.contour(grid_lon, grid_lat, mask, levels=[0.0], colors="black", linestyles="solid")
        ax.set_xlabel('Longitude (radians)')
        ax.set_ylabel('Latitude (radians)')

        # Export each model to a specific file
        filename = f'kde_model_species_{i}.pickle'  # Unique file name for each species
        with open(os.path.join(kde_models_directory, filename), 'wb') as file:
            pickle.dump(kde, file)
            print(f"KDE Model for {species_names[i]} exported as {filename}")

    filename = "KDE.png"
    plt.savefig(os.path.join(directory, "plot", filename))
    plt.close()

    plot_species_scatter(data_list, species_names, directory)

def find_src_dir(start_path: Path) -> Path:
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate a 'src' directory.")

src_dir = find_src_dir(Path(__file__))
base_directory = str(src_dir / 'mammals')
shoreline_paths = str(os.path.join(src_dir, "environment", "shoreline_data"))

main(base_directory)

