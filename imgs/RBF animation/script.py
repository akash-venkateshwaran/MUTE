#!/usr/bin/env python3
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Add custom module paths if necessary
sys.path.append('/workspaces/MELO_workspace/src/noise_propagation_model/interpolator/RBF')
sys.path.append('/workspaces/MELO_workspace/src/environment/environment')

from utils2 import RBFInterpolatorCustom
from py_polyshp import Polyshp  # This returns a dict with a 'geometry' key containing polygons
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep  # For faster spatial queries

plt.rcParams.update({'font.size': 14})  # Adjust font size as needed

def overlay_polygons(ax, polygon_list):
    """Overlay shoreline polygons on the given axis."""
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

def get_reciever_lat_lons(src_lat, src_lon, box_geom, radius_km=100, num_points=1000, polygon_list=None):
    """
    Generate receiver latitudes and longitudes within a given radius from the source,
    ensuring the points lie inside the provided bounding box and outside the polygons.
    """
    valid_lats = []
    valid_lons = []
    batch_size = int(num_points * 1.5)
    
    # If polygons are provided, union them and prepare for fast queries
    if polygon_list is not None:
        union_poly = unary_union(polygon_list['geometry'])
        prep_union = prep(union_poly)
    else:
        prep_union = None

    # Prepare the bounding box for fast containment tests
    prep_box = prep(box_geom)
    
    km_per_deg_lat = 111.0
    src_lat_rad = np.deg2rad(src_lat)
    km_per_deg_lon = 111.0 * np.cos(src_lat_rad)
    
    while len(valid_lats) < num_points:
        candidate_count = max(batch_size, num_points - len(valid_lats))
        rand_vals = np.random.rand(candidate_count)
        r = radius_km * np.sqrt(rand_vals)
        theta = 2 * np.pi * np.random.rand(candidate_count)
        
        delta_lat = r / km_per_deg_lat * np.sin(theta)
        delta_lon = r / km_per_deg_lon * np.cos(theta)
        
        candidate_lat = src_lat + delta_lat
        candidate_lon = src_lon + delta_lon

        # Filter candidates that are inside the bounding box using vectorized check
        # Create Points only for candidates inside the box. We do this by
        # checking the box's bounds directly.
        in_box = (candidate_lat >= box_geom.bounds[1]) & (candidate_lat <= box_geom.bounds[3]) & \
                 (candidate_lon >= box_geom.bounds[0]) & (candidate_lon <= box_geom.bounds[2])
        candidate_lat = candidate_lat[in_box]
        candidate_lon = candidate_lon[in_box]

        # Further filter out candidates that fall inside the union of polygons.
        if prep_union is not None:
            # Use a list comprehension since we need to call contains on each point
            valid_idx = [i for i, (lat, lon) in enumerate(zip(candidate_lat, candidate_lon))
                         if not prep_union.contains(Point(lon, lat))]
            candidate_lat = candidate_lat[valid_idx]
            candidate_lon = candidate_lon[valid_idx]
        
        num_needed = num_points - len(valid_lats)
        valid_lats.extend(candidate_lat[:num_needed])
        valid_lons.extend(candidate_lon[:num_needed])
    
    return np.array(valid_lats), np.array(valid_lons)

# def plotting_tl(tl, lat, lon, src_lat, src_lon, polygon_list, output_file):
#     """
#     Create a scatter plot of predicted transmission loss with shoreline polygons overlaid.
#     Uses the fixed lat/lon limits (from the polygon_list) for plotting.
#     The plot is saved as a PNG file.
#     """
#     fig, ax = plt.subplots(figsize=(10, 8))
    
#     # Increase axis spine and tick thickness for better visibility
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)
#     ax.tick_params(width=2)
    
#     # Scatter plot for predicted transmission loss
#     sc = ax.scatter(lon, lat, c=tl, cmap='jet_r', s=2)
#     ax.set_xlabel('Longitude')
#     ax.set_ylabel('Latitude')
#     ax.set_title(f'Predicted TL at src: ({src_lat:.5f}, {src_lon:.5f})')
    
#     # Overlay the pre-computed polygon list
#     overlay_polygons(ax, polygon_list)
    
#     # Add a colorbar for reference
#     cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02)
#     cbar.set_label('Transmission Loss (dB)')
    
#     plt.tight_layout()
#     plt.savefig(output_file, format='png', dpi=300)
#     plt.close(fig)


def plotting_tl(tl, lat, lon, src_lat, src_lon, polygon_list, output_file):
    """
    Create a contour plot of predicted transmission loss with shoreline polygons overlaid.
    Adds a scatter point for the source location.
    The plot is saved as a PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Increase axis spine and tick thickness for better visibility
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)
    
    # Create a triangulation for contour plotting
    triang = tri.Triangulation(lon, lat)
    
    # Contour plot for predicted transmission loss
    contour = ax.tricontourf(triang, tl, levels=25, cmap='jet_r', alpha=0.8)
    
    # Scatter plot for the source location
    ax.scatter(src_lon, src_lat, color='black', s=50, label='Source Location', zorder=3)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Predicted TL at src: ({src_lat:.5f}, {src_lon:.5f})')
    
    # Overlay the pre-computed polygon list
    overlay_polygons(ax, polygon_list)
    
    # Add a legend for the source marker
    ax.legend(loc='upper right')
    
    # Add a colorbar for reference
    cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Transmission Loss (dB)')
    
    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=300)
    plt.close(fig)

def process_src_locations(src_lats, src_lons, model, polygon_list, output_dir, box_geom):
    """
    For each source location (src_lat, src_lon) in the provided arrays,
    generate receiver locations that are NOT within any of the polygons and lie within the bounding box,
    predict transmission loss using the model, and export a scatter plot with shoreline overlays.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (src_lat, src_lon) in enumerate(zip(src_lats, src_lons)):
        # Generate receiver locations uniformly within a 100 km radius,
        # ensuring they are within the bounding box and outside any of the polygons.
        rec_lat, rec_lon = get_reciever_lat_lons(
            src_lat, src_lon, box_geom, radius_km=100, num_points=4000, polygon_list=polygon_list)
        
        # Prepare model input: [receiver_lat, receiver_lon, src_lat, src_lon]
        src_lat_col = np.full(rec_lat.shape, src_lat)
        src_lon_col = np.full(rec_lon.shape, src_lon)
        model_input = np.column_stack((rec_lat, rec_lon, src_lat_col, src_lon_col))
        
        # Predict transmission loss using the RBF interpolator model
        predicted_tl = model(model_input).ravel()
        predicted_tl = np.clip(predicted_tl, 0, 200)
        
        # Define output file name for the plot
        output_file = os.path.join(output_dir, f"predicted_tl_{i}.png")
        plotting_tl(predicted_tl, rec_lat, rec_lon, src_lat, src_lon, polygon_list, output_file)
        print(f"Saved plot for source ({src_lat}, {src_lon}) to {output_file}")

def main():
    # CSV file path (adjust as needed)
    csv_path = '/workspaces/MELO_workspace/imgs/RBF animation/case_study_C1.csv'
    
    # Import CSV file.
    # Assuming the CSV file has at least 4 columns: we use column index 2 for longitudes and index 3 for latitudes.
    df = pd.read_csv(csv_path, header=None)
    src_lons = df.iloc[:, 2].to_numpy()  # third column: longitudes
    src_lats = df.iloc[:, 3].to_numpy()  # fourth column: latitudes
    src_lats = src_lats[::2]
    src_lons = src_lons[::2]
    
    # Print the number of source locations
    num_src = len(src_lats)
    print(f"Number of source locations: {num_src}")
    
    # Define fixed lat/lon limits based on the overall source locations.
    # These values are used to create a bounding box.
    lat_min, lat_max = 47.5, 50.25
    lon_min, lon_max = -126.7, -122.02
    # Create a Shapely bounding box: note that box expects (minx, miny, maxx, maxy)
    bounding_box = box(lon_min, lat_min, lon_max, lat_max)
    
    # Set paths for model and shoreline data.
    model_path = '/workspaces/MELO_workspace/src/noise_propagation_model/interpolator/RBF/model_single_freq'
    shp_data_path = '/workspaces/MELO_workspace/src/environment/shoreline_data'
    output_dir = '/workspaces/MELO_workspace/imgs/RBF animation/imgs'  # Folder to save the PNG files

    # Load the RBF interpolator model and the polygon class.
    rbf_interpolator = RBFInterpolatorCustom.import_model(model_path)
    shp_class = Polyshp(shp_data_path)
    
    # Define a fixed polygon_list once using the bounding box limits.
    polygon_list = shp_class.get_polygons(lat_min, lat_max, lon_min, lon_max)
    
    # Process each source location using the same polygon list and bounding box.
    process_src_locations(src_lats, src_lons, rbf_interpolator, polygon_list, output_dir, bounding_box)

if __name__ == '__main__':
    main()
