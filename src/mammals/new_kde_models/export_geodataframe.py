import os
import numpy as np
import geopandas as gpd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, default_converter
from shapely.geometry import Point
from pyproj import CRS

# ------------------------------------------------------------
# Base directory and file definitions (globals)
base_directory = '/workspaces/MELO_workspace/src/mammals'
rds_path = os.path.join(base_directory, 'new_kde_models', 'posterior_mean_map.rds')
export_path = os.path.join(base_directory,'new_kde_models', 'posterior_mean_map.csv')
# ------------------------------------------------------------

# Metadata for exported GeoDataFrame columns:
#
# - May, June, July, August, Sept, Oct:
#     Posterior mean log-intensity values (log scale) for each month
# - May_intensity, June_intensity, ..., Oct_intensity:
#     Exponentiated intensity values (linear scale) that are normalized by the sum
# - geometry:
#     Original shapely Point geometry in file's CRS
# - lat, lon:
#     Coordinates in WGS84 (EPSG:4326), decimal degrees
# ------------------------------------------------------------

def load_rds_to_gdf():
    """
    Load an RDS file and return a GeoDataFrame with months renamed.
    """
    posterior_r = ro.r['readRDS'](rds_path)
    with (default_converter + pandas2ri.converter).context():
        df = ro.conversion.rpy2py(posterior_r.slots['data'])

    coords = np.column_stack((
        np.array(posterior_r.slots['coords'])[:, 0],
        np.array(posterior_r.slots['coords'])[:, 1]
    ))

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in coords],
        crs=CRS(str(posterior_r.slots['proj4string']))
    )

    month_names = ["May", "June", "July", "August", "Sept", "Oct"]
    gdf.rename(columns=dict(zip(gdf.columns[:6], month_names)), inplace=True)
    return gdf


def add_intensity_columns(gdf):
    """
    Compute linear intensities for each month and add as columns,
    normalized so each intensity column sums to 1.
    """
    months = ["May", "June", "July", "August", "Sept", "Oct"]
    for m in months:
        if m not in gdf.columns:
            raise ValueError(f"Month '{m}' not found in data")
        # compute raw intensity
        raw_intensity = np.exp(gdf[m].values)
        # normalize so sum(raw_intensity) == 1
        norm_intensity = raw_intensity / raw_intensity.sum()
        gdf[f"{m}_intensity"] = norm_intensity
    return gdf


def export_gdf_csv(gdf):
    """
    Transform to WGS84, add lat/lon, and export as CSV.
    """
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf_wgs['lat'] = gdf_wgs.geometry.y
    gdf_wgs['lon'] = gdf_wgs.geometry.x
    cols = ['lat', 'lon'] + [c for c in gdf_wgs.columns if c not in ['geometry', 'lat', 'lon']]
    gdf_wgs[cols].to_csv(export_path, index=False)
    print(f"Export complete: {export_path}")


if __name__ == '__main__':
    gdf = load_rds_to_gdf()
    gdf = add_intensity_columns(gdf)
    export_gdf_csv(gdf)
