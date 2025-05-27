import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from environment.py_polyshp import Polyshp
from pathlib import Path
from mammals.launch_utils import CSVMammalSampler, validate_samples
# Locate project src directory

def find_src_dir(start_path: Path) -> Path:
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate a 'src' directory.")

# Globals
src_dir = find_src_dir(Path(__file__))
base_directory = str(src_dir / 'mammals')
csv_path = os.path.join(base_directory,'new_kde_models', 'posterior_mean_map.csv')
shoreline_paths = os.path.join(str(src_dir / 'environment'), 'shoreline_data')


def plot_month_probability(df, samples, month, shoreline_paths, coast=None):
    """
    df: pandas.DataFrame with columns ['lat','lon', f'{month}_intensity']
    samples: (N,2) array of [lat, lon] samples
    month: string, e.g. "May"
    shoreline_paths: path or list of paths for Polyshp
    coast: optional GeoDataFrame of coastline; if provided, we’ll plot its boundary
    """
    # 1. Build the probability GeoDataFrame
    prob_df = df[['lat','lon', f'{month}_intensity']].rename(
        columns={f'{month}_intensity': 'prob'}
    ).copy()
    prob_gdf = gpd.GeoDataFrame(
        prob_df,
        geometry=[Point(xy) for xy in zip(prob_df.lon, prob_df.lat)],
        crs='EPSG:4326'
    )

    # 2. Load shoreline if needed
    if coast is None:
        shoreline = Polyshp(shp_data_loc=shoreline_paths)
        polys = shoreline.get_polygons(
            prob_df.lat.min(), prob_df.lat.max(),
            prob_df.lon.min(), prob_df.lon.max()
        )
        if isinstance(polys, gpd.GeoDataFrame):
            coast = polys
        else:
            coast = gpd.GeoDataFrame(
                {'geometry': polys if isinstance(polys, list) else [polys]},
                crs='EPSG:4326'
            )

    # 3. Build the samples GeoDataFrame
    samples_gdf = gpd.GeoDataFrame(
        geometry=[Point(xy) for xy in zip(samples[:,1], samples[:,0])],
        crs='EPSG:4326'
    )

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 4a. Probability map
    vmin1, vmax1 = prob_gdf['prob'].min(), prob_gdf['prob'].max()
    print(f"Sampling Probability vmin={vmin1:.3f}, vmax={vmax1:.3f}")

    prob_gdf.plot(
        column='prob', ax=axes[0],
        cmap="RdYlBu_r", vmin=vmin1, vmax=vmax1,
        legend=True,
        legend_kwds={'label': "Sampling Probability", 'shrink': 0.5}
    )
    coast.boundary.plot(
        ax=axes[0], edgecolor="black", linewidth=0.5
    )
    axes[0].set_title(f"{month} Sampling Probability")
    axes[0].axis("off")

    # 4b. Sample scatter
    samples_gdf.plot(
        ax=axes[1], markersize=5, alpha=0.6
    )
    coast.boundary.plot(
        ax=axes[1], edgecolor="black", linewidth=0.5
    )
    axes[1].set_title(f"Sampled Mammals ({month})")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    # parameters
    month = 'May'
    df = pd.read_csv(csv_path)

    sampler = CSVMammalSampler(csv_path, month)
    samples = sampler.sample(500)
    shoreline_paths = os.path.join(src_dir, "environment", "shoreline_data")
    shoreline = Polyshp(shoreline_paths)
    polygons = shoreline.get_polygons(df['lat'].min(), df['lat'].max(), df['lon'].min(), df['lon'].max())

    headings = np.random.uniform(-180, 180, size=(500, 1))
    speeds = np.random.uniform(1, 4, size=(500, 1))  # 1–4 m/s speeds

    samples = np.hstack((samples, np.ones(500).reshape(500,1), headings, speeds))
    print(samples)
    valid_samples_batch = validate_samples(
                samples, polygons, df['lat'].min(), df['lat'].max(), df['lon'].min(), df['lon'].max()
            )
    
    print(valid_samples_batch)

    # grid for contour
    plot_month_probability(
    df=df, 
    samples=samples, 
    month="May", 
    shoreline_paths=shoreline_paths)
