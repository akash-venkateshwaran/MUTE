import h5py
import os
import numpy as np
import re
import matplotlib.pyplot as plt

# Add any custom module paths only once when this file is imported/executed
import sys
sys.path.append('/workspaces/MUTE_v1/src/noise_propagation_model/interpolator/RBF')
sys.path.append('/workspaces/MUTE_v1/src/environment/environment')

from noise_propagation_model.models.RBF.utils2 import RBFInterpolatorCustom
from environment.py_polyshp import Polyshp

plt.rcParams.update({'font.size': 12})  # slightly smaller for compact layout


def overlay_polygons(ax, polygon_list):
    """Draw grey‑filled shoreline polygons (if any) on *ax*."""
    if polygon_list is None:
        return
    for poly in polygon_list['geometry']:
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=1)
            ax.fill(x, y, color='lightgrey', alpha=1.0)
        elif poly.geom_type == 'MultiPolygon':
            for sub_poly in poly.geoms:
                x, y = sub_poly.exterior.xy
                ax.plot(x, y, color='black', linewidth=1)
                ax.fill(x, y, color='lightgrey', alpha=1.0)


# ────────────────────────────────────────────────────────────────────────────────
#  IO helpers
# ────────────────────────────────────────────────────────────────────────────────

def read_h5_file(file_path: str, dataset_name: str):
    """Return *dataset_name* from *file_path* (HDF5)."""
    with h5py.File(file_path, 'r') as h5_file:
        return h5_file[dataset_name][:]


def extract_src_lat_lon(filename: str):
    """Parse source (lat, lon) pair from *filename* (…_latX_lonY.h5)."""
    m = re.search(r'_lat([-+]?[0-9]*\.?[0-9]+)_lon([-+]?[0-9]*\.?[0-9]+)', filename)
    if m:
        return float(m.group(1)), float(m.group(2))
    raise ValueError(f"Unable to parse lat/lon from {filename}")


# ────────────────────────────────────────────────────────────────────────────────
#  Core plotting routine (NEW – supports multiple files & 2×4 grid)
# ────────────────────────────────────────────────────────────────────────────────

def plot_interpolation_results_multi(coord_path: str,
                                     tl_path: str,
                                     filenames: list[str],
                                     interpolator: RBFInterpolatorCustom,
                                     shp_class: Polyshp,
                                     save_path: str = '/workspaces/MUTE_v1/imgs/interpolation_comparisons2.png'):
    """Create a 2×4 grid comparing TL truth vs RBF prediction for *filenames*.

    The grid layout per row (filename):
        [True TL | Predicted TL | MAE | MAPE]

    A horizontal color‑bar is placed **once** at the top of each column (based
    on the first row). Color scales are shared between rows to aid comparison.
    """
    if len(filenames) != 2:
        raise ValueError("Expected exactly two filenames for a 2×4 grid.")

    # Pre‑scan to determine global color ranges so both rows share a scale
    tl_values, pred_values, mae_values, mape_values = [], [], [], []
    for fname in filenames:
        x_feat = read_h5_file(os.path.join(coord_path, fname), 'x_feature')
        tl = read_h5_file(os.path.join(tl_path, fname), 'TransmissionLoss')[:, 0]
        src_lat, src_lon = extract_src_lat_lon(fname)
        lat, lon = x_feat[:, 3], x_feat[:, 4]
        pred = interpolator(np.column_stack((lat, lon, np.full_like(lat, src_lat), np.full_like(lon, src_lon)))).ravel()
        tl = np.clip(tl, 0, 200)
        pred = np.clip(pred, 0, 200)
        abs_err = np.abs(tl - pred)
        mape = np.where(tl != 0, 100 * abs_err / np.abs(tl), np.nan)
        tl_values.append(tl)
        pred_values.append(pred)
        mae_values.append(abs_err)
        mape_values.append(mape)

    # Define shared vmin/vmax per column for coherent colours
    vmins = [min(np.min(v) for v in col) for col in (tl_values, pred_values, mae_values, mape_values)]
    vmaxs = [max(np.max(v) for v in col) for col in (tl_values, pred_values, mae_values, mape_values)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    col_titles = ['True TL (dB)', 'Predicted TL (dB)', 'Abs. Error (dB)', 'MAPE (%)']

    for row_idx, fname in enumerate(filenames):
        # Grab pre-computed arrays
        lat = read_h5_file(os.path.join(coord_path, fname), 'x_feature')[:, 3]
        lon = read_h5_file(os.path.join(coord_path, fname), 'x_feature')[:, 4]
        tl = tl_values[row_idx]
        pred = pred_values[row_idx]
        abs_err = mae_values[row_idx]
        mape = mape_values[row_idx]


        data_cols = [tl, pred, abs_err, mape]
        polygon_list = shp_class.get_polygons(np.min(lat), np.max(lat), np.min(lon), np.max(lon))

        for col_idx, data in enumerate(data_cols):
            ax = axes[row_idx, col_idx]

            # choose colormap: first two keep 'jet', last two use 'gist_heat'
            cmap = 'jet' if col_idx < 2 else 'gist_heat_r'
            sc = ax.scatter(lon, lat, c=data, cmap=cmap, s=4,
                            vmin=vmins[col_idx], vmax=vmaxs[col_idx])

            overlay_polygons(ax, polygon_list)
            # horizontal colour-bar at top of each column, with ticks on top
            if row_idx == 0:
                cbar = fig.colorbar(
                    sc,
                    ax=[axes[0, col_idx], axes[1, col_idx]],
                    orientation='horizontal',
                    location='top',    # <— add this
                    pad=0.06,          # adjust as needed for spacing
                    aspect=20
                )
                # ticks and label on top (already correct)
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.xaxis.tick_top()

            # only first column gets a y-label; all others drop their y-ticks
            if col_idx == 0:
                ax.set_ylabel(f'Lat (row {row_idx+1})')
            else:
                ax.set_yticklabels([])
            # remove x-ticks/text for the entire top row
            if row_idx == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Lon')

            # tighten up spines and tick widths
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(width=1.5)

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────────
#  Example usage (commented – remove comments when integrating with pipeline)
# ────────────────────────────────────────────────────────────────────────────────
coord_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/bellhop/Coordinate_data/filtered/one_z_plain'
tl_path    = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/bellhop/TL_data/filtered/one_z_plain'
model_path = '/workspaces/MUTE_v1/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq'
shp_data   = '/workspaces/MUTE_v1/src/environment/shoreline_data'
filenames  = [
    '334_origin_lat48.821783203_lon-122.931561706.h5',
    '199_origin_lat49.324162897_lon-123.214719981.h5'  # <- replace with a real second file
]
rbf_interpolator = RBFInterpolatorCustom.import_model(model_path)
shp_class = Polyshp(shp_data)
plot_interpolation_results_multi(coord_path, tl_path, filenames, rbf_interpolator, shp_class)
