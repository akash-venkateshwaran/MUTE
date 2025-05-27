import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
import sys
from pathlib import Path as Path2

def find_src_dir(start_path: Path2) -> Path2:
    """
    Search upward from the start_path until a directory containing a 'src' folder is found.
    Returns the path to that 'src' folder.
    """
    current = start_path.resolve()
    while current != current.parent:  # stop at filesystem root
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not find a 'src' directory in any parent directories.")

src_dir = find_src_dir(Path2(__file__))



def audiogram_function(freq_Hz):
    """
    Computes the threshold (th) for an array of frequencies in Hz
    using the audiogram function from  Killer whale (Orcinus orca) behavioral audiograms JASA (2017).

    Parameters:
    freq_Hz (numpy array): Frequency in Hz

    Returns:
    numpy array: Threshold (th) in dB
    """
    # Parameters for the audiogram function
    T0 = 44.13
    F1 = 12.55  # in kHz
    F2 = 26.50  # in kHz
    A = 40.58
    B = 2.74

    # Convert frequency from Hz to kHz
    freq_kHz = freq_Hz / 1000

    # Calculate threshold
    th = T0 + A * np.log10(1 + (F1 / freq_kHz)) + (freq_kHz / F2) ** B

    return th

# Load data from CSV file
csv_file = str(src_dir / 'mammals' / 'data/depth_data' / 'species_1.csv')
data = pd.read_csv(csv_file)

# Extract depths and counts
depths = data['Depth (m)'].values
sample_counts = data['Counts'].values

# Generate samples based on counts for the KDE and histogram
samples = []
for depth, count in zip(depths, sample_counts):
    samples.extend([depth] * count)
samples = np.array(samples)

# Paths
kde_model_path = str(src_dir / 'mammals' / 'kde_models/depth_data' / 'kde_model_species_0.pickle')
output_path = '/workspaces/MUTE_v1/imgs/Mammal_depth_distribution_and_audiogram.png'

# Load KDE from pickle file
try:
    with open(kde_model_path, 'rb') as f:
        kde = pickle.load(f)
    print("Loaded KDE model from file.")
except FileNotFoundError:
    print("KDE model file not found.")
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

# Create frequency range from 10 to 10^5 Hz and calculate thresholds using the audiogram function
freq_range = np.logspace(1, 5, 500)  # Frequency from 10 Hz to 100,000 Hz
thresholds = audiogram_function(freq_range)  # Thresholds for the audiogram function

# Set up figure and subplots
dpi = 300  # Dots per inch
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5000 / dpi, 2500 / dpi), dpi=dpi)  # Horizontal layout

# Plot depth distribution on the first subplot
ax1.hist(bin_centers, bins=bin_edges, weights=interpolated_counts, color='grey', edgecolor='black', alpha=0.7, linewidth=3)
ax1.set_xscale('log')
ax1.set_xlim(1.0, 100)
ax1.set_xlabel("Depth (m) (log scale)")
ax1.set_ylabel("Interpolated Counts", color='grey')
ax1.tick_params(axis='y', labelcolor='grey')

# Create a secondary y-axis for the KDE in the depth distribution subplot
ax1_kde = ax1.twinx()
if kde:
    depth_range = np.linspace(1.0, 100, 1000)[:, np.newaxis]
    log_density = kde.score_samples(depth_range)
    density = np.exp(log_density)
    
    ax1_kde.plot(depth_range, density, color='black', lw=4, label='KDE')  # Set line width to 4

    ax1_kde.set_ylabel("Density", color='black')
    ax1_kde.tick_params(axis='y', labelcolor='black')

# Plot the audiogram on the second subplot
ax2.plot(freq_range, thresholds, color='blue', lw=4)  # Set line width to 4
ax2.set_xscale('log')
ax2.set_xlabel("Frequency (Hz) (log scale)")
ax2.set_ylabel("Threshold (dB)")
ax2.set_title("Audiogram Function")
ax2.grid(True)

for ax in (ax1, ax2):
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Increase border thickness by 2

# Increase tick size and tick width for both subplots
for ax in (ax1, ax2):
    ax.tick_params(axis='both', which='major', labelsize=ax.xaxis.get_label().get_size() + 2, width=2)  # Increase tick label size and thickness


# Adjust layout and save the figure as PNG
fig.tight_layout()
plt.savefig(output_path, format='png', bbox_inches='tight')
plt.close()
