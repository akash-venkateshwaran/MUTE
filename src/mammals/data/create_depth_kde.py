import os
import csv
import pickle
import numpy as np
import shutil
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

def load_histogram_data(csv_path):
    depths = []
    counts = []
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            depths.append(float(row[0]))
            counts.append(int(row[1]))
    return np.array(depths), np.array(counts)

def create_kde(depths, sample_counts, bandwidth=1.0):
    samples = []
    for depth, count in zip(depths, sample_counts):
        samples.extend([depth] * count)
    
    samples = np.array(samples)[:, np.newaxis]
    
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples)
    
    return kde

def get_files(directory, extension=".csv"):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]


def sample_and_plot_distribution(kde, depths, sample_counts, n_samples, output_path):
    # Sample from the KDE
    samples = kde.sample(n_samples)
    samples = samples.flatten()
    
    # Evaluate the KDE on a range of depth values for the line plot
    depth_range = np.linspace(0, max(depths) * 1.1, 1000)[:, np.newaxis]
    kde_values = np.exp(kde.score_samples(depth_range))
    
    plt.figure(figsize=(10, 6))
    
    # Histogram of sampled data
    plt.hist(samples, bins=30, alpha=0.5, color='green', label='Sampled Data', density=True)
    
    # Create a second y-axis for the KDE
    ax2 = plt.gca().twinx()
    
    # Line plot for the KDE
    ax2.plot(depth_range, kde_values, color='blue', linewidth=2, label='KDE Estimate')
    
    plt.xlabel('Depth (m)')
    plt.ylabel('Probability Density')
    ax2.set_ylabel('KDE Estimate')
    
    plt.title(f'Distribution of {n_samples} Sampled Data Points from KDE')
    
    # Add legends for both y-axes
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.xlim(0, max(depths) * 1.1)  # Ensure the x-axis covers the complete range
    plt.savefig(output_path)
    plt.show()

def main(directory):

    file_paths = get_files(os.path.join(directory, "data", "depth_data"))
    species_names = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]

    # Clear all files in the kde_models folder
    kde_models_directory = os.path.join(directory, "kde_models","depth_data")
    if os.path.exists(kde_models_directory):
        shutil.rmtree(kde_models_directory)
    os.makedirs(kde_models_directory)

    for i, file_path in enumerate(file_paths):
        depths, sample_counts = load_histogram_data(file_path)
        
        kde = create_kde(depths, sample_counts, bandwidth=2.0)
        sample_and_plot_distribution(kde, depths, sample_counts, 500, 
                                      os.path.join(base_directory, "plot", f'depth_distrib_species{i}.png'))
        
        # Export each model to a specific file
        filename = f'kde_model_species_{i}.pickle'  # Unique file name for each species
        with open(os.path.join(kde_models_directory, filename), 'wb') as file:
            pickle.dump(kde, file)
            print(f"KDE Model for {species_names[i]} exported as {filename}")


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


if __name__ == "__main__":
    main(base_directory)
