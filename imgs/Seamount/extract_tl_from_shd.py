import os
import numpy as np
import h5py
from utils_computecanada import extract_tl
import matplotlib.pyplot as plt

def process_single_shd_file(shd_dir, tl_dir, filename):
    """
    Process a single .shd file by its filename and export Transmission Loss (TL) and Frequency data.
    
    Parameters:
    - shd_dir: Directory containing the subfolders with .shd files
    - tl_dir: Directory to save the processed TL and Frequency data
    - filename: Name of the .shd file to process
    """
    TL_export = np.array([])
    freq_export = []

    print(f"Processing and exporting TL and Frequency data for file: {filename}")

    # Construct the full path for the file
    file_path = os.path.join(shd_dir, filename)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    # Extract TL, frequency, and positional data
    TL_each_freq, x, y, z, freq, Pos = extract_tl(file_path)
    x_flat = x.flatten()
    y_flat = y.flatten()
    TL_export = np.empty((0, 4))  # 4 columns for x, y, z, and temp

    for i in range(0, TL_each_freq.shape[0]):
        temp = TL_each_freq[i, :, :].flatten().reshape(-1, 1)
        z_col = np.full((temp.shape[0], 1), z[i])
        result = np.hstack((x_flat.reshape(-1, 1), y_flat.reshape(-1, 1), z_col, temp))
        
        # Stack result row-wise into TL_export
        TL_export = np.vstack((TL_export, result))


    print(TL_export.shape)

    # TL_flattened = TL_each_freq.flatten().reshape(-1, 1)  # Reshape to column vector

    # # Accumulate TL data across frequencies
    # TL_export = TL_flattened
    # freq_export.append(freq)

    # Ensure the output directory exists
    os.makedirs(tl_dir, exist_ok=True)

    # Define export file name
    export_filename = os.path.join(tl_dir, f'{filename.replace(".shd", "")}.h5')

    # Save data to HDF5 format
    with h5py.File(export_filename, 'w') as hf:
        hf.create_dataset('Seamount_XYZ_TL', data=TL_export, compression='gzip', compression_opts=5)

    # # Output shapes for verification
    # print(f"Shape of TL for one frequency: {TL_each_freq.shape}")
    # print(f"Shape of TL exported: {TL_export.shape}")
    # print(f"Exported data to: {export_filename}\n\n")

# Example usage
shd_dir = '/workspaces/MELO_workspace/imgs/Seamount'
tl_dir = '/workspaces/MELO_workspace/imgs/Seamount'
filename = '3D_Seamount.shd'  # Replace with the specific .shd filename to process
process_single_shd_file(shd_dir, tl_dir, filename)
