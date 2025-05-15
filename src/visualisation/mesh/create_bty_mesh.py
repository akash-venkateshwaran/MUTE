import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import surf2stl
import os
from pyproj import Proj
from scipy.ndimage import gaussian_filter

# Load the data
gebco_file = '/workspaces/MELO_workspace/src/environment/bathy_data/gebco_2023_n49.9814_s46.8063_w-125.2002_e-121.5601.nc'
output_dir = '/workspaces/MELO_workspace/src/visualisation/mesh'

nc = Dataset(gebco_file)
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
lat_coord = np.repeat(lat, len(lon))
lon_coord = np.tile(lon, len(lat))

ele = nc.variables['elevation'][:].astype(float)
ele = ele / 1000

mean_lat = np.mean(lat)
mean_lon = np.mean(lon)

# Define a projection with pyproj
proj = Proj(proj='aeqd', lat_0=mean_lat, lon_0=mean_lon, units='km')

# Project lat and lon to x and y in kilometers
x, y = proj(lon_coord, lat_coord)

# Create meshgrid
x_mesh = np.reshape(x, (len(lat), len(lon)))
y_mesh = np.reshape(y, (len(lat), len(lon)))

# Smoothen the elevation data
ele_smoothed = gaussian_filter(ele, sigma=1)

# Plot surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_mesh, y_mesh, ele_smoothed, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, label='Elevation')
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Elevation (km)')
ax.set_title('Surface Plot of Elevation Data')

output_file = os.path.join(output_dir, 'ele_surface.png')
plt.savefig(output_file)
plt.close()

# Write the smoothened surface to an STL format file
output_file_stl = os.path.join(output_dir, 'bathymetry.stl')
surf2stl.write(output_file_stl, x_mesh, y_mesh, ele_smoothed)

print(f'STL file saved to {output_file_stl}')
