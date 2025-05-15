import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import importlib.util


utils_path = '/workspaces/MELO_workspace/src/noise_propagation_model/interpolator/RBF/utils.py'
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
sys.modules["utils"] = utils
spec.loader.exec_module(utils)
RBFInterpolatorCustom = utils.RBFInterpolatorCustom


model_path = '/workspaces/MELO_workspace/src/noise_propagation_model/interpolator/RBF/model'
print("done")
rbf_loaded = RBFInterpolatorCustom.import_model(model_path)
print("done2")
coordinate_dir = '/workspaces/MELO_workspace/src/noise_propagation_model/models/bellhop_new/Coordinate_data'

npy_file = os.path.join(coordinate_dir, '0_origin_lat48.530345926_lon-125.164387079.npy')
x_matrix = np.load(npy_file)
print("import done")

y_matrix = []

i = 0
for row in x_matrix:
    print(i)
    i+=1
    if row.shape[0] == 5:
        x_feature = row.reshape(1, -1)
        y_feature = rbf_loaded(x_feature)
        y_matrix.append(y_feature)


y_matrix = np.vstack(y_matrix)

print(x_matrix.shape)
print(y_matrix.shape)

# num_cols = y_matrix.shape[1]
# fig, axes = plt.subplots(1, num_cols, figsize=(15, 5), subplot_kw={'projection': '3d'})

# for j in range(num_cols):
#     ax = axes[j] if num_cols > 1 else axes
#     r_lat = x_matrix[:, 2]
#     r_lon = x_matrix[:, 3]
#     r_z = x_matrix[:, 4]
    
#     sc = ax.scatter(r_lat, r_lon, r_z, c=y_matrix[:, j], cmap=cm.viridis)
    
#     ax.set_xlabel('Latitude')
#     ax.set_ylabel('Longitude')
#     ax.set_zlabel('Depth (Z)')
#     ax.set_title(f'Plot {j+1}')
    
#     fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)

# plt.tight_layout()
# plt.show()
