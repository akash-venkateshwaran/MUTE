import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import os

base_dir = '/workspaces/MELO_workspace/src/environment'
file_path = os.path.join(base_dir, 'ssp_data', 'SSP Salish sea fig7.csv') 
df = pd.read_csv(file_path, header=None)

df.columns = ['SSP1_depth', 'SSP1_speed', 'SSP2_depth', 'SSP2_speed', 'SSP3_depth', 'SSP3_speed']

ssp1_speed = df['SSP1_speed'].dropna()
ssp1_depth = df['SSP1_depth'].dropna()
ssp2_speed = df['SSP2_speed'].dropna()
ssp2_depth = df['SSP2_depth'].dropna()
ssp3_speed = df['SSP3_speed'].dropna()
ssp3_depth = df['SSP3_depth'].dropna()

save_dir = os.path.join(base_dir, 'plot')
os.makedirs(save_dir, exist_ok=True)

print('done')

plt.figure()
plt.scatter(ssp1_speed, ssp1_depth, label='SSP1')
plt.scatter(ssp2_speed, ssp2_depth, label='SSP2')
plt.scatter(ssp3_speed, ssp3_depth, label='SSP3')
plt.xlabel('Speed (m/s)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
plt.legend()
plt.title('SSP Profiles')
plt.savefig(f'{save_dir}/ssp_profiles.png')
plt.close()

print('done')

depths_0_200 = np.linspace(0, 200, 50) # high resolution upto depth 200 m
depths_200_1000 = np.linspace(200, 1000, 5)
depths = np.concatenate((depths_0_200, depths_200_1000))

ssp1_interp = interp1d(ssp1_depth, ssp1_speed, fill_value="extrapolate")
ssp2_interp = interp1d(ssp2_depth, ssp2_speed, fill_value="extrapolate")
ssp3_interp = interp1d(ssp3_depth, ssp3_speed, fill_value="extrapolate")

ssp1_fit = ssp1_interp(depths)
ssp2_fit = ssp2_interp(depths)
ssp3_fit = ssp3_interp(depths)

avg_ssp = (ssp1_fit + ssp2_fit + ssp3_fit) / 3

ssp_matrix = np.column_stack((depths, avg_ssp))

ssp_data_dir = os.path.join(base_dir, 'ssp_data')
os.makedirs(ssp_data_dir, exist_ok=True)
np.save(os.path.join(ssp_data_dir, 'ssp.npy'), ssp_matrix)

plt.figure()
plt.plot(ssp1_fit, depths, label='SSP1 Interpolated')
plt.plot(ssp2_fit, depths, label='SSP2 Interpolated')
plt.plot(ssp3_fit, depths, label='SSP3 Interpolated')
plt.plot(avg_ssp, depths, label='Average SSP', color='black')
plt.xlabel('Speed (m/s)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
plt.legend()
plt.title('Interpolated SSP Profiles with Average SSP')
plt.savefig(f'{save_dir}/fitted_ssp_profiles_with_avg.png')
plt.close()