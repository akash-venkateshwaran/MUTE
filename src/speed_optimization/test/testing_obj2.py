from custom_interfaces.msg import State, HelperPosition, HeadingAngle
from py_objective2 import SpeedOptimizationObjective
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from speed_optimization.py_util import calculate_path_length
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.variable import Real
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_column_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [float(row[0]) for row in reader]
    
def plot_ship_speed_profile(speed_profile, timestep):
    # Create a time vector based on the timestep and the number of speeds
    time_vec = np.arange(0, speed_profile.size)

    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, speed_profile.flatten(), linestyle='-', marker='o', color='blue', label='Ship Speed Profile')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (knots)')
    plt.title('Ship Speed Profile Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()


import sys
sys.path.append('/workspaces/MELO_workspace/src/environment/environment')
from py_polyshp import Polyshp  # Ensure this is the correct import path


ship_lats = load_column_from_csv("/workspaces/MELO_workspace/src/speed_optimization/temp/ship_lats.csv")
ship_lons = load_column_from_csv("/workspaces/MELO_workspace/src/speed_optimization/temp/ship_lons.csv")
ship_speeds = load_column_from_csv("/workspaces/MELO_workspace/src/speed_optimization/temp/ship_speed2.csv")

# Define test parameters
timestep = 0.2

waypoints = [
    HelperPosition(latitude=lat, longitude=lon, depth=0.0)
    for lat, lon in zip(ship_lats, ship_lons)
]

# waypoints = [
#     HelperPosition(latitude=48.92610473619462, longitude=-123.19756092133534, depth=0.0),
#     HelperPosition(latitude=48.82121769660953, longitude=-122.97184176946791, depth=0.0),
#     HelperPosition(latitude=48.72110592117163, longitude=-123.10812503097277, depth=0.0),
#     HelperPosition(latitude=48.70080976158849, longitude=-123.28179154823765, depth=0.0)
# ]

# Define the mammal state
mammal_state = State(
    position=HelperPosition(latitude=48.5729370117188, longitude=-123.32502746582, depth=0.0),
    angle=HeadingAngle(heading=0.0),
    speed=0.0  # Zero velocity
)




# Frequency array
freq = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 
                 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])

# Shoreline data
shp_data_path = '/workspaces/MELO_workspace/src/environment/shoreline_data'
shoreline = Polyshp(shp_data_loc=shp_data_path)
polygon_list = shoreline.get_polygons(lat_min=47.0, lat_max=49.0, lon_min=-124.0, lon_max=-122.0)  # Define appropriate bounds

# Ship type and length
ship_type = 'Bulker'
ship_length = 692.25

# Instantiate the optimization objective

# Test speed profile

# test_speed_profile = np.array([[5.0, 5.0, 5.0, 5.0, 5.0 , 5.0 , 10.0]])  # Example speed profile (knots)
# test_speed_profile =np.full((1, self.n_var), vel_constant)
test_speed_profile = np.array([ship_speeds])



n_var = test_speed_profile.shape[1]
max_v = 15.9 # knots
min_v = 0.1 # knots
obj_func = SpeedOptimizationObjective(
    timestep=timestep,
    waypoints=waypoints,
    mammal_states=[mammal_state],
    shoreline_polygons=polygon_list,
    freq=freq,
    ship_type=ship_type,
    ship_length=ship_length,
    n_var = n_var,
    min_v= min_v,
    max_v=max_v, TL_model_path='/workspaces/MELO_workspace/src/noise_propagation_model/interpolator/RBF/model'
)


# Run cost function for noise and print results
noise_cost = obj_func.cost_function_noise(test_speed_profile)
print("Output of cost_function_noise:", noise_cost)

# Print the total path distance vector
print("Total distance to cover", calculate_path_length(waypoints=waypoints))
print("Total distance covered: ", np.sum(test_speed_profile * timestep))


# Function to plot waypoints, mammal, and ship trajectory
def plot_waypoints_mammal_ship(waypoints, mammal, ship_traj):
    waypoint_lats = [wp.latitude for wp in waypoints]
    waypoint_lons = [wp.longitude for wp in waypoints]

    plt.figure(figsize=(10, 8))
    plt.scatter(waypoint_lons, waypoint_lats, color='blue', label='Waypoints', s=50)
    plt.scatter(mammal.longitude, mammal.latitude, color='red', s=100, label='Mammal')

    # Plot ship trajectory using dotted lines
    ship_lats = ship_traj[:, :, 0].flatten()
    ship_lons = ship_traj[:, :, 1].flatten()
    plt.plot(ship_lons, ship_lats, linestyle='-', marker='o', color='green', label='Ship Trajectory')

    # Annotate waypoints and mammal
    # for i, (lat, lon) in enumerate(zip(waypoint_lats, waypoint_lons)):
    #     plt.text(lon, lat, f'WP{i+1}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # plt.text(mammal.longitude, mammal.latitude, 'Mammal', fontsize=12, color='red', verticalalignment='bottom')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Waypoints, Mammal Location, and Ship Trajectory')
    plt.grid(True)
    plt.legend()
    plt.show()

# Compute ship trajectory
time_vec = np.ones(test_speed_profile.shape) * timestep
ship_trajectory = obj_func.compute_ship_traj(time_vec, test_speed_profile)

# Call the function to plot
plot_waypoints_mammal_ship(waypoints, mammal_state.position, ship_trajectory)
plot_ship_speed_profile(test_speed_profile, timestep)
