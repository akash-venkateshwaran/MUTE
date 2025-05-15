from pyproj import Geod
from custom_interfaces.msg import State, HelperPosition, HeadingAngle, Path

import numpy as np
from matplotlib import pyplot as plt

import rclpy

# Define the geodesic object for WGS84
GEODESIC = Geod(ellps="WGS84")

def calculate_path_length(waypoints: list[HelperPosition]) -> float:
        # Calculate the total geodesic distance between each pair of waypoints
        total_distance = 0
        for i in range(len(waypoints) - 1):
            waypoint1 = waypoints[i]
            waypoint2 = waypoints[i + 1]
            _, _, distance = GEODESIC.inv(waypoint1.longitude, waypoint1.latitude, waypoint2.longitude, waypoint2.latitude)
            total_distance += distance
        total_distance *= 0.000539957  # meters to NM
        return total_distance




# TEST:
# class ShipTrajectoryCalculator:
#     def __init__(self, path_dist_vec):
#         self.path_dist_vec = np.array(path_dist_vec)
#         self.waypoints = [HelperPosition(latitude=10.0, longitude=10.0, depth=0.0),
#              HelperPosition(latitude=20.0, longitude=20.0, depth=0.0),
#              HelperPosition(latitude=30.0, longitude=30.0, depth=0.0),
#              HelperPosition(latitude=40.0, longitude=40.0, depth=0.0),
#              HelperPosition(latitude=50.0, longitude=50.0, depth=0.0),
#              HelperPosition(latitude=60.0, longitude=60.0, depth=0.0),
#              HelperPosition(latitude=70.0, longitude=70.0, depth=0.0),
#              HelperPosition(latitude=80.0, longitude=80.0, depth=0.0)]

#     def compute_ship_traj(self, time_vec, speed_profile):
#         # Calculate the ship distance vector
#         ship_dist_vec = speed_profile * time_vec  # shape: (num_gen, num_var)
#         ship_dist_vec = np.cumsum(ship_dist_vec, axis=1)
        
#         latitude_vec = np.zeros(ship_dist_vec.shape)
#         longitude_vec = np.zeros(ship_dist_vec.shape)

#         # Iterate over each row (each ship or generator)
#         for i, row in enumerate(ship_dist_vec):
#             # Iterate over each element in the row (distance covered at various times)
#             for j, dist in enumerate(row):
#                 idx = np.searchsorted(self.path_dist_vec, dist, side='right')

#                 if idx == 0:
#                     # Use the first waypoint
#                     lat1 = long1 = lat2 = long2 = self.waypoints[0].latitude, self.waypoints[0].longitude
#                 elif idx >= len(self.path_dist_vec):
#                     # Use the last waypoint
#                     lat1 = long1 = lat2 = long2 = self.waypoints[-1].latitude, self.waypoints[-1].longitude
#                 else:
#                     # Normal case: interpolate between the nearest waypoints
#                     lat1, long1 = self.waypoints[idx-1].latitude, self.waypoints[idx-1].longitude
#                     lat2, long2 = self.waypoints[idx].latitude, self.waypoints[idx].longitude
#                     x0 = self.path_dist_vec[idx - 1] if idx > 0 else 0
#                     x1 = self.path_dist_vec[idx]
#                     w1 = (x1 - dist) / (x1 - x0)
#                     w2 = 1 - w1

#                 # Calculate interpolated latitude and longitude
#                 latitude_vec[i, j] = w1 * lat1 + w2 * lat2
#                 longitude_vec[i, j] = w1 * long1 + w2 * long2

#         # Combine into a single 3D array
#         ship_pos_vec = np.dstack((latitude_vec, longitude_vec))
        
#         return ship_pos_vec

# data1 = [
#     [1, 1,1,1],
#     [1,1,1,1]
# ]

# data2 = [
#     [0.1, 2,0.1,4],
#     [1, 2,3,1.5]
# ]

# # Example usage:
# path_distances = [0, 1.5, 3, 4.5, 6,7.5,9,10.5]
# time_vector = np.array(data1)
# speed_profile = np.array(data2)

# calculator = ShipTrajectoryCalculator(path_distances)
# temp = calculator.compute_ship_traj(time_vector, speed_profile)
# waypoints = calculator.waypoints
# # Plot latitude and longitude points
# fig, ax = plt.subplots()
# ship_latitudes = temp[0, :, 0]
# ship_longitudes = temp[0, :, 1]
# ship_alpha_values = np.linspace(0.2, 1, len(ship_latitudes))
#         # Plot ship positions with varying alpha values

# for lat, lon, alpha in zip(ship_latitudes, ship_longitudes, ship_alpha_values):
#             ax.plot(lon, lat, 'bo', alpha=alpha)

# for i in range(len(waypoints)):
#             ax.plot(waypoints[i].longitude,waypoints[i].latitude,'ko',fillstyle='none', markersize=5)

# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.legend()

# # Save the figure
# filename = "temp.png"
# plt.savefig(f"imgs/{filename}")
# plt.close()


# ship_pos = np.zeros((1, 5, 2))
# ship_pos[0, :, 0] = np.array([2, 3, 4, 5, 6])
# ship_pos[0, :, 1] = np.array([11, 21, 31, 41, 51])
# mammal_pos = np.zeros((5, 1, 2))
# mammal_pos[:,:,0] = np.array([[1],[2],[3],[4],[5]])
# mammal_pos[:,:,1] = np.array([[10],[20],[30],[40],[50]])
# cost_vec = np.empty(ship_pos.shape[0])
# for i, row in enumerate(ship_pos):
#             temp1 = np.reshape(row,(ship_pos.shape[1],2))
#             temp2 = np.reshape(mammal_pos,(ship_pos.shape[1],2))
#             temp = (temp1 - temp2)**2
#             print(temp)
#             cost_vec[i] = np.sum(np.sqrt(temp[:,0] + temp[:,1]))

# print(cost_vec)