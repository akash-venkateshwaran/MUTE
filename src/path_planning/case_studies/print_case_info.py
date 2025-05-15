import os
import pandas as pd
from pyproj import Geod

GEODESIC = Geod(ellps="WGS84")
def process_case_study(data):
    
    # Extract longitude, latitude, and velocity
    longitudes = data.iloc[:, 2]
    latitudes = data.iloc[:, 3]
    velocities_knots = data.iloc[:, 1]
    
    # Convert velocities from knots to meters per second
    velocities_mps = velocities_knots * 0.514444
    
    
    # Calculate total distance in nautical miles
    total_distance_nm = 0
    for i in range(1, len(longitudes)):
        lon1, lat1 = longitudes[i - 1], latitudes[i - 1]
        lon2, lat2 = longitudes[i], latitudes[i]
        
        # Compute the geodesic distance in meters
        _, _, distance_m = GEODESIC.inv(lon1, lat1, lon2, lat2)
        # Convert meters to nautical miles
        distance_nm = distance_m / 1852
        total_distance_nm += distance_nm
    
    print(f"Total Distance: {total_distance_nm:.2f} nautical miles")
    
    # Compute ETA (Estimated Time of Arrival)
    if velocities_mps.sum() > 0:
        average_velocity_mps = velocities_mps.mean()
        total_distance_m = total_distance_nm * 1852  # Convert NM to meters
        total_time_seconds = total_distance_m / average_velocity_mps
        total_time_minutes = total_time_seconds / 60
        
        print(f"ETA: {int(total_time_minutes)} minutes")
    else:
        print("ETA: Cannot compute, velocity is zero.")



def compute_lag_time_case_study( df):
        """
        This function is to calc lag time in case studies btw dist based on vel and path:
        - Extracting velocity, longitude, and latitude columns.
        - Calculating the distance between consecutive latitude/longitude points.
        -  Summing all distances to get the total distance in meters.
        - Summing all velocities to get the total velocity in m/s.
        - Computing the total time based on total distance and average velocity.
        - Returning the total time in minutes as an integer.
        """
        longitudes = df.iloc[:, 2]
        latitudes = df.iloc[:, 3]
        velocities_knots = df.iloc[:, 1]
        
        velocities_mps = velocities_knots * 0.514444
        
        distances_m = []
        for i in range(1, len(longitudes)):
            lon1, lat1 = longitudes[i - 1], latitudes[i - 1]
            lon2, lat2 = longitudes[i], latitudes[i]
            _, _, distance_m = GEODESIC.inv(lon1, lat1, lon2, lat2)
            distances_m.append(distance_m)
        
        distances_m.insert(0, 0)
        df['distance_between_points_m'] = distances_m

        
        total_distance_m = sum(distances_m)
        total_velocity_mps = sum(velocities_mps)

       
        if total_velocity_mps > 0:
            total_time_seconds = total_distance_m / (total_velocity_mps / len(velocities_mps))
        else:
            total_time_seconds = 0
        
        total_time_minutes = total_time_seconds / 60.0
        
        return int(total_time_minutes)

csv_file = os.path.join('/workspaces/MELO_workspace/src/path_planning/case_studies/Case_3', 'case_study.csv')
    
if not os.path.isfile(csv_file):
        raise FileNotFoundError("case_study.csv not found in the specified folder.")
    

data = pd.read_csv(csv_file, header=None)

process_case_study(data)
compute_lag_time_case_study(data)

