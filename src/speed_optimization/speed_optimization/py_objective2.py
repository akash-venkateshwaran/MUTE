from custom_interfaces.msg import State, HelperPosition, HeadingAngle, Path
from speed_optimization.py_util import calculate_path_length
import numpy as np
from pyproj import Geod
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.integrate import simps
from audiogram_func.audiog import audiogram_function
from models.jomopans_echo.model import jomopans_model_matrix
import matplotlib.pyplot as plt

# Define the geodesic object for WGS84
GEODESIC = Geod(ellps="WGS84")


# NOTE This is for multiple freq:
# from utils_rbf import RBFInterpolatorCustom


from noise_propagation_model.models.RBF.utils2 import RBFInterpolatorCustom

class SpeedOptimizationObjective():
        def __init__(self, waypoints: list[HelperPosition], mammal_states: list[State], shoreline_polygons: list[Polygon], 
                     freq: np.ndarray, ship_type: str, ship_length: float, n_var: int, min_v: float, max_v: float, TL_model_path: str,
                     sample_step : int = 1):
                self.waypoints = waypoints
                self.mammal_states = mammal_states
                self.path_dist_vec = self.compute_path_dist_vec(waypoints=waypoints)  # This numpy array has cumulative distances of consecutive waypoints
                self.shoreline_polygons = shoreline_polygons
                self.verbose_logging = False
                self.freq = freq
                self.ship_type = ship_type
                self.ship_length = ship_length
                self.n_var = n_var
                self.min_v = min_v
                self.max_v = max_v
                self.TL_model = RBFInterpolatorCustom.import_model(TL_model_path)
                self.th = audiogram_function(freq) # SRKW
                self.sample_step = max(1, sample_step)   # k = 1  ⇒  original behaviour


        def cost_function_noise(self, x: np.array, timestep: float):
                """
                x shape: (n_pop, n_var), timestep of x
                Noise (TL & SL) is evaluated only every `sample_step` time indices
                while trajectories remain full-resolution.
                """
                # ── 1.  Full-resolution Δt vector and trajectories ──────────────────
                time_step_vec = np.ones_like(x) * timestep
                ship_pos      = self.compute_ship_traj(time_step_vec, x)                # (pop, n_var, 2)
                mammal_poss   = self.compute_mammal_traj(time_step_vec, self.mammal_states, timestep)  # (n_mammals, n_var, 2)

                # ── 2.  Pick only every k-th time-index for acoustics ───────────────
                k   = self.sample_step
                idx = np.arange(0, x.shape[1], k)                                       # indices to keep

                ship_sub   = ship_pos[:, idx]                                           # (pop, n_idx, 2)
                mammal_sub = mammal_poss[:, idx]                                        # (n_mammals, n_idx, 2)
                speed_sub  = x[:, idx]                                                  # (pop, n_idx)


                # ── 3.  Source level for sampled steps ──────────────────────────────
                SL, f_decidecade = jomopans_model_matrix(
                        self.freq, self.ship_type, speed_sub, self.ship_length)             # (pop, n_idx, n_freq)

                # ── 4.  Transmission loss for sampled ship–mammal pairs ─────────────
                pop, n_idx, n_mammals = ship_sub.shape[0], ship_sub.shape[1], mammal_sub.shape[0]
                n_freq = SL.shape[2]

                ship_flat   = np.broadcast_to(ship_sub[:, :, None, :],
                                                (pop, n_idx, n_mammals, 2)).reshape(-1, 2)
                
                mammal_sub_swapped = np.transpose(mammal_sub, (1, 0, 2))   # (n_idx, n_mammals, 2)
                mammal_flat = np.broadcast_to(mammal_sub_swapped[None, ...],
                                                (pop, n_idx, n_mammals, 2)).reshape(-1, 2)

                TL_flat = self.compute_tranmission_loss_single_freq(mammal_flat, ship_flat)
                TL      = TL_flat.reshape(pop, n_idx, n_mammals, n_freq).transpose(0,1,3,2)  # (pop,n_idx,freq,mammals)

                # ── 5.  Noise level, peaks, SEL (Δt scaled by k) ────────────────────
                NL  = np.clip(SL[..., None] - TL, 0, None)                              # (pop,n_idx,freq,mammals)
                NLp = NL.max(axis=2)                                                    # (pop,n_idx,mammals)

                Delta_t_eff = timestep * k                                         # each sample covers k steps
                SEL = self.compute_exposure_levels(NLp, Delta_t_eff)                    # (pop,mammals)

                return SEL.mean(axis=1)   # (pop,)

        # NOTE this is the old cost_func that doesnot have the truncation property. It computes the noise at every step of x
        # def cost_function_noise(self, x):
        #         # Note: x is in knots and time is just repeated timesteps in hrs, not the time vector
        #         time_step_vec = np.ones(x.shape) * self.timestep

        #         ship_pos = self.compute_ship_traj(time_vec=time_step_vec, speed_profile=x)  # Shape: (num_gen, num_var, 2)
        #         mammal_poss = self.compute_mammal_traj(time_vec=time_step_vec, mammal_states=self.mammal_states)  # Shape: (num_mammals, num_var, 2)

        #         L_s_JE_decidecade, f_decidecade = jomopans_model_matrix(self.freq, self.ship_type, x, self.ship_length)  # Shape: (num_gen, num_var, num_freq)

        #         num_gen = ship_pos.shape[0]
        #         num_var = ship_pos.shape[1]
        #         num_mammals = mammal_poss.shape[0]
        #         num_freq = L_s_JE_decidecade.shape[2]

        #         # Expand ship_pos to shape (num_gen, num_var, num_mammals, 2)
        #         ship_pos_expanded = ship_pos[:, :, np.newaxis, :]  # Shape: (num_gen, num_var, 1, 2)
        #         ship_pos_broadcasted = np.broadcast_to(ship_pos_expanded, (num_gen, num_var, num_mammals, 2))

        #         # Expand mammal_poss to shape (num_gen, num_var, num_mammals, 2)
        #         # Transpose mammal_poss to align time steps
        #         mammal_poss_transposed = np.transpose(mammal_poss, (1, 0, 2))  # Shape: (num_var, num_mammals, 2)
        #         mammal_poss_expanded = mammal_poss_transposed[np.newaxis, :, :, :]  # Shape: (1, num_var, num_mammals, 2)
        #         mammal_pos_broadcasted = np.broadcast_to(mammal_poss_expanded, (num_gen, num_var, num_mammals, 2))

        #         # Reshape to (N, 2), where N = num_gen * num_var * num_mammals
        #         ship_loc_flat = ship_pos_broadcasted.reshape(-1, 2)
        #         mammal_loc_flat = mammal_pos_broadcasted.reshape(-1, 2)

        #         TL_flat = self.compute_tranmission_loss_single_freq(mammal_loc=mammal_loc_flat, ship_loc=ship_loc_flat)  # Output: (N, num_freq) 

        #         # Reshape TL_flat to (num_gen, num_var, num_mammals, num_freq)
        #         TL_matrix = TL_flat.reshape(num_gen, num_var, num_mammals, num_freq)
        #         # Transpose to get shape (num_gen, num_var, num_freq, num_mammals)
        #         TL_matrix = np.transpose(TL_matrix, (0, 1, 3, 2))

        

        #         # Expand SL_matrix to match dimensions
        #         SL_matrix = L_s_JE_decidecade[:, :, :, np.newaxis]  # Shape: (num_gen, num_var, num_freq, 1)
        #         SL_matrix = np.broadcast_to(SL_matrix, (num_gen, num_var, num_freq, num_mammals))

        #         # Compute NL_matrix
        #         NL_matrix = SL_matrix - TL_matrix  # Shape: (num_gen, num_var, num_freq, num_mammals)
        #         NL_matrix[NL_matrix < 0] = 0


        #         NL_exceeding_th = NL_matrix
        #         # NL_exceeding_th = NL_matrix - self.th[np.newaxis, np.newaxis, :, np.newaxis] # TODO Removed the thresold for testing
        #         # NL_exceeding_th[NL_exceeding_th < 0] = 0 # TODO the test_obj has zero as obj cause NL never exceeds audiogram

        #         # Integral:
        #         # area = simps(NL_exceeding_th, x=f_decidecade, axis=2)  # Shape: (num_gen, num_var, num_mammals) TODO Removed the integral

        #         # Peak SPL:
        #         NL_peaks = np.max(NL_exceeding_th, axis=2)

        #         # Sum over mammals TODO changed sum to mean
        #         # area_summed = np.mean(NL_peaks, axis=2)  # Shape: (num_gen, num_var)
        #         # cost = np.mean(area_summed, axis=1)  # Shape: (num_gen,)

        #         # SEL:
        #         SEL = self.compute_exposure_levels(NL_peaks) # Shape (num_gen, num_mammals)
        #         cost = np.mean(SEL, axis=1) # Shape (num_gen, )

        #         return cost
        
        
        
        # def compute_exposure_levels(self,NL_matrix):
        #         """ NL_matrix has shape (num_gen, num_var, num_mammals) and it exports SEL in dB ref uPa^2 h """
        #         Delta_t = self.timestep # NOTE The timestep is in hr hence exposure level is in dB ref uPa^2 h 

        #         E_i = Delta_t * np.power(10, NL_matrix / 10)
        #         E_total = np.sum(E_i, axis=1)  # Shape: (num_gen, num_mammals)
        #         SEL_total = 10 * np.log10(E_total)

        #         return SEL_total  # Shape: (num_gen, num_mammals)

        def compute_exposure_levels(self, NL_matrix, Delta_t):
                """
                NL_matrix shape: (pop, n_idx, n_mammals)
                Delta_t: time represented by each NL value (hrs)
                """

                E_i      = Delta_t * np.power(10, NL_matrix / 10)
                E_total  = E_i.sum(axis=1)                          # integrate over time
                SEL_tot  = 10 * np.log10(E_total)
                return SEL_tot                                      # (pop, n_mammals)

        def compute_simple_transmission_loss(self, mammal_loc, ship_loc):
                """
                Compute the simple transmission loss (TL) using geodesic distance between ship and mammal locations.

                Args:
                        mammal_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [mammal_lat, mammal_lon].
                        ship_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [ship_lat, ship_lon].
                
                Returns:
                        numpy.ndarray: The transmission loss output, shape (N, num_freq).
                """
                assert mammal_loc.shape[1] == 2, "mammal_loc must have shape (N, 2) with columns [latitude, longitude]"
                assert ship_loc.shape == mammal_loc.shape, "ship_loc must have the same shape as mammal_loc"

                N = mammal_loc.shape[0]

                distances = np.zeros(N)
                for k in range(N):
                        _, _, distance = GEODESIC.inv(
                        ship_loc[k, 1], ship_loc[k, 0], mammal_loc[k, 1], mammal_loc[k, 0]
                        )
                        distances[k] = distance

                distances = np.where(distances == 0, 1e-10, distances)
                transmission_loss = 20 * np.log10(distances)
                model_output = np.tile(transmission_loss.reshape(-1, 1), (1, len(self.freq)))

                return model_output  # Shape: (N, num_freq)

        def compute_tranmission_loss_single_freq(self, mammal_loc, ship_loc):
                """
                Compute the model output for transmission loss (TL) given mammal and ship locations.

                Args:
                        mammal_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [mammal_lat, mammal_lon].
                        ship_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [ship_lat, ship_lon].
                
                Returns:
                        numpy.ndarray: The transmission loss output from the model, shape (N, num_freq)
                """
                assert mammal_loc.shape[1] == 2, "mammal_loc must have shape (N, 2) with columns [latitude, longitude]"
                assert ship_loc.shape == mammal_loc.shape, "ship_loc must have the same shape as mammal_loc"

                N = mammal_loc.shape[0]
                model_input = np.hstack((ship_loc, mammal_loc))  # Shape: (N, 4)

                assert model_input.shape == (N, 4), "model_input must have shape (N, 4)"
                model_output = self.TL_model(model_input)  # Output: (N, num_freq)
                assert model_output.shape[0] == N, "model_output must have the first dimension equal to N"

                # Repeating the single freq to match the shape
                model_output_repeated = np.repeat(model_output, len(self.freq), axis=1)


                return model_output_repeated  # Shape: (N, num_freq)
        

        # NOTE This func was used for multiple freq however was computationally expensive ofr multiple mammals
        # def compute_tranmission_loss(self, mammal_loc, ship_loc):
        #         """
        #         Compute the model output for transmission loss (TL) given mammal and ship locations.

        #         Args:
        #                 mammal_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [mammal_lat, mammal_lon].
        #                 ship_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [ship_lat, ship_lon].
                
        #         Returns:
        #                 numpy.ndarray: The transmission loss output from the model, shape (N, num_freq)
        #         """
        #         assert mammal_loc.shape[1] == 2, "mammal_loc must have shape (N, 2) with columns [latitude, longitude]"
        #         assert ship_loc.shape == mammal_loc.shape, "ship_loc must have the same shape as mammal_loc"

        #         N = mammal_loc.shape[0]
        #         model_input = np.hstack((ship_loc, mammal_loc))  # Shape: (N, 4)

        #         assert model_input.shape == (N, 4), "model_input must have shape (N, 4)"
        #         model_output = self.TL_model(model_input)  # Output: (N, num_freq)

        #         assert model_output.shape[0] == N, "model_output must have the first dimension equal to N"

        #         # Adding the last column to match frequency dimensions (optional based on your data) to match the freq for 10k Hz
        #         last_column = model_output[:, -1].reshape(-1, 1)
        #         model_output = np.hstack((model_output, last_column))

        #         return model_output  # Shape: (N, num_freq)



        def compute_path_dist_vec(self, waypoints: list[HelperPosition]) -> np.ndarray:
                dist_vec = []
                
                # Calculate distance between consecutive waypoints
                for i in range(len(waypoints) - 1):
                        waypoint1 = waypoints[i]
                        waypoint2 = waypoints[i + 1]
                        _, _, distance = GEODESIC.inv(waypoint1.longitude, waypoint1.latitude, waypoint2.longitude, waypoint2.latitude)
                        dist_vec.append(distance)
                
                # Convert meters to Nautical Miles (NM)
                dist_vec = np.array(dist_vec) * 0.000539957
                
                # Calculate cumulative distance and prepend zero
                cumulative_dist = np.cumsum(dist_vec)
                cumulative_dist_with_zero = np.insert(cumulative_dist, 0, 0)
                
                return cumulative_dist_with_zero

        def check_state(self, longitude, latitude):
                point = Point(longitude, latitude)
                for geom in self.shoreline_polygons['geometry']:
                        if isinstance(geom, MultiPolygon):
                                for polygon in geom.geoms:
                                        if polygon.contains(point):
                                                if self.verbose_logging:
                                                        self.get_logger().info(f"State {point} is inside polygon, reversing direction")
                                                return True

                        elif isinstance(geom, Polygon):
                                if geom.contains(point):
                                        if self.verbose_logging:
                                                self.get_logger().info(f"State {point} is inside polygon, reversing direction")
                                        return True
                        return False
                        

        
        def compute_ship_traj(self, time_vec, speed_profile):
                # Calculate the ship distance vector

                ship_dist_vec = speed_profile * time_vec  # shape: (num_gen, num_var)
                ship_dist_vec = np.cumsum(ship_dist_vec, axis=1)
                
                latitude_vec = np.zeros(ship_dist_vec.shape)
                longitude_vec = np.zeros(ship_dist_vec.shape)

                # Iterate over each row (each ship or generator)
                for i, row in enumerate(ship_dist_vec):
                # Iterate over each element in the row (distance covered at various times)
                        for j, dist in enumerate(row):
                                idx = np.searchsorted(self.path_dist_vec, dist, side='right')

                                if idx == 0:
                                # Use the first waypoint
                                        lat1, long1 = self.waypoints[0].latitude, self.waypoints[0].longitude
                                        lat2, long2 = lat1, long1  # Duplicate values
                                elif idx >= len(self.path_dist_vec):
                                # Use the last waypoint
                                        lat1, long1 = self.waypoints[-1].latitude, self.waypoints[-1].longitude
                                        lat2, long2 = lat1, long1  # Duplicate values
                                else:
                                # Normal case: interpolate between the nearest waypoints
                                        lat1, long1 = self.waypoints[idx-1].latitude, self.waypoints[idx-1].longitude
                                        lat2, long2 = self.waypoints[idx].latitude, self.waypoints[idx].longitude
                                        x0 = self.path_dist_vec[idx - 1] if idx > 0 else 0
                                        x1 = self.path_dist_vec[idx]
                                        w1 = (x1 - dist) / (x1 - x0)
                                        w2 = 1 - w1

                                # Calculate interpolated latitude and longitude
                                latitude_vec[i, j] = w1 * lat1 + w2 * lat2
                                longitude_vec[i, j] = w1 * long1 + w2 * long2

                # Combine into a single 3D array
                ship_pos_vec = np.dstack((latitude_vec, longitude_vec))
                return ship_pos_vec
    
    
        def compute_mammal_traj(self, time_vec, mammal_states: list[State], timestep: float):
                """
                Computes the trajectory for multiple mammals.

                Args:
                - time_vec: Time vector for simulation.
                - mammal_states: List of initial states for each mammal.
                - timestep (hrs)

                Returns:
                - mammal_pos_vec: 3D tensor of shape (num_mammals, num_var, 2) containing latitude and longitude.
                """
                num_mammals = len(mammal_states)
                num_var = np.shape(time_vec[0, :])[0]

                mammal_pos_vec = np.zeros((num_mammals, num_var, 2))

                for j, mammal_state in enumerate(mammal_states):
                        mammal_dist_vec = mammal_state.speed * np.ones(num_var) * timestep * 60
                        mammal_dist_vec = np.reshape(mammal_dist_vec, (num_var, 1))  # Shape: (num_var, 1)

                        latitude_vec = np.zeros(mammal_dist_vec.shape[0])
                        longitude_vec = np.zeros(mammal_dist_vec.shape[0])
                        latitude_vec = np.insert(latitude_vec, 0, mammal_state.position.latitude)
                        longitude_vec = np.insert(longitude_vec, 0, mammal_state.position.longitude)
                        angle = mammal_state.angle.heading

                        for i, dist in enumerate(mammal_dist_vec):
                                dest_lon, dest_lat, _ = GEODESIC.fwd(
                                        longitude_vec[i], latitude_vec[i], angle, dist)
                                # print("Not changed", mammal_state.angle.heading)
                                if self.check_state(longitude = dest_lon, latitude = dest_lat): # Checking if the mammal is on one of the islands
                                        angle = (angle + 180) % 360
                                        dest_lon, dest_lat, _ = GEODESIC.fwd(
                                        longitude_vec[i], latitude_vec[i], angle, dist)

                                latitude_vec[i+1] = dest_lat
                                longitude_vec[i+1] = dest_lon


                        mammal_pos_vec[j, :, 0] = latitude_vec[1:]
                        mammal_pos_vec[j, :, 1] = longitude_vec[1:]

                return mammal_pos_vec
        
        