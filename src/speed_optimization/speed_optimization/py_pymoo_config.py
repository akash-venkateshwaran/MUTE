from pymoo.core.problem import Problem

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
import numpy as np
from pymoo.util.display.output import Output
from custom_interfaces.msg import State, HelperPosition
from pyproj import Geod
import matplotlib.pyplot as plt
import os

from shapely.geometry import Polygon
import pandas as pd
from pymoo.core.repair import Repair
from scipy.interpolate import splrep, splev, BSpline, interp1d

from speed_optimization.py_util import calculate_path_length
from speed_optimization.py_objective2 import SpeedOptimizationObjective
# Define the geodesic object for WGS84
GEODESIC = Geod(ellps="WGS84")

class SpeedOptimizationProblem(Problem):
    def __init__(self, n_var, n_obj, xl, xu, timestep, waypoints: list[HelperPosition], mammal_states:list[State], shoreline_polygons: list[Polygon], freq: np.ndarray,
                 ship_type: str, ship_length: float, extra_time: float, TL_model_path: str,
                 acc_min=-10.5,                #  knots/h  (deceleration allowed)
                 acc_max=+10.5,                #  knots/h  (acceleration allowed)
                 obj_computation_step : int = 1):
        
        # n_ieq_constr = 1 + 2 * (n_var - 1) # This is the num of contratint for acc 1 is for total_dist 2*(n_var -1) are the acc constrinat min and max # TODO uncomment to add acc constraint
        n_ieq_constr = 1
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_ieq_constr, xl=xl, xu=xu) # TODO This is single evaluation, remove this if parallel works


        self.waypoints = waypoints
        self.path_length = calculate_path_length(waypoints)
        self.timestep = timestep
        self.timestep_min = 1/60
        self.ETA = (n_var * timestep) + extra_time *timestep # hrs
        print("Available Travel time (h)", self.ETA, '-->', self.ETA*60, "mins")

        # self.time_vector =  np.arange(1, n_var + 1) * timestep # shape: (n_steps,) [timestep, 2*timestep, ..., n_steps*timestep]
        # self.time_vector_min =  np.arange(1, self.ETA*60) * self.timestep_min # time_vector with timestep of 1 min increment in hourly (0.01666)

        self.acc_min = acc_min
        self.acc_max = acc_max
        self.speed_min = np.full((1, n_var), xl) # SHape: (1, n_var)
        self.speed_max = np.full((1, n_var), xu)
        self.extra_time = extra_time  # Extra time due to divisiblity(hrs)
        self.obj_computation_step = obj_computation_step

        
        self.speed_objective = SpeedOptimizationObjective(waypoints=waypoints,mammal_states=mammal_states,
                                                           shoreline_polygons=shoreline_polygons, freq = freq, 
                                                           ship_type = ship_type, ship_length = ship_length, n_var = n_var, min_v = xl, max_v = xu, TL_model_path=TL_model_path,
                                                           sample_step = obj_computation_step)
        
    def compute_total_distance(self, x: np.ndarray, timestep: float, extra_time: float) -> np.ndarray:
        """ x is in knots, timestep is in hrs and total travel dist is in NM
        """
        # sum distance over all time steps
        travelled = np.sum(x * timestep, axis=1) # Shape: (gen_size,)
        # add extra distance from last speed
        extra = extra_time * timestep * x[:, -1] # Shape: (gen_size,)
        return travelled + extra
    
    

    def _evaluate(self, x, out, *args, **kwargs): # The evalutation method without smootheneing
        # Note that x has shape of (gen_size,n_decision_vars)
        # out["F"] must have shape (gen_size,)

        total_travelled_distance = self.compute_total_distance(x, self.timestep, self.extra_time) # Shape: (gen_size,)


        out["F"] = self.speed_objective.cost_function_noise(x=x, timestep = self.timestep) # Shape: (gen_size,)

        g_dist = np.abs(total_travelled_distance - self.path_length) - 0.1      # shape (pop,)
        # acc = np.diff(x, axis=1) / self.timestep_min                 # shape (pop, n_var-1)
        # # TODO I have replaced the division by timestep with the actual timestep that is 1/60 (1 min) based on the logic used in ship node compute_current_speed()

        # g_acc_upper = acc - self.acc_max                          #   acc ≤ acc_max
        # g_acc_lower = self.acc_min - acc                          #   acc ≥ acc_min

        # # flatten the two acc matrices and concatenate with g_dist
        # out["G"] = np.hstack((g_dist[:, None],
        #                       g_acc_upper,
        #                       g_acc_lower)) #TODO Uncomment this to add acc contraint
        
        out["G"] = np.hstack(g_dist[:, None])
        

    # def _evaluate(self, x, out, *args, **kwargs):
    #     # Note that x has shape of (gen_size,n_decision_vars)
    #     # out["F"] must have shape (gen_size,)

    #     x = self.speed_profile_smoothener(x) # Upscaling and smoothening Shape: (gen_size, ETA*60)
    #     total_travelled_distance = self.compute_total_distance(x, self.timestep_min, 0.0) # Shape: (gen_size,) #NOTE The timestep is min because the speed profile x is upscaled to every min


    #     out["F"] = self.speed_objective.cost_function_noise(x=x, timestep = self.timestep_min) # Shape: (gen_size,) #NOTE The timestep is min because the speed profile x is upscaled to every min

    #     g_dist = np.abs(total_travelled_distance - self.path_length) - 0.1      # shape (pop,)
    #     acc = np.diff(x, axis=1) / self.timestep_min #   # shape (pop, n_var-1) NOTE I have replaced the division by timestep with the actual timestep that is 1/60 (1 min) based on the logic used in ship node compute_current_speed()

    #     g_acc_upper = acc - self.acc_max                          #   acc ≤ acc_max
    #     g_acc_lower = self.acc_min - acc                          #   acc ≥ acc_min

    #     # flatten the two acc matrices and concatenate with g_dist
    #     # out["G"] = np.hstack((g_dist[:, None],
    #     #                       g_acc_upper,
    #     #                       g_acc_lower)) TODO Uncomment this to add acc contraint

    #     out["G"] = np.hstack(g_dist[:, None])
        

    # def speed_profile_smoothener(self, x: np.ndarray, spline_s: float = 10.0) -> np.ndarray:
    #     """
    #     Vector-friendly upsampling + B-spline smoothing.

    #     Parameters
    #     ----------
    #     x         : (n_gen , n_low) coarse speeds
    #     spline_s  : smoothing factor for `splrep`

    #     Returns
    #     -------
    #     speed_hi  : (n_gen , n_high) smoothed profile
    #     """
    #     n_gen, _ = x.shape
    #     t_low  = self.time_vector # Low resol time vector
    #     t_high = self.time_vector_min # Minute resol time vector

    #     # vectorised linear interpolation (axis=1 ⇒ per-row interpolation)
    #     speed_hi = interp1d(t_low, x, kind='linear',
    #                         axis=1, bounds_error=False,
    #                         fill_value='extrapolate')(t_high)

    #     # B-spline smoothing row-wise (tight loop only on coef computation)
    #     for i in range(n_gen):
    #         coeff = splrep(t_high, speed_hi[i], s=spline_s)
    #         speed_hi[i] = splev(t_high, coeff)

    #     return speed_hi 
    
    

class SpeedOptimizationCallback(Output):
    def __init__(self, export_path,  every_n_gen=100, initial_gen = 0):
        super().__init__()
        self.every_n_gen = every_n_gen
        self.export_path = export_path
        self.initial_gen = initial_gen
        self.min_f_values = []

    def update(self, algorithm):
        super().update(algorithm)
        if algorithm.n_gen % self.every_n_gen == 0 or algorithm.n_gen == 1:
            F = algorithm.pop.get("F")

            min_f = np.min(F)
            min_f_index = np.argmin(F)

            self.min_f_values.append(min_f)
            population_data = algorithm.pop.get("X")  # Entire population as a 2D numpy array
            total_travelled_distance = np.sum(population_data[min_f_index, :] * algorithm.problem.timestep)
            path_diff = np.abs(total_travelled_distance - algorithm.problem.path_length).min()

            # Determine additional info
            pop_shape = population_data.shape

            # Print header only at the first generation
            if algorithm.n_gen == 1:
                self.print_header()

            # Print generation, minimum F, path difference, population shape, solution found
            self.print_text(algorithm.n_gen, min_f, path_diff, pop_shape)

            # Export current population to CSV
            columns = [f"Variable_{i+1}" for i in range(population_data.shape[1])]
            df_population = pd.DataFrame(population_data, columns=columns)
            output_file = f"{self.export_path}/optimal_population_{algorithm.n_gen + self.initial_gen}.csv"
            df_population.to_csv(output_file, index=False)


    def print_header(self):
        header = (
            f"{'n_gen':>10} | {'F_min':>15} | {'Path_diff':>15} | "
            f"{'Pop_size,n_var':>15} "
        )
        print("=" * len(header))
        print(header)
        print("=" * len(header))


    def print_text(self, n_gen, min_f, path_diff, pop_shape):
        text = (
            f"{n_gen:>10} | {min_f:>15.4f} | {path_diff:>15.4f} | "
            f"{str(pop_shape):>15} "
        )
        print(text)

    def plot_and_export(self, n_gen, X, min_f):

        
        # Plot X values
        plt.figure(figsize=(8, 6))
        plt.plot(X, label='Design Variables', marker='o', linestyle='-')
        plt.title(f"Generation: {n_gen}, F_min: {min_f:.4f}", fontsize=14)
        plt.xlabel("Design Variable Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True)

        # Define output directory and filename
        output_dir = "/workspaces/MUTE_v1/imgs"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"X_plot_gen_{n_gen}.png")

        # Save plot as PNG
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"Plot for generation {n_gen} exported to {filename}")



class AdaptiveMutation(Mutation):
    def __init__(self, initial_rate, final_rate, n_gen):
        super().__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.n_gen = n_gen

    def _do(self, problem, X, **kwargs):
        current_gen = kwargs['algorithm'].n_gen
        rate = self.initial_rate + (self.final_rate - self.initial_rate) * (current_gen / self.n_gen)
        mutation = PolynomialMutation(prob=rate, eta=10) 
        return mutation._do(problem, X)

class AdaptiveCrossover(Crossover):
    def __init__(self, initial_rate, final_rate, n_gen):
        super().__init__(2, 2)
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.n_gen = n_gen

    def _do(self, problem, X, **kwargs):
        current_gen = kwargs['algorithm'].n_gen
        rate = self.initial_rate + (self.final_rate - self.initial_rate) * (current_gen / self.n_gen)
        crossover = SBX(prob=rate, eta=15)
        return crossover._do(problem, X)
    


class AccelerationRepairOperator(Repair):
    """
    Repair operator that enforces acceleration constraints
    by clipping speeds according to acc_min/acc_max bounds.

    Logic:
        1. Copy X to avoid in-place edits.
        2. For up to max iterations, compute acc = diff(X, axis=1)/dt (shape (pop_size, n_var-1)).
        3. Identify violations above acc_max or below acc_min.
        4. Compute target speeds for next step and adjust only those entries.
        5. Clip all speeds to overall bounds.
        6. Break early if no violations remain.
    """

    def __init__(self, max_iter: int = 5):
        super().__init__()
        self.max_iter = max_iter

    def _do(self, problem, X, **kwargs):
        dt = problem.timestep_min
        acc_min, acc_max = problem.acc_min, problem.acc_max
        speed_min, speed_max = problem.speed_min, problem.speed_max

        repaired = X.copy()

        for _ in range(self.max_iter):
            # compute accelerations between adjacent segments
            acc = np.diff(repaired, axis=1) / dt

            # detect violations
            too_fast = acc > acc_max
            too_slow = acc < acc_min
            violation = too_fast | too_slow
            if not violation.any():
                break

            # compute adjusted next speeds
            next_upper = repaired[:, :-1] + acc_max * dt
            next_lower = repaired[:, :-1] + acc_min * dt

            # build adjustment array
            adj = np.zeros_like(acc)
            adj[too_fast] = next_upper[too_fast]
            adj[too_slow] = next_lower[too_slow]

            # apply adjustments only where needed
            repaired[:, 1:] = np.where(violation, adj, repaired[:, 1:])

            # enforce overall speed bounds
            repaired = np.clip(repaired, speed_min, speed_max)

        return repaired