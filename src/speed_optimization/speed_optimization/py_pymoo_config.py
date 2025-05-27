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

from speed_optimization.py_util import calculate_path_length
from speed_optimization.py_objective2 import SpeedOptimizationObjective
# Define the geodesic object for WGS84
GEODESIC = Geod(ellps="WGS84")

class SpeedOptimizationProblem(Problem):
    def __init__(self, n_var, n_obj, xl, xu, timestep, waypoints: list[HelperPosition], mammal_states:list[State], shoreline_polygons: list[Polygon], freq: np.ndarray,
                 ship_type: str, ship_length: float, extra_time: float, TL_model_path: str,
                 acc_min=-10.5,                #  knots/h  (deceleration allowed)
                 acc_max=+10.5,                #  knots/h  (acceleration allowed)
                 ):
        
        n_ieq_constr = 1 + 2 * (n_var - 1)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_ieq_constr, xl=xl, xu=xu)
        self.waypoints = waypoints
        self.path_length = calculate_path_length(waypoints)
        self.timestep = timestep
        self.acc_min = acc_min
        self.acc_max = acc_max

        self.extra_time = extra_time  # Extra time due to divisiblity(hrs)
        self.speed_objective = SpeedOptimizationObjective(timestep=timestep,waypoints=waypoints,mammal_states=mammal_states,
                                                           shoreline_polygons=shoreline_polygons, freq = freq, 
                                                           ship_type = ship_type, ship_length = ship_length, n_var = n_var, min_v = xl, max_v = xu, TL_model_path=TL_model_path)


    def _evaluate(self, x, out, *args, **kwargs):
        # Note that x has shape of (gen_size,n_decision_vars)
        # out["F"] must have shape (gen_size,)
        travelled_distance = np.sum(x * self.timestep, axis=1)  # Shape: (gen_size,)
        extra_travelled_distance = self.extra_time * self.timestep * x[:, -1]  # Shape: (gen_size,)
        total_travelled_distance = travelled_distance + extra_travelled_distance 

        out["F"] = self.speed_objective.cost_function_noise(x=x)

        g_dist = np.abs(total_travelled_distance - self.path_length) - 0.1      # shape (pop,)
        acc = np.diff(x, axis=1) / self.timestep                  # shape (pop, n_var-1)

        g_acc_upper = acc - self.acc_max                          #   acc ≤ acc_max
        g_acc_lower = self.acc_min - acc                          #   acc ≥ acc_min

        # flatten the two acc matrices and concatenate with g_dist
        out["G"] = np.hstack((g_dist[:, None],
                              g_acc_upper,
                              g_acc_lower))
    

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
            total_travelled_distance = np.sum(population_data[min_f_index,:] * algorithm.problem.timestep)
            path_diff = np.abs(total_travelled_distance - algorithm.problem.path_length).min()

            if algorithm.n_gen == 1:  # Print header only at the first generation
                self.print_header()
            self.print_text(algorithm.n_gen, min_f, path_diff)

            
            columns = [f"Variable_{i+1}" for i in range(population_data.shape[1])]
            df_population = pd.DataFrame(population_data, columns=columns)
            output_file = f"{self.export_path}/optimal_population_{algorithm.n_gen + self.initial_gen}.csv"
            df_population.to_csv(output_file, index=False)
            
            # self.plot_and_export(algorithm.n_gen, X, min_f) # NOTE This plotting is to view the intermediate solutions

    def print_header(self):
        header = f"{'n_gen':>10} | {'F_min':>15} | {'Path_diff':>15}"
        print("=" * len(header))
        print(header)
        print("=" * len(header))

    def print_text(self, n_gen, min_f, path_diff):
        text = f"{n_gen:>10} | {min_f:>15.4f} | {path_diff:>15.4f}"
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