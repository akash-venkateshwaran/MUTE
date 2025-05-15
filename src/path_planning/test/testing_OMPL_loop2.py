#!/usr/bin/env python3

import math
import time
import os

import numpy as np
import matplotlib.pyplot as plt

# OMPL imports
import ompl.base as ob
import ompl.geometric as og
from ompl.base import PlannerData
from ompl import util as ou

# Adjust these as necessary for your project
from path_planning.new_objectives2 import get_ompl_objective
from pyproj import Proj, Geod
from typing import List

ou.setLogLevel(ou.LOG_DEBUG)


# Example definition for HelperPosition (float32 replaced by float in Python)
class HelperPosition:
    """
    HelperPosition:
    latitude: degrees, range: -90 <= latitude <= 90
    longitude: degrees, range: -180 <= longitude <= 180
    depth: meters
    """
    def __init__(self, latitude: float, longitude: float, depth: float = 0.0):
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth

# Reduce verbosity from OMPL



# ------------------------------------------------------------------------------
# 1. Define the main lat/lon references
# ------------------------------------------------------------------------------
# Single mammal position at "origin" (in lat/lon)
mammal_positions = [
    HelperPosition(latitude=49.287273533593155, longitude=-123.43513765779794, depth=0.0),
    HelperPosition(latitude=49.292273533593155, longitude=-123.33513765779794, depth=0.0),
    HelperPosition(latitude=49.242273533593155, longitude=-123.55513765779794, depth=0.0)
]

# Start lat/lon
start_lat = 49.267615006858904
start_lon = -123.5727675016527

# Goal lat/lon
goal_lat = 49.29750118269959
goal_lon = -123.20784781427093

# ------------------------------------------------------------------------------
# 2. Create a projection (proj) based on the start lat/lon, plus a geodesic
# ------------------------------------------------------------------------------
ref_lat = start_lat
ref_lon = start_lon
proj_string = f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +datum=WGS84"
proj = Proj(proj_string)
GEODESIC = Geod(ellps="WGS84")


# ------------------------------------------------------------------------------
# 3. Define helper functions: latlon_to_xy and xy_to_latlon
# ------------------------------------------------------------------------------
def latlon_to_xy(latlon: HelperPosition) -> tuple:
    """
    Convert a geographical coordinate to a 2D Cartesian coordinate
    using the fixed reference point. Output in kilometers.
    """
    x_m, y_m = proj(latlon.longitude, latlon.latitude)
    return (x_m / 1000.0, y_m / 1000.0)  # convert meters to kilometers

def xy_to_latlon(xy: tuple) -> HelperPosition:
    """
    Convert a 2D Cartesian coordinate (in km) back to lat/lon
    using the fixed reference point.
    """
    lon_deg, lat_deg = proj(xy[0] * 1000.0, xy[1] * 1000.0, inverse=True)
    return HelperPosition(latitude=lat_deg, longitude=lon_deg, depth=0.0)


# ------------------------------------------------------------------------------
# 4. OMPL Setup
# ------------------------------------------------------------------------------
def is_state_valid(si, state):
    """
    Example state validity checker.
    For this demo, we assume the entire space is collision-free.
    """
    return True

def create_simple_setup():
    """
    Create and configure a SimpleSetup using ABITstar in a 2D "SE2" style space,
    but using (x,y) in kilometers derived from lat/lon.
    """

    # Convert start and goal to (x, y)
    start_xy = latlon_to_xy(HelperPosition(start_lat, start_lon))
    goal_xy  = latlon_to_xy(HelperPosition(goal_lat, goal_lon))

    # Compute bounding box from start/goal with some margin
    xs = [start_xy[0], goal_xy[0]]
    ys = [start_xy[1], goal_xy[1]]
    margin = 5.0  # km of padding on each side

    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    # 1) Define the SE2 state space
    space = ob.SE2StateSpace()

    # 2) Set the bounds for x and y
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, min_x)
    bounds.setHigh(0, max_x)
    bounds.setLow(1, min_y)
    bounds.setHigh(1, max_y)
    space.setBounds(bounds)

    # 3) Create the SimpleSetup
    simple_setup = og.SimpleSetup(space)

    # 4) Set a state validity checker
    si = simple_setup.getSpaceInformation()
    svc = ob.StateValidityCheckerFn(lambda s: is_state_valid(si, s))
    simple_setup.setStateValidityChecker(svc)

    # 5) Create the ABITstar planner
    planner = og.ABITstar(si)
    # planner.setUseKNearest(True)  # ABIT* can use k-nearest or r-disc
    simple_setup.setPlanner(planner)

    return simple_setup

def set_start_and_goal(simple_setup):
    """
    Set start/goal states in XY (km) converted from lat/lon.
    """
    space = simple_setup.getStateSpace()

    # Convert lat/lon to xy
    start_xy = latlon_to_xy(HelperPosition(start_lat, start_lon))
    goal_xy  = latlon_to_xy(HelperPosition(goal_lat, goal_lon))

    # Create OMPL states
    start = ob.State(space)
    start_SE2 = start()
    start_SE2.setX(start_xy[0])
    start_SE2.setY(start_xy[1])
    start_SE2.setYaw(0.0)

    goal = ob.State(space)
    goal_SE2 = goal()
    goal_SE2.setX(goal_xy[0])
    goal_SE2.setY(goal_xy[1])
    goal_SE2.setYaw(0.0)

    simple_setup.setStartAndGoalStates(start, goal)

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_solution_with_contour(simple_setup, iteration, export_path):
    """
    Plots a contour map of the optimization objective over the state-space bounds,
    along with the current solution path (if it exists) and start/goal states.

    :param simple_setup: The ompl.geometric.SimpleSetup instance
    :param iteration:    Integer indicating the iteration number (used in the plot title/filename)
    :param export_path:  Directory where the output plot image is saved
    """

    # 1) Check if a solution exists
    if not simple_setup.haveSolutionPath():
        print("Solution does not exist")
        return

    # 2) Get the space and its bounds
    space = simple_setup.getSpaceInformation().getStateSpace()
    bounds = space.getBounds()
    min_x, max_x = bounds.low[0], bounds.high[0]
    min_y, max_y = bounds.low[1], bounds.high[1]

    # 3) Set up a grid (increase or decrease resolution as desired)
    resolution = 100
    x_vals = np.linspace(min_x, max_x, resolution)
    y_vals = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # 4) Evaluate the objective on the grid
    objective = simple_setup.getOptimizationObjective()

    # We'll allocate one state repeatedly (for performance)
    state = space.allocState()
    for i in range(resolution):
        for j in range(resolution):
            state.setX(X[i, j])
            state.setY(Y[i, j])
            cost_val = objective.stateCost(state).value()
            Z[i, j] = cost_val
    space.freeState(state)

    # 5) Retrieve the solution path
    path_geom = simple_setup.getSolutionPath()
    states = path_geom.getStates()
    path_points = [(st.getX(), st.getY()) for st in states]

    # 6) Plot contour + path
    plt.figure(figsize=(8, 6))
    plt.title(f"Solution at iteration {iteration}")

    # Contour plot of the objective function
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label="Objective Value")

    # Plot the path
    path_x, path_y = zip(*path_points)
    plt.plot(path_x, path_y, 'k-', linewidth=2, label="Solution Path")

    # Mark start & goal
    start_pt = path_points[0]
    goal_pt  = path_points[-1]
    plt.plot(start_pt[0], start_pt[1], 'ro', label="Start")
    plt.plot(goal_pt[0], goal_pt[1], 'go', label="Goal")

    # Configure axes and legend
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.legend()
    plt.grid(True)

    # 7) Save the plot
    os.makedirs(export_path, exist_ok=True)
    filename = os.path.join(export_path, f"solution_iteration_{iteration}.png")
    plt.savefig(filename)
    plt.close()

    print(f"Saved solution plot to {filename}")

def print_solution_path_and_cost(ss, objective):
    """
    Prints the final solution path from the given SimpleSetup,
    along with the total path cost and the per-state cost.
    """
    if not ss.haveSolutionPath():
        print("No final solution found; cannot print path or cost.")
        return

    path = ss.getSolutionPath()            # ompl.geometric.PathGeometric
    path_cost_ompl = path.cost(objective).value()
    path_states = path.getStates()

    print("\n=== Final Solution Path ===")
    print(f"OMPL-reported total path cost: {path_cost_ompl:.4f}")

    # (Optional) Print the per-state cost
    for i, st in enumerate(path_states):
        x = st.getX()
        y = st.getY()
        sc = objective.stateCost(st).value()
        print(f"  State {i}: (x={x:.4f}, y={y:.4f}), cost = {sc:.6f}")

def create_and_print_custom_path_cost(ss, objective):
    """
    Creates a custom path with fixed coordinates and computes the cost
    under the same 'objective'.
    """
    si = ss.getSpaceInformation()
    space = si.getStateSpace()

    # -- 1) Define our custom coordinates in the same (x,y) space units (km).
    custom_coords = [(0, 0),
                     (5, -4),
                     (20,-2),
                     (26.5424, 3.3879)]

    # -- 2) Construct a PathGeometric object
    custom_path = og.PathGeometric(si)

    # -- 3) For each (x,y), allocate a SE2 state, set the position, and append
    for (cx, cy) in custom_coords:
        st = space.allocState()
        st.setX(cx)
        st.setY(cy)
        st.setYaw(0.0)
        custom_path.append(st)

    # -- 4) Compute the path cost
    path_cost_ompl= custom_path.cost(objective).value()

    custom_path_states = custom_path.getStates()

   # -- 5) Print results
    print("\n=== Custom Path ===")
    print(f"OMPL-reported total path cost: {path_cost_ompl:.4f}")

    
    for i, st in enumerate(custom_path.getStates()):
        x = st.getX()
        y = st.getY()
        # optional: per-state cost
        state_cost = objective.stateCost(st).value()
        print(f"  State {i}: (x={x:.2f}, y={y:.2f}), cost = {state_cost:.6f}")

    # -- 6) Free the allocated states (optional; Python’s GC usually handles it)
    for st in custom_path.getStates():
        space.freeState(st)



def plot_ompl_graph(planner_data, export_path=None):
    """
    Plots vertices and edges from a PlannerData object and optionally saves the figure.
    
    :param planner_data: The OMPL PlannerData object containing the graph.
    :param export_path:  (Optional) Path to save the plot image. If None, the plot is not saved.
    """
    import matplotlib.pyplot as plt

    # ---------------------
    # 1. Extract the 2D (x, y) coordinates of each vertex
    # ---------------------
    vertices = []
    for i in range(planner_data.numVertices()):
        state = planner_data.getVertex(i).getState()
        x = state.getX()
        y = state.getY()
        vertices.append((x, y))

    # ---------------------
    # 2. Plotting
    # ---------------------
    plt.figure(figsize=(10, 8))
    plt.title("OMPL Search Graph")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)





def iterative_solve_demo():
    """
    Creates the SimpleSetup, sets the start/goal, sets the new objective,
    and iteratively calls solve(...) to demonstrate incremental improvement.
    """

    # 1) Create the SimpleSetup
    ss = create_simple_setup()
    si = ss.getSpaceInformation()

    # 2) Set start and goal states
    set_start_and_goal(ss)

    # 3) Set our new objective
    #    from path_planning.new_objectives2 import get_ompl_objective
    #    Here we call it, passing the needed arguments.
    objective = get_ompl_objective(
        space_information=si,
        simple_setup=ss,
        mammal_positions=mammal_positions,  # list of HelperPosition
        projector=proj,
        GEODESIC=GEODESIC
    )
    ss.setOptimizationObjective(objective)

    # 4) Solve in multiple iterations
    num_iterations = 10
    per_iteration_time = 30.0  # seconds per iteration

    print("Starting iterative solve demo ...")

    for i in range(1, num_iterations + 1):
        print(f"\n=== Iteration {i} ===")
        
        start_time = time.time()
        ss.solve(per_iteration_time)
        end_time = time.time()

        planner = ss.getPlanner() 
        print("Number of planner iterations:", planner.numIterations())
        # print("Initial Parameters:", planner.getCurrentInflationFactor(), planner.getCurrentTruncationFactor(), "Scaling:",planner.getInflationScalingParameter() )
        # print("Initial Samples per batch", planner.getSamplesPerBatch())

        execution_time = end_time - start_time
        print(f"  -> Execution time for iteration {i}: {execution_time:.4f} seconds")

        if ss.haveSolutionPath():
            path = ss.getSolutionPath()
            cost_val = path.cost(objective).value()
            print(f"  -> Found path with cost = {cost_val:.4f}")

            # Optionally plot (commented out to reduce memory usage)
            plot_solution_with_contour(ss, iteration=i, export_path="/workspaces/MELO_workspace/src/path_planning/test/temp")
        else:
            print("  -> No solution found yet.")

    # 5) After all iterations, we can get the final planner data
    planner_data = PlannerData(si)
    ss.getPlanner().getPlannerData(planner_data)

    num_vertices = planner_data.numVertices()
    num_edges = planner_data.numEdges()
    print("\n=== Final Graph Statistics ===")
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of edges: {num_edges}")
    export_graph_path = "/workspaces/MELO_workspace/src/path_planning/test/temp/ompl_graph.png"
    plot_ompl_graph(planner_data, export_path=export_graph_path)
    
    print_solution_path_and_cost(ss, objective)
    create_and_print_custom_path_cost(ss, objective)    


def main():
    iterative_solve_demo()


if __name__ == "__main__":
    main()
