#!/usr/bin/env python3

import math
import time
import matplotlib.pyplot as plt
import numpy as np
# OMPL imports (if installed via pip install ompl)
import ompl.base as ob
import ompl.geometric as og
import os
from ompl.base import PlannerData

from ompl import util as ou

###########################3
###########################
# IMPORTANT NOTSE
# - You don't need a termination class just use the default solve(10.0)
# - The increase in memory is due to plotting. Get rid of that


class MinimizeOriginDistanceObjective(ob.StateCostIntegralObjective):
    """
    A custom objective that integrates the distance from the origin along the path.
    This encourages the planner to find a path that stays as close to (0,0) as possible,
    minimizing the total "distance-from-origin" cost.
    """
    def __init__(self, si):
        # 'true' => this objective is "strictly" monotonic, i.e. shorter is always better
        super(MinimizeOriginDistanceObjective, self).__init__(si, True)

    def stateCost(self, state):
    # Extract x, y from the SE2 state
        x, y = state.getX(), state.getY()

        # Parameters for the first Gaussian (centered at origin)
        std1 = 1.0
        dist_from_origin = math.sqrt(x**2 + y**2)
        gaussian_center = (1 / (2 * math.pi * std1**2)) * math.exp(-dist_from_origin**2 / (2 * std1**2))

        # Parameters for the second Gaussian (centered at (5, 0))
        std2 = 2.5
        dist_from_center2 = math.sqrt((x - 7)**2 + y**2)
        gaussian_offset = 2*(1 / (2 * math.pi * std2**2)) * math.exp(-dist_from_center2**2 / (2 * std2**2))
        std3 = 2.5
        dist_from_center3 = math.sqrt((x + 7.0)**2 + (y - 7.0)**2)
        gaussian_offset2 = 2*(1 / (2 * math.pi * std3**2)) * math.exp(-dist_from_center3**2 / (2 * std3**2))

        # Combine the two Gaussian values (you can use a weighted sum if needed)
        combined_value = gaussian_center # + gaussian_offset + gaussian_offset2

        return ob.Cost(combined_value)




def create_simple_setup():
    """
    Create and configure a SimpleSetup using ABITstar in an SE2 space bounded by [-10,10] x [-10,10].
    The objective is to minimize the integrated distance from the origin.
    Returns the SimpleSetup instance.
    """

    # 1) Define the SE2 state space
    space = ob.SE2StateSpace()

    # 2) Set the bounds for x and y from -10 to 10
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, -10.0)
    bounds.setHigh(0, 10.0)
    bounds.setLow(1, -10.0)
    bounds.setHigh(1, 10.0)
    space.setBounds(bounds)

    # 3) Create the SimpleSetup
    simple_setup = og.SimpleSetup(space)

    # 4) Set a state validity checker (always valid in this demo)
    si = simple_setup.getSpaceInformation()
    svc = ob.StateValidityCheckerFn(lambda s: is_state_valid(si, s))
    simple_setup.setStateValidityChecker(svc)

    # 5) Define custom optimization objective: Minimize distance from origin
    objective = MinimizeOriginDistanceObjective(si)
    simple_setup.setOptimizationObjective(objective)

    # 6) Create the ABITstar planner
    planner = og.ABITstar(si)
    planner.setUseKNearest(True)  # ABIT* can use k-nearest or r-disc
    simple_setup.setPlanner(planner)

    return simple_setup


def set_start_and_goal(simple_setup):
    """
    Set start in the 3rd quadrant, goal in the 1st quadrant.
    For example:
      start = (-8, -8)
      goal  = ( 8,  8)
    """
    space = simple_setup.getStateSpace()

    start = ob.State(space)
    start_SE2 = start()
    start_SE2.setX(-8.0)
    start_SE2.setY(-8.0)
    start_SE2.setYaw(0.0)  # orientation in SE2

    goal = ob.State(space)
    goal_SE2 = goal()
    goal_SE2.setX(8.0)
    goal_SE2.setY(8.0)
    goal_SE2.setYaw(0.0)

    simple_setup.setStartAndGoalStates(start, goal)


def plot_solution_with_contour(simple_setup, iteration, export_path):
    """
    Plot the solution path (if it exists) along with start and goal.
    Overlay the contour plot of the cost function and export the plot.
    """
    # Generate the grid for the cost function
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    space = simple_setup.getStateSpace()
    objective = simple_setup.getOptimizationObjective()

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = space.allocState()
            state.setX(X[i][j])
            state.setY(Y[i][j])
            cost = objective.stateCost(state)
            Z[i][j] = cost.value()

    # Extract the solution path
    path_geometric = simple_setup.getSolutionPath()
    states = path_geometric.getStates()
    path_points = [(st.getX(), st.getY()) for st in states]
    xs, ys = zip(*path_points)

    # Start and goal
    start = path_points[0]
    goal = path_points[-1]

    # Clear the figure and plot
    plt.clf()
    plt.title(f"Iteration {iteration}: Path with Cost Contour")

    # Plot the cost function contour
    plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Cost Function Value")

    # Plot the solution path
    plt.plot(xs, ys, 'b-o', label="Path", linewidth=2)

    # Mark start and goal
    plt.plot(start[0], start[1], 'rs', label="Start", markersize=10)
    plt.plot(goal[0], goal[1], 'g*', label="Goal", markersize=12)

    # Set plot limits and grid
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.grid(True)
    plt.legend()

    # Export the plot
    filename = os.path.join(export_path, f"solution_iteration_{iteration}.png")
    plt.savefig(filename)
    plt.close()

def iterative_solve_demo():
        """
        Runs an iterative planning demo with ABITstar.
        Each iteration calls solve(...) with a timed PlannerTerminationCondition,
        prints the cost (if found), and displays the current path.
        """

        # Create the SimpleSetup & set start/goal
        ss = create_simple_setup()
        set_start_and_goal(ss)

        # Some iteration parameters
        num_iterations = 12
        per_iteration_time = 10.0  # seconds per iteration

        objective = ss.getOptimizationObjective()

        print("Starting iterative solve demo ...")

        for i in range(1, num_iterations + 1):
                print(f"\n=== Iteration {i} ===")

                start_time = time.time()

                ss.solve(per_iteration_time)
                end_time = time.time()

                planner = ss.getPlanner() 
                print("Number of iterations:", planner.numIterations())

                execution_time = end_time - start_time
                print(f"  -> Execution time for iteration {i}: {execution_time:.4f} seconds")

                if ss.haveSolutionPath():
                        path = ss.getSolutionPath()
                        cost_val = path.cost(objective).value()
                        print(f"  -> Found path with cost = {cost_val:.4f}")

                        # Plot the path
                        # plot_solution_with_contour(ss, iteration=i, export_path = '/workspaces/MELO_workspace/src/path_planning/test/temp')
                else:
                        print("  -> No solution found yet.")


        planner_data = PlannerData(ss.getSpaceInformation())
        ss.getPlanner().getPlannerData(planner_data)  # Populate PlannerData with current planner state

        # Retrieve graph properties
        num_vertices = planner_data.numVertices()  # Number of vertices (states)
        num_edges = planner_data.numEdges()        # Number of edges

        # Print the final graph statistics
        print("\n=== Final Graph Statistics ===")
        print(f"Number of vertices: {num_vertices}")
        print(f"Number of edges: {num_edges}")


def main():
    iterative_solve_demo()



if __name__ == "__main__":
    main()
