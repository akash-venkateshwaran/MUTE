import ompl.base as ob
import ompl.geometric as og
import matplotlib.pyplot as plt
import numpy as np


def create_path(space, si, num_states=20):
    """
    Creates a PathGeometric with random perturbations and turns.
    Converts a list of coordinates to base::State objects.
    """
    path = og.PathGeometric(si)

    # Generate states with perturbations
    for i in range(num_states):
        x = i  # x-coordinate
        y = np.sin(i * 0.2) * 5 + np.random.uniform(-1, 1)  # y-coordinate with sine and random noise
        theta = 0.0  # orientation (not used in this example)

        # Allocate a state and set its values
        state = si.allocState()
        state.setXY(x, y)

        # Append the state to the path
        path.append(state)
    return path


def path_to_coordinates(path):
    """
    Convert OMPL PathGeometric to a list of (x, y) coordinates.
    """
    coords = [(state[0][0], state[0][1]) for state in path.getStates()]
    return np.array(coords)


def plot_paths(original_coords, simplified_coords_list, titles):
    # original_coords is already Nx2
    plt.figure(figsize=(10, 8))

    # Plot original path
    plt.plot(original_coords[:, 0], original_coords[:, 1], label="Original Path", linewidth=2, marker="o")

    # Plot simplified paths
    for coords_array, title in zip(simplified_coords_list, titles):
        plt.plot(coords_array[:, 0], coords_array[:, 1], label=title, linewidth=2, linestyle="--", marker="x")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path Simplification")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Create SE2 state space
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-10)
    bounds.setHigh(30)
    space.setBounds(bounds)

    # SpaceInformation instance
    si = ob.SpaceInformation(space)
    si.setup()  # Make sure to setup before using PathSimplifier

    # Create the original path
    original_path = create_path(space, si)
    original_coords = path_to_coordinates(original_path)

    # Path simplifier
    path_simplifier = og.PathSimplifier(si)

    # Test various simplifications
    simplified_coords_list = []
    titles = []

    # Simplify Max
    simplified_path = og.PathGeometric(original_path)
    print(path_simplifier.simplifyMax(simplified_path))
    simplified_coords_list.append(path_to_coordinates(simplified_path))
    titles.append("Simplify Max")

    # Reduce Vertices
    simplified_path = og.PathGeometric(original_path)
    print(path_simplifier.reduceVertices(simplified_path, rangeRatio = 0.1))
    simplified_coords_list.append(path_to_coordinates(simplified_path))
    titles.append("Reduce Vertices")

    # Smooth B-Spline
    simplified_path = og.PathGeometric(original_path)
    print(path_simplifier.smoothBSpline(simplified_path))
    simplified_coords_list.append(path_to_coordinates(simplified_path))
    titles.append("Smooth B-Spline")

    # Plot all
    plot_paths(original_coords, simplified_coords_list, titles)

if __name__ == "__main__":
    main()
