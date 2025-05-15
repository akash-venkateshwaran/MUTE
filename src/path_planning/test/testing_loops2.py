import ompl.base as ob
import ompl.geometric as og
import matplotlib.pyplot as plt
import numpy as np

def create_looping_path(space, si, num_states=50):
    """
    Creates a PathGeometric with loops.
    """
    path = og.PathGeometric(si)

    # waypoint_x = [0, 1, 2,  2, 3, 4, 5, 6]
    # waypoint_y = [0, 1, 0,1.0,  -1, 0, 1, 0]

    # for i in range(len(waypoint_x)):
    #     state = si.allocState()
    #     state.setXY(waypoint_x[i], waypoint_y[i])

    #     path.append(state)


    for i in range(num_states):
        x = i  # x-coordinate

        # Create loops by adjusting y-coordinate
        if i < num_states // 3:
            y = np.sin(i * 0.2) * 5
        elif i < 2 * num_states // 3:
            y = np.cos((i - num_states // 3) * 0.3) * 8  # loop section
        else:
            y = np.sin((i - 2 * num_states // 3) * 0.2) * 5

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

def plot_paths(original_coords, simplified_coords, titles):
    """
    Plot original and simplified paths.
    """
    plt.figure(figsize=(10, 8))

    # Plot original path
    plt.plot(original_coords[:, 0], original_coords[:, 1], label="Original Path", linewidth=2, marker="o")

    # Plot simplified path
    plt.plot(simplified_coords[:, 0], simplified_coords[:, 1], label=titles, linewidth=2, linestyle="--", marker="x")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path Simplification with Loop Removal")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Create SE2 state space
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-10)
    bounds.setHigh(60)
    space.setBounds(bounds)

    # SpaceInformation instance
    si = ob.SpaceInformation(space)
    si.setup()  # Make sure to setup before using PathSimplifier

    # Create the original path with loops
    original_path = create_looping_path(space, si)
    original_coords = path_to_coordinates(original_path)

    # Path simplifier
    path_simplifier = og.PathSimplifier(si)

    # Simplify path to remove loops
    simplified_path = og.PathGeometric(original_path)
    path_simplifier.reduceVertices(simplified_path, maxSteps=0, maxEmptySteps=0, rangeRatio = 0.05)  # Specify maxSteps and maxEmptySteps
    # path_simplifier.smoothBSpline(simplified_path)
    simplified_coords = path_to_coordinates(simplified_path)

    # Plot before and after simplification
    plot_paths(original_coords, simplified_coords, "Simplified Path")

if __name__ == "__main__":
    main()
