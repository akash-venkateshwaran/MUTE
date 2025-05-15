import matplotlib.pyplot as plt
from typing import List
import math
from typing import List, Tuple, Union



def calculate_unit_vector(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate the unit vector from point p1 to point p2.
        
        Args:
            p1 (Tuple[float, float]): Coordinates of the starting point (x1, y1).
            p2 (Tuple[float, float]): Coordinates of the ending point (x2, y2).
        
        Returns:
            Tuple[float, float]: The unit vector (vx, vy) or (0, 0) if the magnitude is zero.
        """
        vector_x = p2[0] - p1[0]
        vector_y = p2[1] - p1[1]
        magnitude = math.sqrt(vector_x**2 + vector_y**2)
        
        if magnitude == 0:
            return (0, 0)  # Return zero vector if points are the same
        
        return (vector_x / magnitude, vector_y / magnitude)


def remove_loops_waypoints( waypoint_x: List[float], waypoint_y: List[float]):
        """
        Reduce the waypoints by removing points based on the dot product of unit vectors.

        Args:
            waypoint_x (List[float]): List of x-coordinates of waypoints.
            waypoint_y (List[float]): List of y-coordinates of waypoints.

        Returns:
            Tuple[List[float], List[float]]: Reduced waypoint_x and waypoint_y lists.
        """
        assert len(waypoint_x) == len(waypoint_y), "waypoint_x and waypoint_y must have the same length."
        waypoints = list(zip(waypoint_x, waypoint_y))
        i = 0
        removed_count = 0
        while i < len(waypoints) - 2:
            p1, p2, p3 = waypoints[i], waypoints[i + 1], waypoints[i + 2]
            vec1 = calculate_unit_vector(p1, p2)
            vec2 = calculate_unit_vector(p2, p3)
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            if dot_product < -0.5:
                waypoints.pop(i + 1)
                removed_count += 1
            else:
                i += 1
        reduced_x, reduced_y = zip(*waypoints) if waypoints else ([], [])
        print(f"Number of waypoints removed from loops: {removed_count}")
        return list(reduced_x), list(reduced_y)

# Function to plot waypoints
def plot_waypoints(before_x: List[float], before_y: List[float], after_x: List[float], after_y: List[float]):
    """
    Plot waypoints before and after reduction.

    Args:
        before_x (List[float]): Original x-coordinates of waypoints.
        before_y (List[float]): Original y-coordinates of waypoints.
        after_x (List[float]): Reduced x-coordinates of waypoints.
        after_y (List[float]): Reduced y-coordinates of waypoints.
    """
    plt.figure(figsize=(10, 6))

    # Plot original waypoints
    plt.plot(before_x, before_y, 'o-', label="Original Waypoints", color='blue')

    # Plot reduced waypoints
    plt.plot(after_x, after_y, 'o-', label="Reduced Waypoints", color='red', alpha=0.6)

    # Add labels and legend
    plt.title("Waypoints Before and After Reduction")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

# Test the WaypointReducer
if __name__ == "__main__":
    # Example waypoints
    waypoint_x = [0, 1, 2,  2, 3, 4, 5, 6]
    waypoint_y = [0, 1, 0,1.0,  -1, 0, 1, 0]

    reduced_x, reduced_y = remove_loops_waypoints(waypoint_x, waypoint_y)

    # Print results
    print("Original Waypoints (X, Y):")
    print(list(zip(waypoint_x, waypoint_y)))

    print("Reduced Waypoints (X, Y):")
    print(list(zip(reduced_x, reduced_y)))

    # Visualize the waypoints
    plot_waypoints(waypoint_x, waypoint_y, reduced_x, reduced_y)
