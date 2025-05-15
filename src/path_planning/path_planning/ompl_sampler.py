from ompl import base as ob
from ompl import geometric as og
import random
from custom_interfaces.msg import State, HelperPosition, HeadingAngle
import shapely
import time
from shapely.geometry import Polygon, Point, LineString
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
from typing import List, Tuple

class WaypointGuidedSampler(ob.StateSampler):
    """
    A custom state sampler for OMPL that generates random states near a given set of waypoints. It creates a LineString i.e.
    Polygon with buffer radius using the waypoints. This sampler is used for faster covergence of optimal path.
    
    Attributes:
        space (ob.StateSpace): The state space where the sampling occurs.
        waypoints (List[Tuple[float, float]]): Waypoints around which states are sampled.
        sampling_radius (float): Radius around each waypoint to limit the sampling.
        bounds (ob.RealVectorBounds): The bounds of the state space.
        sample_polygon (shapely.geometry.Polygon): The polygon created around the waypoints within the sampling radius.
    """

    def __init__(self, space: ob.StateSpace, waypoints: List[Tuple[float, float]], sampling_radius: float):
        super(WaypointGuidedSampler, self).__init__(space)
        self.space = space
        self.waypoints = waypoints
        self.sampling_radius = sampling_radius
        # TODO Add overlap of buffer polygon and the space bounds box
        self.bounds = space.getBounds()
        self.sample_polygon = self.createBufferPolygon(sampling_radius)

    def sampleUniform(self, state):
        """
        Samples a state uniformly within the buffer polygon around the waypoints.
        """

        min_x, min_y, max_x, max_y = self.sample_polygon.bounds
        while True:
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            potential_point = Point(x, y)
            if self.sample_polygon.contains(potential_point):
                break

        state.setXY(x, y)
        return state

    def createBufferPolygon(self, sampling_radius):

        points = [Point(wp) for wp in self.waypoints]
        line = LineString(points)
        line_buffer = line.buffer(sampling_radius, cap_style=2)
        simple_polygon = line_buffer.simplify(sampling_radius / 10.0)
        
        return simple_polygon
        
#     def plot_waypoints_and_polygon(self):

#         # Check if the buffer_polygon is valid and not empty
#         if self.sample_polygon.is_empty or not self.sample_polygon.is_valid:
#             raise ValueError("Buffer polygon is empty or invalid.")
        
#         fig, ax = plt.subplots()
#         ax.set_aspect('equal')

#         # Plot waypoints
#         x_coords, y_coords = zip(*self.waypoints)
#         ax.plot(x_coords, y_coords, 'o', label='Waypoints')

#         # Handle MultiPolygon objects correctly
#         if isinstance(self.sample_polygon, MultiPolygon):
#             for polygon in self.sample_polygon.geoms:  # Use .geoms to iterate over individual polygons
#                 WaypointGuidedSampler.plot_polygon(ax, polygon)
#         elif isinstance(self.sample_polygon, shapely.geometry.Polygon):
#             WaypointGuidedSampler.plot_polygon(ax, self.sample_polygon)  # If it's a single Polygon, plot it directly
#         else:
#             raise TypeError("Buffer polygon is neither a Polygon nor a MultiPolygon.")

#         # Set plot labels and legend
#         ax.set_title('Waypoints and Buffer Polygon')
#         ax.set_xlabel('X coordinate')
#         ax.set_ylabel('Y coordinate')
#         plt.legend()
        
#         # Save the figure
#         filename = f"Polygon_{time.time()}.png"
#         plt.savefig(f"imgs/{filename}")
#         plt.close()

#     @staticmethod
#     def plot_polygon(ax, polygon):
#         """Helper function to plot a single polygon using matplotlib's Polygon."""
#         if not polygon.is_empty and polygon.is_valid:
#             x, y = polygon.exterior.xy
#             # Create a Polygon patch from the exterior coordinates of the Shapely polygon
#             patch = MplPolygon(list(zip(x, y)))
#             patch.set_alpha(0.5)  # Set the alpha transparency
#             patch.set_zorder(2)   # Set the z-order
#             ax.add_patch(patch)
