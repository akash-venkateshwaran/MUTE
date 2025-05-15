from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
from typing import List, Tuple, Union
from pyproj import Geod
import math
import shapely
from shapely.geometry import Polygon, Point, LineString
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon

import csv
import os

import time


import numpy as np
import matplotlib.pyplot as plt

from path_planning.new_objectives2 import get_ompl_objective 
from custom_interfaces.msg import State, HelperPosition
from path_planning.ompl_sampler import WaypointGuidedSampler
from pyproj import Proj


# ou.setLogLevel(ou.LOG_WARN) # NOTE This is to print the information executed in OMPL


class OMPLPathState:
    def __init__(self,ship_current_state: State, mammal_current_states: np.ndarray, ship_start_state: State, ship_end_state: State, bounding_box: list, count: float, shipplanes_polygons: List[Polygon], GEODESIC: Geod, TL_model_path: str, previous_waypoints: List[HelperPosition] = None):
        

        self.GEODESIC = GEODESIC
        self.TL_model_path = TL_model_path

        # Creating HelperPosition for all the inputs
        self.mammal_positions = [self.state_to_helperposition(state) for state in mammal_current_states]
        self.ship_position = self.state_to_helperposition(ship_current_state)
        self.start_position = self.reference_latlon = self.state_to_helperposition(ship_start_state)


        ref_lat, ref_lon = self.reference_latlon.latitude, self.reference_latlon.longitude
        self.proj_string = f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +datum=WGS84"
        self.proj = Proj(self.proj_string) # The projector used for conversion

        self.goal_position = self.state_to_helperposition(ship_end_state)
        self.count = count
        self.shipplanes_polygons = shipplanes_polygons


        # Coverting HelperPosition from lat/lon to XY
        self.start_position_XY = self.latlon_to_xy(self.start_position)
        self.goal_position_XY = self.latlon_to_xy(self.goal_position)
        self.mammal_position_XY = [self.latlon_to_xy(pos) for pos in self.mammal_positions]
        self.ship_position_XY = self.latlon_to_xy(self.ship_position) 
        self.shipplanes_polygons_XY = []
        for poly in shipplanes_polygons:
            xy_polygon = self.convert_polygon_latlon_to_xy(polygon=poly)
            self.shipplanes_polygons_XY.append(Polygon(xy_polygon))

        self.mammal_current_states_XY = [
        State(
            position=HelperPosition(latitude=pos[0], longitude=pos[1], depth=mammal_current_states[i].position.depth),
            angle=mammal_current_states[i].angle,
            speed=mammal_current_states[i].speed
        ) for i, pos in enumerate(self.mammal_position_XY)]

        # The state bounds for OMPL
        self.domain_lat = (bounding_box[0], bounding_box[2])
        self.range_lon = (bounding_box[1], bounding_box[3])


        # We'll use the four corners: (min_lat, min_lon), (min_lat, max_lon), (max_lat, min_lon), (max_lat, max_lon)
        # Then find the min and max XY values from these points
        min_lat_xy = self.latlon_to_xy(HelperPosition(latitude=self.domain_lat[0], longitude=self.range_lon[0], depth=0.0))
        max_lat_xy = self.latlon_to_xy(HelperPosition(latitude=self.domain_lat[1], longitude=self.range_lon[0], depth=0.0))
        min_lon_xy = self.latlon_to_xy(HelperPosition(latitude=self.domain_lat[0], longitude=self.range_lon[1], depth=0.0))
        max_lon_xy = self.latlon_to_xy(HelperPosition(latitude=self.domain_lat[1], longitude=self.range_lon[1], depth=0.0))

        # Now extract the minimum and maximum X and Y values to define the domain and range in XY coordinates
        min_x = min(min_lat_xy[0], max_lat_xy[0], min_lon_xy[0], max_lon_xy[0])
        max_x = max(min_lat_xy[0], max_lat_xy[0], min_lon_xy[0], max_lon_xy[0])
        min_y = min(min_lat_xy[1], max_lat_xy[1], min_lon_xy[1], max_lon_xy[1])
        max_y = max(min_lat_xy[1], max_lat_xy[1], min_lon_xy[1], max_lon_xy[1])

        # Assign domain_XY and range_XY using the minimum and maximum XY values
        self.domain_XY = (min_x, max_x)
        self.range_XY = (min_y, max_y)

        # Variables for convergence plot
        self.path_cost = []
        self.path_goal_dist = []
        self.convergence_time = []

        self.previous_waypoints_XY = [self.latlon_to_xy(wp) for wp in previous_waypoints] if previous_waypoints else []
        self.sampler_buffer_r = 5.0



    def state_to_helperposition(self, state):
            return HelperPosition(
                    latitude=state.position.latitude,
                    longitude=state.position.longitude,
                    depth=state.position.depth
                )

    def allocWaypointGuidedSampler(self, space):
        return WaypointGuidedSampler(space, self.previous_waypoints_XY, self.sampler_buffer_r)
    
    def latlon_to_xy(self, latlon: HelperPosition) -> tuple:
        """Convert a geographical coordinate to a 2D Cartesian coordinate using the fixed reference point."""
        x, y = self.proj(latlon.longitude, latlon.latitude)
        return (x / 1000, y / 1000)  # Convert meters to kilometers

    def xy_to_latlon(self, xy: tuple) -> HelperPosition:
        """Convert a 2D Cartesian coordinate to a geographical coordinate using the fixed reference point."""
        lon, lat = self.proj(xy[0] * 1000, xy[1] * 1000, inverse=True)  # Convert kilometers back to meters
        return HelperPosition(latitude=lat, longitude=lon, depth=0.0)
    

    
    def convert_polygon_latlon_to_xy(self, polygon: Polygon) -> List[tuple]:
        """Convert a Polygon from geographical coordinates to 2D Cartesian coordinates using the fixed reference point."""
        xy_points = []
        for point in polygon.exterior.coords:
            latlon = HelperPosition(latitude=point[1], longitude=point[0], depth=0.0)  # Assuming no depth for shoreline
            xy = self.latlon_to_xy(latlon) 
            xy_points.append(xy)
        return xy_points



class OMPLPath:
    """Represents the general OMPL Path.

    Attributes
        _logger (RcutilsLogger): ROS logger of this class.
        _simple_setup (og.SimpleSetup): OMPL SimpleSetup object.
        solved (bool): True if the path is a solution to the OMPL query, else false.
    """

    def __init__(
        self,
        ompl_state: OMPLPathState,
        # parent_logger: RcutilsLogger,
        max_runtime: float
    ):
        """Initialize the OMPLPath Class. Attempt to solve for a path.

        Args:
            parent_logger (RcutilsLogger): Logger of the parent class.
            max_runtime (float): Maximum amount of time in seconds to look for a solution path.
            local_path_state (LocalPathState): State of Sailbot.
        """
        # self._logger = parent_logger.get_child(name="ompl_path")
        self.ompl_state = ompl_state
        self._simple_setup = self._init_simple_setup(ompl_state)
        self.simplification_time = 1.0  #  Set the max simplification time for PathSimplifier in seconds
        self.path_simplifier = og.PathSimplifier(self._simple_setup.getSpaceInformation())

        self.solved = False


        if ompl_state.previous_waypoints_XY:
            print("Using previous waypoints to guide planning for faster convergence")
            space = self._simple_setup.getStateSpace()
            space.setStateSamplerAllocator(ob.StateSamplerAllocator(ompl_state.allocWaypointGuidedSampler))
            self.solve_incrementally(max_runtime, increment=30.0)
            
        else:
            print("Starting planning from scratch. Might take time to converge")
            self.solve_incrementally(max_runtime, increment=10.0)
        # self.fig = plt.figure()


    def get_ompl_solution(self):
        if not self.solved:
            print("Trying to get the waypoints of an unsolved OMPLPath")
            return []

        solution_path = self._simple_setup.getSolutionPath()
        waypoints_x = []
        waypoints_y = []

        for state in solution_path.getStates():
            waypoints_x.append(state.getX())
            waypoints_y.append(state.getY())

        return waypoints_x, waypoints_y


    def get_waypoints(self, format: str = 'latlon') -> Union[List[HelperPosition], List[Tuple[float, float]]]:
        """Gets the waypoints, performs smoothening and removes loops in the path.

        Args:
            format (str): The format of the waypoints, 'latlon' for latitude and longitude, 'xy' for XY coordinates.

        Returns:
            list: A list of waypoints. Each waypoint is a HelperPosition or tuple representing either the latitude and longitude or the X and Y coordinates,
                depending on the specified format. Outputs an empty list and prints a warning message if path not solved.
        """

        waypoints_x, waypoints_y = self.get_ompl_solution()

        final_waypoints_x, final_waypoints_y =  waypoints_x, waypoints_y

        if format == 'xy':
            waypoints = [(x, y) for x, y in zip(final_waypoints_x, final_waypoints_y)]
        else:
            waypoints = [
                self.ompl_state.xy_to_latlon(xy=(x, y))
                for x, y in zip(final_waypoints_x, final_waypoints_y)
            ]

        return waypoints
    
    def export_waypoints_to_csv(self, waypoints: List[HelperPosition], export_path: str):
        os.makedirs(export_path, exist_ok=True)
        file_path = os.path.join(export_path, 'optimal_path.csv')
        with open(file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Latitude', 'Longitude'])
            csv_writer.writerows([[wp.latitude, wp.longitude] for wp in waypoints])
        print(f"{len(waypoints)} Waypoints exported to {file_path}")


    
    def solve_incrementally(self, max_runtime, increment=30.0):
        start_time = time.time()

        while time.time() - start_time < max_runtime:
            increment_start = time.time()
            elapsed_increment = 0

            while elapsed_increment < increment:
                self._simple_setup.solve(increment)
                self.solved = self._simple_setup.haveSolutionPath()
                elapsed_increment = time.time() - increment_start

            if self.solved:
                solution_path = self._simple_setup.getSolutionPath()

                # NOTE Use this path_simplifier if needed
                # self.path_simplifier.reduceVertices(solution_path, maxSteps=30, maxEmptySteps=30, rangeRatio=0.05)

                current_cost = solution_path.cost(self._simple_setup.getOptimizationObjective()).value()
                           
                last_waypoint_distance = self.distance_from_goal()

                self.ompl_state.path_cost.append(current_cost)
                self.ompl_state.path_goal_dist.append(last_waypoint_distance)
                self.ompl_state.convergence_time.append(time.time() - start_time)
 

        print(f"Finished solving incrementally in {time.time() - start_time}")

        


    def distance_from_goal(self):
        if self.solved:
            waypoints_x, waypoints_y = self.get_ompl_solution()
            waypoints = list(zip(waypoints_x, waypoints_y))
            if waypoints:
                last_waypoint = waypoints[-1]
                goal = (self.ompl_state.goal_position_XY[0], self.ompl_state.goal_position_XY[1])
                distance = math.sqrt((last_waypoint[0] - goal[0]) ** 2 + (last_waypoint[1] - goal[1]) ** 2)
                return distance  # Distance in the same units as your state space (typically meters or kilometers)
        return float('inf')  # Return an infinite distance if unsolved or no waypoints


    def _init_simple_setup(self,ompl_state: OMPLPathState) -> og.SimpleSetup:
        """Initialize and configure the OMPL SimpleSetup object.

        Returns:
            og.SimpleSetup: Encapsulates the various objects necessary to solve a geometric or
                control query in OMPL.
        """
        space = ob.SE2StateSpace()
        self.ompl_state = ompl_state

        # set the bounds of the state space
        bounds = ob.RealVectorBounds(dim=2)
        bounds.setLow(index=0, value=self.ompl_state.domain_XY[0])
        bounds.setLow(index=1, value=self.ompl_state.range_XY[0])
        bounds.setHigh(index=0, value=self.ompl_state.domain_XY[1])
        bounds.setHigh(index=1, value=self.ompl_state.range_XY[1])

        print(
            "state space bounds: "
            f"x=[{bounds.low[0]}, {bounds.high[0]}]; "
            f"y=[{bounds.low[1]}, {bounds.high[1]}]"
        )
        bounds.check()  # check if bounds are valid
        space.setBounds(bounds)

        # create a simple setup object
        simple_setup = og.SimpleSetup(space)
        simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

        # set the goal and start states of the simple setup object
        start = ob.State(space)
        goal = ob.State(space)
        start().setXY(self.ompl_state.ship_position_XY[0], self.ompl_state.ship_position_XY[1])
        goal().setXY(self.ompl_state.goal_position_XY[0], self.ompl_state.goal_position_XY[1])



        print(
            "start and goal state: "
            f"start=({start().getX()}, {start().getY()}); "
            f"goal=({goal().getX()}, {goal().getY()})"
        )
        
        simple_setup.setStartAndGoalStates(start, goal)

        space_information = simple_setup.getSpaceInformation()
        if space_information is None:
            print("Space information not initialized correctly.")
            return None

        objective = get_ompl_objective(
            space_information=space_information,
            simple_setup=simple_setup,
            mammal_positions=self.ompl_state.mammal_positions,  # List of HelperPosition.msg objects
            projector=self.ompl_state.proj, GEODESIC = self.ompl_state.GEODESIC, TL_model_path = self.ompl_state.TL_model_path          # Projection object
        )


        if objective is None:
            print("Objective not created correctly.")
            return None
        simple_setup.setOptimizationObjective(objective)

        planner = og.ABITstar(space_information)

        print("Scaling factors of ABITstar planner:", planner.getInflationScalingParameter(), planner.getTruncationScalingParameter())
        planner.setUseKNearest(True)
        if planner is None:
            print("Planner is not initialized correctly.")
            return None

        simple_setup.setPlanner(planner)

        # Setting the path direction and motion constraint NOTE ReverseMotionValidator is not used. Check the reason at the bottom
        # start_x, start_y = self.ompl_state.ship_position_XY
        # goal_x, goal_y = self.ompl_state.goal_position_XY
        # self.path_vector = calculate_unit_vector(p1 = (start_x, start_y), p2 = (goal_x, goal_y))
        # motion_validator = ReverseMotionValidator(space_information, self.path_vector, self.is_state_valid)
        # space_information.setMotionValidator(motion_validator)

        return simple_setup
    
    
    def plot_ompl_solution(self, timesteps, ship_state: State, mammal_states: np.ndarray, waypoints = List[HelperPosition]):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = np.linspace(self.ompl_state.domain_XY[0], self.ompl_state.domain_XY[1], 250)
        y = np.linspace(self.ompl_state.range_XY[0], self.ompl_state.range_XY[1], 250)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)

        space = self._simple_setup.getStateSpace()
        objective = self._simple_setup.getOptimizationObjective()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                state = space.allocState()
                state.setX(X[i][j])
                state.setY(Y[i][j])
                cost = objective.stateCost(state)
                Z[i][j] = cost.value()

        cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        fig.colorbar(cp, ax=ax, orientation='vertical', label='Cost')

        ax.set_title('Solution Space with Waypoints')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        for poly in self.ompl_state.shipplanes_polygons_XY:
            x, y = poly.exterior.xy
            vertices = list(zip(x, y)) + [(x[0], y[0])]
            codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
            path = Path(vertices, codes)
            patch = PathPatch(path, facecolor='black', lw=2, edgecolor='black', alpha=0.5, zorder=10)
            ax.add_patch(patch)

        ax.set_xlim(self.ompl_state.domain_XY)
        ax.set_ylim(self.ompl_state.range_XY)

        if waypoints:
            waypoints_xy = []
            for waypoint in waypoints:
                x, y = self.ompl_state.latlon_to_xy(waypoint)
                waypoints_xy.append((x, y))
            waypoints_x = [wp[0] for wp in waypoints_xy]
            waypoints_y = [wp[1] for wp in waypoints_xy]

            ax.plot(waypoints_x, waypoints_y, '-', label='Waypoints XY', color='white')


        ship_position = self.ompl_state.state_to_helperposition(ship_state)
        ship_position_XY = self.ompl_state.latlon_to_xy(latlon = ship_position)

        start_x, start_y = self.ompl_state.start_position_XY
        goal_x, goal_y = self.ompl_state.goal_position_XY

        # Plot the scatter points
        ax.scatter([start_x], [start_y], c='green', label='Start Position', zorder=5, edgecolors='black')
        ax.scatter([goal_x], [goal_y], c='orange', label='Goal Position', zorder=5, edgecolors='black')
        ax.scatter([ship_position_XY[0]], [ship_position_XY[1]], c='blue', label='Ship Position', zorder=5, edgecolors='black')

        mammal_positions_XY = []

        for mammal_state in mammal_states:
            mammal_position = self.ompl_state.state_to_helperposition(mammal_state)
            mammal_position_XY = self.ompl_state.latlon_to_xy(latlon = mammal_position)
            mammal_positions_XY.append(mammal_position_XY)

        mammal_positions_X = [pos[0] for pos in mammal_positions_XY]
        mammal_positions_Y = [pos[1] for pos in mammal_positions_XY]

        ax.scatter(mammal_positions_X, mammal_positions_Y, c='red', label='Mammals', zorder=5, edgecolors='black')
        

        # Get the plot limits
        padding = 0.3  # Adjust this value as needed
        x_min, x_max = min(start_x, goal_x), max(start_x, goal_x)
        y_min, y_max = min(start_y, goal_y), max(start_y, goal_y)   
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        ax.legend()

        filename = f"OMPL_{timesteps}.png"
        fig.savefig("imgs/" + filename, dpi=300)
        plt.close()

    def plot_convergence(self, filename = "OMPL_convergence.png"):
            
            fig, ax1 = plt.subplots()
            
            color1 = 'tab:blue'
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Path Cost', color=color1)
            ax1.plot(self.ompl_state.convergence_time,
                    self.ompl_state.path_cost, 
                    marker='o', color=color1, label='Path Cost')
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # Create the second y-axis for the Path Goal Distance
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Path Goal Distance', color=color2)
            ax2.plot(self.ompl_state.convergence_time,
                    self.ompl_state.path_goal_dist, 
                    marker='x', color=color2, label='Path Goal Distance')
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Add a title and grid
            plt.title('Convergence of Path Cost and Goal Distance over Time')
            plt.grid(True)
            
            fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
            plt.savefig("imgs/" + filename)
            plt.clf()            
            plt.close('all')



    def is_state_valid(self, state: ob.SE2StateSpace) -> bool:
        # Checker function to make sure all states are in the lane
        point = Point(state.getX(), state.getY())  # Convert the OMPL state to a Shapely Point
        for polyg in self.ompl_state.shipplanes_polygons_XY:  
            if polyg.contains(point):
                return True
        return False

    

    

# NOTE This class was implemented to prevent looping however it doesn't work cause path_vector 
# is always fixed form start to goal. You want to compare with vector that changes along the line.
# HENCE NOT USED:

# class ReverseMotionValidator(ob.MotionValidator):
#     def __init__(self, si, path_vector, state_validity_checker):
#         super().__init__(si)
#         self.si = si
#         self.path_vector = path_vector
#         self.state_validity_checker = state_validity_checker

#     def checkMotion(self, s1, s2):
#         x1, y1 = s1.getX(), s1.getY()
#         x2, y2 = s2.getX(), s2.getY()

#         move_unit_vector = calculate_unit_vector(p1 = (x1, y1), p2 = (x2, y2))
#         magnitude = math.hypot(move_unit_vector[0], move_unit_vector[1])
#         if magnitude == 0:
#             return True  # No movement, valid


#         # Compute the dot product with the path_vector
#         dot_product = move_unit_vector[0] * self.path_vector[0] + move_unit_vector[1] * self.path_vector[1]

#         if dot_product < -0.5:
#             return False  # Reverse motion detected

#         # Perform collision checking between s1 and s2
#         resolution = self.si.getStateValidityCheckingResolution()
#         num_steps = int(magnitude / resolution) + 1

#         state_space = self.si.getStateSpace()
#         test_state = state_space.allocState()

#         for i in range(1, num_steps + 1):
#             t = float(i) / num_steps
#             state_space.interpolate(s1, s2, t, test_state)
#             if not self.state_validity_checker(test_state):
#                 state_space.freeState(test_state)
#                 return False  # Invalid intermediate state

#         state_space.freeState(test_state)
#         return True  # All checks passed


# def calculate_unit_vector(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
#         """
#         Calculate the unit vector from point p1 to point p2.
        
#         Args:
#             p1 (Tuple[float, float]): Coordinates of the starting point (x1, y1).
#             p2 (Tuple[float, float]): Coordinates of the ending point (x2, y2).
        
#         Returns:
#             Tuple[float, float]: The unit vector (vx, vy) or (0, 0) if the magnitude is zero.
#         """
#         vector_x = p2[0] - p1[0]
#         vector_y = p2[1] - p1[1]
#         magnitude = math.sqrt(vector_x**2 + vector_y**2)
        
#         if magnitude == 0:
#             return (0, 0)  # Return zero vector if points are the same
        
#         return (vector_x / magnitude, vector_y / magnitude)
