from ompl import base as ob
from typing import List
import numpy as np
from pyproj import Geod
from custom_interfaces.msg import State, HelperPosition
import math
from pyproj import Proj

from noise_propagation_model.models.RBF.utils_rbf import RBFInterpolatorCustom





class Objective(ob.StateCostIntegralObjective):
    """All of our optimization objectives inherit from this class.

    Attributes:
        space_information (StateSpacePtr): Contains all the information about
            the space planning is done in.
        max_motion_cost (float): The maximum motion cost between any two states in the state space.
    """

    def __init__(self, space_information, num_samples: int):
        super().__init__(si=space_information, enableMotionCostInterpolation=True)
        self.space_information = space_information

        states = self.sample_states(num_samples)
        # initialize to 1 so that motionCost is not normalized when finding the maximum motion cost
        self.max_state_cost = 1.0
        self.max_state_cost = self.find_maximum_state_cost(states)

    def find_maximum_state_cost(self, states: list[ob.SE2StateSpace]) -> float:
        """Finds the maximum motion cost between any two states in `states`.

        Args:
            states (list[ob.SE2StateSpace]): OMPL states.

        Returns:
            float: Maximum motion cost.
        """

        return max(abs(self.stateCost(s).value())
            for s in states)

    def sample_states(self, num_samples: int) -> list[ob.SE2StateSpace]:
        """Samples `num_samples` states from the state space.

        Args:
            num_samples (int): Number of states to sample.

        Returns:
            list[ob.SE2StateSpace]: OMPL states.
        """
        sampler = self.space_information.getStateSpace().allocDefaultStateSampler()

        sampled_states = []

        for _ in range(num_samples):
            state = self.space_information.getStateSpace().allocState()
            sampler.sampleUniform(state)
            sampled_states.append(state)

        return sampled_states

    def normalization(self, cost: float) -> float:
        """Normalizes cost using max_motion_cost and caps it at 1.

        Args:
            cost (float): motionCost value from an objective function.
        Returns:
            float: normalized cost between 0 to 1.
        """
        normalized_cost = cost / self.max_state_cost
        return normalized_cost
    



class NoiseObjective(Objective):
    def __init__(self, space_information, simple_setup, mammal_positions: List[HelperPosition], projector: Proj, GEODESIC : Geod, TL_model_path: str):
        self.GEODESIC = GEODESIC
        self.space_information = space_information
        self.simple_setup = simple_setup
        self.mammal_positions_array = np.array(
            [[mammal.latitude, mammal.longitude] for mammal in mammal_positions]
        )  # Array of size (N,2)

        self.approx_max_tl = self.get_approx_max_TL()
        self.projector = projector
        # self.TL_model = RBFInterpolatorCustom.import_model(TL_model_path)
        super().__init__(space_information, num_samples=100)

    def inverse_projection(self, x, y):
        """
        Converts x, y coordinates to latitude and longitude using the provided projection.
        
        Args:
            x (float): X-coordinate in km.
            y (float): Y-coordinate in km.

        Returns:
            lat (float): Latitude.
            lon (float): Longitude.
        """
        lon, lat = self.projector(x * 1000, y * 1000, inverse=True)
        return lat, lon
    
    def get_approx_max_TL(self):

        """ Computed TL based on the distance btw start and goal"""
        problem_definition = self.simple_setup.getProblemDefinition()
        start_state = problem_definition.getStartState(0)  # Assuming one start state
        goal = problem_definition.getGoal()
        goal_state = goal.getState()

        s_x, s_y = start_state.getX(), start_state.getY()
        g_x, g_y = goal_state.getX(), goal_state.getY()

        dist = math.sqrt((s_x - g_x) ** 2 + (s_y - g_y) ** 2)

        dist = np.where(dist == 0, 1e-10, dist)

        tl =  20 * np.log10(dist*1000)
        return tl

        

    def stateCost(self, s: ob.State) -> ob.Cost:
        """
        Calculates the cost for a given SE2 state.

        Args:
            s (ob.State): The state in SE2 space.

        Returns:
            ob.Cost: The cost based on the transmission loss to mammal states.
        """
        x, y = s.getX(), s.getY()


        src_lat, src_lon = self.inverse_projection(x, y)
        ship_loc = np.array([[src_lat, src_lon]])

        ship_loc_repeated = np.repeat(ship_loc, self.mammal_positions_array.shape[0], axis=0)
        # TODO Change this to actual TL
        transmission_loss = self.compute_simple_transmission_loss(self.mammal_positions_array, ship_loc_repeated)

        TL_mean = transmission_loss.mean()

        # Return negative TL_mean because we want to maximize, Adding the approx_max_TL so that the OMPL solver works. If its all negative it won't work
        return ob.Cost(self.normalization(self.approx_max_tl - (TL_mean)))
    
    def compute_simple_transmission_loss(self, mammal_loc, ship_loc):
            """
            Compute the simple transmission loss (TL) using geodesic distance between ship and mammal locations.

            Args:
                mammal_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [mammal_lat, mammal_lon].
                ship_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [ship_lat, ship_lon].
            
            Returns:
                numpy.ndarray: The transmission loss output, shape (N, 1).
            """
            assert mammal_loc.shape[1] == 2, "mammal_loc must have shape (N, 2) with columns [latitude, longitude]"
            assert ship_loc.shape == mammal_loc.shape, "ship_loc must have the same shape as mammal_loc"

            _, _, distances = self.GEODESIC.inv(
                ship_loc[:, 1], ship_loc[:, 0],  # Longitudes and latitudes of ship locations
                mammal_loc[:, 1], mammal_loc[:, 0]  # Longitudes and latitudes of mammal locations
            )

            distances = np.where(distances == 0, 1e-10, distances)

            transmission_loss = 20 * np.log10(distances)
            return transmission_loss.reshape(-1, 1)  # Shape: (N, 1)


    # def compute_tranmission_loss(self, mammal_loc, ship_loc):
    #             """
    #             Compute the model output for transmission loss (TL) given mammal and ship locations.

    #             Args:
    #                     mammal_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [mammal_lat, mammal_lon].
    #                     ship_loc (numpy.ndarray): Array of shape (N, 2), where each row contains [ship_lat, ship_lon].
                
    #             Returns:
    #                     numpy.ndarray: The transmission loss output from the model, shape (N, num_freq)
    #             """
    #             assert mammal_loc.shape[1] == 2, "mammal_loc must have shape (N, 2) with columns [latitude, longitude]"
    #             assert ship_loc.shape == mammal_loc.shape, "ship_loc must have the same shape as mammal_loc"

    #             N = mammal_loc.shape[0]
    #             model_input = np.hstack((ship_loc, mammal_loc))  # Shape: (N, 4)

    #             assert model_input.shape == (N, 4), "model_input must have shape (N, 4)"
    #             model_output = self.TL_model(model_input)  # Output: (N, num_freq)

    #             assert model_output.shape[0] == N, "model_output must have the first dimension equal to N"

    #             # Adding the last column to match frequency dimensions (optional based on your data) to match the freq for 10k Hz
    #             last_column = model_output[:, -1].reshape(-1, 1)
    #             model_output = np.hstack((model_output, last_column))

    #             return model_output  # Shape: (N, num_freq)



def get_ompl_objective(
    space_information,
    simple_setup,
    mammal_positions: List[HelperPosition],
    projector: Proj, GEODESIC: Geod, TL_model_path: str) -> ob.OptimizationObjective:
        # TODO Import mammal states and maybe find the projected traj for objective
        """
        Creates and returns a NoiseObjective for the path planning problem.

        Args:
                space_information: OMPL's space information object.
                simple_setup: OMPL's simple setup object.
                mammal_states (List[HelperPosition]): List of mammal positions in lat,long
                projector: Projection function to convert x, y coordinates to latitude and longitude.

        Returns:
                ob.OptimizationObjective: The noise-based optimization objective.
        """

        # NOTE These are for multi onj
        # objective = ob.MultiOptimizationObjective(si=space_information)
        # objective.addObjective(
        #     objective=NoiseObjective(space_information, simple_setup, mammal_positions, projector, GEODESIC),
        #     weight=1.0,
        # )
        # objective.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))

        objective = NoiseObjective(space_information, simple_setup, mammal_positions, projector, GEODESIC, TL_model_path)
        objective.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
        return objective

def costToGo(s:ob.SE2StateSpace,g:ob.Goal) -> ob.Cost:
        goal_state = g.getState()
        distance = math.sqrt((s.getX() -goal_state.getX()) ** 2 + (s.getY() - goal_state.getY()) ** 2)
        distance_from_origin = math.sqrt((goal_state.getX()) ** 2 + (goal_state.getY()) ** 2)
        return ob.Cost(distance/distance_from_origin)
