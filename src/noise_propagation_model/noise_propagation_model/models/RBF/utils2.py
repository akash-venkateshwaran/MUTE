
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import RBFInterpolator
import pickle
import os
import geopandas as gpd
from shapely.geometry import box
from concurrent.futures import ProcessPoolExecutor



class RBFInterpolatorCustom(RBFInterpolator):
    """
    A custom RBF interpolator that handles vector-valued functions.
    The input `xobs` should be a 2D array where rows represent the number of data points,
    and the columns represent the features (coordinates of the data points).
    The input `yobs` should also be a 2D array where rows represent the number of data points,
    and the columns represent the components of the vector-valued function being interpolated.
    """
    def __init__(self, xobs, yobs, model_path, *args, **kwargs):
        assert yobs.ndim == 2, "yobs must be a 2D array with multiple components (columns)."

        xobs, yobs = self._remove_duplicates(xobs, yobs)

        self.model_path = model_path

        self.num_components = yobs.shape[1]
        self.interpolators = [
            RBFInterpolator(xobs, yobs[:, i], neighbors=75, kernel='gaussian',epsilon=1000.0, *args, **kwargs)
            for i in range(self.num_components)
            if not print(f"Component {i+1}/{self.num_components} is done")
        ]


    def __call__(self, x):
            y = np.column_stack([interp(x) for interp in self.interpolators])
            return y

    
    def _remove_duplicates(self, xobs, yobs, tolerance=1e-5):
        """
        Removes duplicate or near-duplicate points in xobs and the corresponding yobs.
        """
        _, unique_indices = np.unique(np.round(xobs / tolerance), axis=0, return_index=True)
        xobs_unique = xobs[unique_indices]
        yobs_unique = yobs[unique_indices]
        print(f"Removed {len(xobs) - len(unique_indices)} duplicate points.")
        return xobs_unique, yobs_unique

    def export_model(self):
        # Ensure the model directory exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        model_file = os.path.join(self.model_path, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.interpolators, f)
        print(f"Model exported to {model_file}")

    @staticmethod
    def import_model(model_path):
        # Load the RBF interpolators
        model_file = os.path.join(model_path, 'model.pkl')
        with open(model_file, 'rb') as f:
            interpolators = pickle.load(f)
        

        # Create an empty instance and assign the loaded model and scalers
        instance = RBFInterpolatorCustom.__new__(RBFInterpolatorCustom)  # Create an instance without calling __init__
        instance.interpolators = interpolators
        instance.model_path = model_path  # Assign the path for future use

        print(f"Model loaded from {model_path}")
        return instance
    

