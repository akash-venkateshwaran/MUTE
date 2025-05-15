
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import RBFInterpolator
import pickle
import os
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import IncrementalPCA


from sklearn.decomposition import IncrementalPCA

class RBFInterpolatorCustom(RBFInterpolator):
    """
    A custom RBF interpolator that handles vector-valued functions.
    The input `xobs` should be a 2D array where rows represent the number of data points,
    and the columns represent the features (coordinates of the data points).
    The input `yobs` should also be a 2D array where rows represent the number of data points,
    and the columns represent the components of the vector-valued function being interpolated.
    """
    def __init__(self, xobs, yobs, model_path, n_components, *args, **kwargs):
        assert yobs.ndim == 2, "yobs must be a 2D array with multiple components (columns)."

        xobs, yobs = self._remove_duplicates(xobs, yobs)

        self.model_path = model_path

        # Check if PCA should be applied
        if n_components >= yobs.shape[1]:
            print("Number of components is greater than or equal to the number of features. Skipping PCA.")
            yobs_transformed = yobs
            self.ipca = None
            self.yobs_mean = None
        else:
            yobs_transformed = self._create_PCA(yobs, n_components)

        self.num_components = yobs_transformed.shape[1]
        self.interpolators = [
            RBFInterpolator(xobs, yobs_transformed[:, i],neighbors=75, kernel='gaussian',epsilon=1000.0, *args, **kwargs)
            for i in range(self.num_components)
            if not print(f"Component {i+1}/{self.num_components} is done")
        ]

    def _create_PCA(self, yobs, n_components):
        self.yobs_mean = yobs.mean(axis=0)
        yobs_centered = yobs - self.yobs_mean
        self.ipca = IncrementalPCA(n_components=n_components)

        batch_size = 10000  # Adjust the batch size based on your memory constraints
        n_samples = yobs_centered.shape[0]

        # Fit the IncrementalPCA in batches
        for i in range(0, n_samples, batch_size):
            batch = yobs_centered[i:i+batch_size]
            self.ipca.partial_fit(batch)

        # Now, transform the data in batches
        yobs_transformed_batches = []
        for i in range(0, n_samples, batch_size):
            batch = yobs_centered[i:i+batch_size]
            yobs_transformed_batches.append(self.ipca.transform(batch))

        yobs_transformed = np.vstack(yobs_transformed_batches)
        return yobs_transformed

    def __call__(self, x):
        y_components = []
        for i, interp in enumerate(self.interpolators):
            y_comp = interp(x)
            y_components.append(y_comp)
        y_transformed = np.column_stack(y_components)
        if self.ipca is not None:
            # Inverse transform in batches if necessary
            batch_size = 10000  # Adjust based on your memory constraints
            n_samples = y_transformed.shape[0]
            y_centered_batches = []
            for i in range(0, n_samples, batch_size):
                batch = y_transformed[i:i+batch_size]
                y_centered_batches.append(self.ipca.inverse_transform(batch))
            y_centered = np.vstack(y_centered_batches)
            y = y_centered + self.yobs_mean
        else:
            y = y_transformed
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

        # Export the interpolators
        model_file = os.path.join(self.model_path, 'interpolators.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.interpolators, f)
        print(f"Interpolators exported to {model_file}")

        # Export the IncrementalPCA model only if it exists
        if self.ipca is not None:
            ipca_file = os.path.join(self.model_path, 'ipca.pkl')
            with open(ipca_file, 'wb') as f:
                pickle.dump(self.ipca, f)
            print(f"IncrementalPCA model exported to {ipca_file}")

            # Export the yobs_mean
            mean_file = os.path.join(self.model_path, 'yobs_mean.pkl')
            with open(mean_file, 'wb') as f:
                pickle.dump(self.yobs_mean, f)
            print(f"yobs_mean exported to {mean_file}")

    @staticmethod
    def import_model(model_path):
        # Load the interpolators
        model_file = os.path.join(model_path, 'interpolators.pkl')
        with open(model_file, 'rb') as f:
            interpolators = pickle.load(f)

        # Load the IncrementalPCA model if it exists
        ipca_file = os.path.join(model_path, 'ipca.pkl')
        if os.path.exists(ipca_file):
            with open(ipca_file, 'rb') as f:
                ipca = pickle.load(f)

            # Load the yobs_mean
            mean_file = os.path.join(model_path, 'yobs_mean.pkl')
            with open(mean_file, 'rb') as f:
                yobs_mean = pickle.load(f)
        else:
            ipca = None
            yobs_mean = None

        # Create an empty instance and assign the loaded models
        instance = RBFInterpolatorCustom.__new__(RBFInterpolatorCustom)
        instance.interpolators = interpolators
        instance.num_components = len(interpolators)
        instance.ipca = ipca
        instance.yobs_mean = yobs_mean
        instance.model_path = model_path

        print(f"Model loaded from {model_path}")
        return instance

