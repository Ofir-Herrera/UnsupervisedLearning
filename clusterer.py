from sklearn.cluster import KMeans
import numpy as np

class Clusterer:
    """Applies KMeans clustering to feature vectors extracted from images."""

    def __init__(self, n_clusters=4, random_state=42):
        """
        Initialize the clustering model.

        Args:
            n_clusters (int, optional): Number of clusters to form. Defaults to 4.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

    def prepare_feature_matrix(self, *features):
        """
        Prepare a feature matrix for clustering by flattening and stacking input feature images.

        Args:
            *features: One or more 2D feature arrays.

        Returns:
            tuple:
                - np.ndarray: Flattened and stacked feature matrix (pixels × features).
                - int: Height of the original image.
                - int: Width of the original image.
        """
        h, w = features[0].shape
        flattened = [f.reshape(-1) for f in features]
        return np.column_stack(flattened), h, w

    def cluster(self, feature_matrix):
        """
        Perform KMeans clustering on the provided feature matrix.

        Args:
            feature_matrix (np.ndarray): The feature matrix (pixels × features).

        Returns:
            np.ndarray: Cluster labels for each pixel.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        return kmeans.fit_predict(feature_matrix)
