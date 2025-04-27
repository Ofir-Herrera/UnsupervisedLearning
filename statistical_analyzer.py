import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class StatisticalAnalyzer:
    """Performs statistical analysis to evaluate clustering quality over different K values."""

    def __init__(self, random_state=42):
        """
        Initialize the analyzer.

        Args:
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.random_state = random_state

    @staticmethod
    def evaluate_k_range(features, k_range=list(range(2, 11)), random_state=42):
        """
        Evaluate silhouette scores for a range of K values using KMeans clustering.

        Args:
            features (np.ndarray): Feature matrix (pixels × features).
            k_range (list, optional): List of K values to evaluate. Defaults to range(2, 11).
            random_state (int, optional): Random seed for KMeans. Defaults to 42.

        Returns:
            list: Silhouette scores corresponding to each K value.
        """
        silhouette_scores = []
        print(k_range)  # Debugging: Print the K values being tested
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
        print(silhouette_scores)  # Debugging: Print the silhouette scores
        return silhouette_scores
