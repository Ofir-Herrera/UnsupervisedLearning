import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    """Provides visualization and saving tools for images, features, and clustering results."""

    def __init__(self, output_dir="plots"):
        """
        Initialize the visualizer.

        Args:
            output_dir (str, optional): Directory where plots will be saved. Defaults to "plots".
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_image(self, image, title, filename, cmap="gray"):
        """Save a single grayscale image with a title."""
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
        plt.title(title)
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def save_multiple_images(self, images, titles, filename_prefix, cmap="gray"):
        """Save multiple images separately, each with its own title."""
        for i, (img, title) in enumerate(zip(images, titles)):
            self.save_image(img, title, f"{filename_prefix}_{i}.png", cmap=cmap)

    def save_histogram(self, image, filename="histogram.png"):
        """Save the histogram of pixel intensities of an image."""
        min_original, max_original = image.min(), image.max()
        plt.figure(figsize=(6, 4))
        plt.hist(image.flatten(), bins=50, color='black', alpha=0.7, range=(min_original, max_original))
        plt.title("Histogram of Image Intensities")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def save_clustered_image(self, labels, h, w, filename="clusters.png"):
        """Save an image showing cluster assignments as colors."""
        clustered_img = labels.reshape((h, w))
        plt.figure(figsize=(6, 6))
        plt.imshow(clustered_img, cmap='tab10')
        plt.axis('off')
        plt.title("Clustered Image (KMeans Labels)")
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def save_clustered_tif(self, labels, h, w, filename="clusters.tif"):
        """Save the clustered image as a TIFF file."""
        clustered_img = labels.reshape((h, w))
        normalized_img = (255 * (clustered_img / (np.max(clustered_img) + 1e-8))).astype(np.uint8)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, normalized_img)

    def plot_3d_features(self, features, labels, filename="3d_clusters.png"):
        """Create and save a 3D scatter plot of feature vectors colored by cluster."""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2],
                             c=labels, cmap='tab10', s=1)

        ax.set_xlabel('Edges')
        ax.set_ylabel('Blur Diff')
        ax.set_zlabel('Laplacian')
        ax.set_title("3D Feature Clusters", pad=20)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Cluster ID')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def plot_feature_correlation(self, feature_matrix, feature_names=None, filename="feature_correlation.png"):
        """
        Plot and save a heatmap showing correlation between different feature vectors.

        Args:
            feature_matrix (np.ndarray): Feature matrix (pixels × features).
            feature_names (list, optional): List of feature names. Defaults to automatic naming.
            filename (str, optional): Output file name.
        """
        import pandas as pd
        import seaborn as sns

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(feature_matrix.shape[1])]

        df = pd.DataFrame(feature_matrix, columns=feature_names)
        corr_matrix = df.corr()

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title("Feature Correlation Matrix")

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def save_cluster_masks(self, labels, h, w, original_image):
        """
        Save separate images showing only pixels belonging to each cluster.

        Args:
            labels (np.ndarray): Flattened cluster labels.
            h (int): Image height.
            w (int): Image width.
            original_image (np.ndarray): Grayscale source image.
        """
        clustered_img = labels.reshape((h, w))
        unique_labels = np.unique(clustered_img)
        original_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        import matplotlib.pyplot as plt
        colormap = plt.cm.get_cmap("tab10", len(unique_labels))

        for label in unique_labels:
            rgb = np.array(colormap(label)[:3]) * 255
            bgr = tuple(int(x) for x in rgb[::-1])

            mask = (clustered_img == label).astype(np.uint8) * 255
            overlay = np.zeros_like(original_color)
            overlay[:, :] = bgr
            colored_cluster = cv2.bitwise_and(overlay, overlay, mask=mask)

            filename = f"cluster_overlay_{label}.png"
            cv2.imwrite(os.path.join(self.output_dir, filename), colored_cluster)

    def plot_silhouette_vs_k(self, silhouette_scores, k_range, filename="k_vs_silhouette.png"):
        """
        Plot and save silhouette scores as a function of the number of clusters K.

        Args:
            silhouette_scores (list): Silhouette scores for different K values.
            k_range (list): List of tested K values.
            filename (str, optional): Output file name.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, silhouette_scores, marker='o', color='blue')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs K")
        plt.grid(True)

        best_k = k_range[int(np.argmax(silhouette_scores))]
        best_score = max(silhouette_scores)
        plt.axvline(x=best_k, color='red', linestyle='--', label=f"Best K = {best_k}")
        plt.legend()

        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()
