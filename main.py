import os.path
import numpy as np
import cv2
from config import (
    IMAGE, KERNEL_SIZE_1, KERNEL_SIZE_2,
    N_CLUSTERS, RANDOM_STATE, IMAGE_INDEX
)
from utils import normalize
from image_loader import ImageLoader
from feature_extractor import FeatureExtractor
from clusterer import Clusterer
from visualizer import Visualizer
from statistical_analyzer import StatisticalAnalyzer

def run_pipeline():
    """
    Full segmentation pipeline:
    - Load image
    - Extract handcrafted features
    - Apply KMeans clustering
    - Save visualizations
    - Perform statistical analysis to select optimal K
    """

    # Load the image
    IMAGE_PATH = os.path.join(os.getcwd(), "images", IMAGE[0])
    loader = ImageLoader(IMAGE_PATH)
    image = loader.load_image(index=IMAGE_INDEX)

    # Initialize visualizer
    visualizer = Visualizer()
    visualizer.save_image(image, "Original Image", "original.png")
    visualizer.save_histogram(image)

    # Feature extraction
    extractor = FeatureExtractor(KERNEL_SIZE_1, KERNEL_SIZE_2)
    norm, blur_diff, laplacian, edges = extractor.extract_features(image)

    # Normalize features
    blur_diff = normalize(blur_diff)
    laplacian = normalize(laplacian)
    edges = normalize(edges)

    # Save feature images
    visualizer.save_multiple_images(
        [norm, blur_diff, laplacian, edges],
        ["Normalized", "Blur Difference", "Laplacian", "Canny Edges"],
        "features"
    )

    # Clustering
    clusterer = Clusterer(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    feature_matrix, h, w = clusterer.prepare_feature_matrix(blur_diff, laplacian, edges)
    labels = clusterer.cluster(feature_matrix)

    # Visualizations
    visualizer.plot_3d_features(feature_matrix, labels)
    visualizer.save_cluster_masks(labels, h, w, image)

    # Feature correlation heatmap
    feature_names = ["blur_diff", "laplacian", "edges"]
    visualizer.plot_feature_correlation(feature_matrix, feature_names)

    # --- Silhouette analysis on resized feature images ---
    # Downsample features to reduce computational cost
    resize_shape = (256, 256)
    blur_diff_small = cv2.resize(blur_diff, resize_shape, interpolation=cv2.INTER_AREA)
    laplacian_small = cv2.resize(laplacian, resize_shape, interpolation=cv2.INTER_AREA)
    edges_small = cv2.resize(edges, resize_shape, interpolation=cv2.INTER_AREA)

    X, Y = np.meshgrid(np.arange(resize_shape[1]), np.arange(resize_shape[0]))
    feature_matrix_small = np.column_stack((
        blur_diff_small.flatten(),
        laplacian_small.flatten(),
        edges_small.flatten()
    ))

    # Statistical evaluation
    analyzer = StatisticalAnalyzer(random_state=RANDOM_STATE)
    k_range = list(range(2, 11))
    silhouette_scores = analyzer.evaluate_k_range(feature_matrix_small, k_range=k_range)
    visualizer.plot_silhouette_vs_k(silhouette_scores, k_range)

if __name__ == "__main__":
    run_pipeline()
