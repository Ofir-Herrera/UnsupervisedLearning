import cv2
import numpy as np
from utils import make_odd

class FeatureExtractor:
    def __init__(self, kernel_size_1, kernel_size_2):
        self.kernel_size_1 = make_odd(kernel_size_1)
        self.kernel_size_2 = make_odd(kernel_size_2)

    def normalize_image(self, image):
        return image / 255.0

    def blur_difference(self, image):
        blur1 = cv2.GaussianBlur(image, (self.kernel_size_1, self.kernel_size_1), 0).astype(np.float32)
        blur2 = cv2.GaussianBlur(image, (self.kernel_size_2, self.kernel_size_2), 0).astype(np.float32)
        return cv2.subtract(blur1, blur2)

    def extract_features(self, image):
        normalized = self.normalize_image(image)
        blur_diff = self.blur_difference(normalized)
        # Apply Median Blur (good for salt & pepper noise)
        median_filtered = cv2.medianBlur(blur_diff, 5)

        laplacian = cv2.Laplacian(median_filtered, cv2.CV_32F)
        edges = cv2.Canny((median_filtered * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0  # Normalize back
        # Load the already edge-detected image (Canny output)
        edges_8bit = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Find contours in the edge image
        contours, _ = cv2.findContours(edges_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Filter contours based on their length (perimeter)
        min_length = 150  # Change this value as needed
        filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > min_length]

        # Create an empty image to draw the filtered edges
        filtered_edges = np.zeros_like(edges_8bit)

        # Draw only the long contours
        cv2.drawContours(filtered_edges, filtered_contours, -1, (255), thickness=1)


        return normalized, blur_diff, laplacian, filtered_edges
