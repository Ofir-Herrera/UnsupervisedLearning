import cv2
import numpy as np


class ImageLoader:
    """Loads grayscale images as float32 arrays for further processing."""

    def __init__(self, image_paths):
        """
        Initialize the loader with a list of image paths.

        Args:
            image_paths (list or str): List of image file paths or a single image path.
        """
        self.image_paths = image_paths

    def load_image(self, index=0):
        """
        Load an image by index from the provided paths.

        Args:
            index (int, optional): Index of the image to load. Defaults to 0.

        Returns:
            np.ndarray: Loaded grayscale image as a float32 array.

        Raises:
            IndexError: If the given index is out of bounds.
            FileNotFoundError: If the image file could not be loaded.
        """
        if index < 0 or index >= len(self.image_paths):
            raise IndexError("Image index out of range.")

        image = cv2.imread(self.image_paths, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_paths[index]}")

        return image
