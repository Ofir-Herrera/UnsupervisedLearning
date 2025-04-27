import numpy as np

def make_odd(k):
    """
    Ensure that the input integer is odd.

    Args:
        k (int): Input integer.

    Returns:
        int: The input itself if it's odd, otherwise the next odd integer.
    """
    return k if k % 2 == 1 else k + 1

def normalize(img):
    """
    Normalize an image to the [0, 1] range.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image with pixel values scaled between 0 and 1.
    """
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Avoid division by zero
    return img
