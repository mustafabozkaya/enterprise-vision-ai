"""
Image utility functions for Enterprise Vision AI.
"""

from typing import Optional

import cv2
import numpy as np
from PIL import Image


def load_image(file) -> Optional[np.ndarray]:
    """
    Load image from file.

    Args:
        file: File object or path

    Returns:
        np.ndarray: Image array or None
    """
    try:
        image = Image.open(file)
        image = np.array(image)

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image
    except Exception as e:
        print(f"Image loading error: {e}")
        return None


def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Resize image to maximum size while maintaining aspect ratio.

    Args:
        image: Input image
        max_size: Maximum edge length

    Returns:
        np.ndarray: Resized image
    """
    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model inference.

    Args:
        image: Input image

    Returns:
        np.ndarray: Preprocessed image
    """
    # Resize if needed
    image = resize_image(image)

    return image
