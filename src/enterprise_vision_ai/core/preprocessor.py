"""
Image preprocessing module.

Handles:
- Image loading and decoding
- Resizing and normalization
- Color space conversion
- Data augmentation
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


class Preprocessor:
    """Image preprocessing pipeline."""

    # Default preprocessing parameters
    DEFAULT_INPUT_SIZE = 640
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        input_size: int = DEFAULT_INPUT_SIZE,
        normalize: bool = True,
        mean: Optional[list] = None,
        std: Optional[list] = None,
    ):
        """
        Initialize preprocessor.

        Args:
            input_size: Target image size (square)
            normalize: Whether to normalize pixel values
            mean: Normalization mean values
            std: Normalization std values
        """
        self.input_size = input_size
        self.normalize = normalize
        self.mean = np.array(mean or self.DEFAULT_MEAN)
        self.std = np.array(std or self.DEFAULT_STD)

    def process(
        self,
        image: Union[np.ndarray, Image.Image, str],
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image for inference.

        Args:
            image: Input image (numpy array, PIL Image, or path)
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Preprocessed image as numpy array
        """
        # Load image if path provided
        if isinstance(image, str):
            image = self._load_image(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        if maintain_aspect_ratio:
            image = self._resize_maintain_aspect(image)
        else:
            image = cv2.resize(image, (self.input_size, self.input_size))

        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std

        return image

    def process_batch(
        self,
        images: list,
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """
        Preprocess batch of images.

        Args:
            images: List of input images
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Batch of preprocessed images
        """
        processed = [
            self.process(img, maintain_aspect_ratio)
            for img in images
        ]
        return np.stack(processed, axis=0)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not load image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _resize_maintain_aspect(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding."""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)

        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image
        padded = np.full(
            (self.input_size, self.input_size, 3),
            114,  # Gray padding
            dtype=image.dtype
        )

        # Center the resized image
        y_offset = (self.input_size - new_h) // 2
        x_offset = (self.input_size - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        Reverse normalization for visualization.

        Args:
            image: Normalized image

        Returns:
            Denormalized image (0-255 range)
        """
        if not self.normalize:
            return image

        image = image * self.std + self.mean
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return image


class VideoPreprocessor:
    """Preprocessor for video streams."""

    def __init__(self, preprocessor: Optional[Preprocessor] = None, **kwargs):
        """
        Initialize video preprocessor.

        Args:
            preprocessor: Image preprocessor instance
            **kwargs: Arguments for creating Preprocessor
        """
        self.preprocessor = preprocessor or Preprocessor(**kwargs)

    def process_frame(
        self,
        frame: np.ndarray,
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """
        Preprocess single video frame.

        Args:
            frame: Input frame (BGR format from OpenCV)
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Preprocessed frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.preprocessor.process(frame_rgb, maintain_aspect_ratio)

    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
    ) -> list:
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to extract

        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                frames.append(frame)

                if max_frames and len(frames) >= max_frames:
                    break

            frame_count += 1

        cap.release()
        return frames
