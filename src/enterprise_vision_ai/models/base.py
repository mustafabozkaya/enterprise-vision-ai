"""
Base model interface for Enterprise Vision AI.

All model implementations must inherit from BaseModel.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image


class BaseModel(ABC):
    """Abstract base class for all AI models."""

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize base model.

        Args:
            model_path: Path to model weights
            config: Model configuration dictionary
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""
        pass

    @abstractmethod
    def predict(
        self, image: Union[np.ndarray, Image.Image], confidence: float = 0.25, **kwargs
    ) -> List[Dict]:
        """
        Run inference on single image.

        Args:
            image: Input image
            confidence: Confidence threshold
            **kwargs: Additional model-specific parameters

        Returns:
            List of detection results
        """
        pass

    @abstractmethod
    def predict_batch(
        self, images: List[Union[np.ndarray, Image.Image]], confidence: float = 0.25, **kwargs
    ) -> List[List[Dict]]:
        """
        Run inference on batch of images.

        Args:
            images: List of input images
            confidence: Confidence threshold
            **kwargs: Additional model-specific parameters

        Returns:
            List of detection results for each image
        """
        pass

    @abstractmethod
    def export(self, format: str, output_path: str) -> str:
        """
        Export model to different format.

        Args:
            format: Export format (onnx, torchscript, etc.)
            output_path: Path to save exported model

        Returns:
            Path to exported model
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload(self) -> None:
        """Unload model from memory."""
        self.model = None
        self._loaded = False
        import gc

        gc.collect()
