"""
YOLO model adapter for Ultralytics YOLO.

Provides unified interface for YOLO11 models.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from enterprise_vision_ai.models.base import BaseModel


class YOLOAdapter(BaseModel):
    """Adapter for Ultralytics YOLO models."""

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize YOLO adapter.

        Args:
            model_path: Path to YOLO weights (.pt file)
            config: Configuration dict with optional keys:
                - device: 'cpu', 'cuda', 'mps', or 'auto'
                - task: 'detect', 'segment', 'classify'
                - verbose: bool for verbose output
        """
        super().__init__(model_path, config)
        self.device = self.config.get("device", "auto")
        self.task = self.config.get("task", "segment")
        self.verbose = self.config.get("verbose", False)
        self._class_names = {}

    def load(self) -> None:
        """Load YOLO model from weights file."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics is required. Install with: pip install ultralytics")

        self.model = YOLO(self.model_path)
        self._loaded = True

        # Extract class names
        if hasattr(self.model, "names"):
            self._class_names = self.model.names
        elif hasattr(self.model, "model") and hasattr(self.model.model, "names"):
            self._class_names = self.model.model.names

    def predict(
        self, image: Union[np.ndarray, Image.Image], confidence: float = 0.25, **kwargs
    ) -> Any:
        """
        Run YOLO inference on image.

        Args:
            image: Input image
            confidence: Confidence threshold
            **kwargs: Additional YOLO-specific parameters:
                - iou: IoU threshold for NMS
                - max_det: Maximum detections
                - classes: Filter by class IDs

        Returns:
            YOLO results object
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare parameters
        params = {"conf": confidence, "verbose": self.verbose, **kwargs}

        # Add device if specified
        if self.device != "auto":
            params["device"] = self.device

        # Run inference
        results = self.model(image, **params)

        # Return first result (batch size 1)
        return results[0] if results else None

    def predict_batch(
        self, images: List[Union[np.ndarray, Image.Image]], confidence: float = 0.25, **kwargs
    ) -> List[Any]:
        """
        Run inference on batch of images.

        Args:
            images: List of input images
            confidence: Confidence threshold
            **kwargs: Additional parameters

        Returns:
            List of YOLO results
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        params = {"conf": confidence, "verbose": self.verbose, **kwargs}

        if self.device != "auto":
            params["device"] = self.device

        return self.model(images, **params)

    def export(self, format: str, output_path: str) -> str:
        """
        Export model to different format.

        Args:
            format: Export format (onnx, torchscript, engine, etc.)
            output_path: Path to save exported model

        Returns:
            Path to exported model
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.model.export(format=format, output=output_path)
        return output_path

    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "model_path": self.model_path,
            "task": self.task,
            "device": self.device,
            "loaded": self._loaded,
        }

        if self._class_names:
            info["classes"] = self._class_names
            info["num_classes"] = len(self._class_names)

        # Get model metadata if available
        if hasattr(self.model, "model") and hasattr(self.model.model, "args"):
            args = self.model.model.args
            if hasattr(args, "imgsz"):
                info["input_size"] = args.imgsz

        return info

    def get_class_names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        return self._class_names

    @property
    def class_names(self) -> Dict[int, str]:
        """Alias for get_class_names()."""
        return self._class_names
