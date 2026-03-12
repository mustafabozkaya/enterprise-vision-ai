"""
Inference engine for orchestrating model inference.

Coordinates:
- Model loading
- Preprocessing
- Inference
- Postprocessing
- Caching
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from enterprise_vision_ai.core.model_loader import ModelLoader
from enterprise_vision_ai.core.postprocessor import DetectionResult, Postprocessor
from enterprise_vision_ai.core.preprocessor import Preprocessor


class InferenceEngine:
    """Main inference engine for running models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "auto",
        preprocessor: Optional[Preprocessor] = None,
        postprocessor: Optional[Postprocessor] = None,
        cache_enabled: bool = False,
        device: str = "auto",
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model file
            model_type: Type of model
            preprocessor: Custom preprocessor instance
            postprocessor: Custom postprocessor instance
            cache_enabled: Whether to enable result caching
            device: Device to run inference on (auto, cpu, cuda, mps)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.preprocessor = preprocessor or Preprocessor()
        self.postprocessor = postprocessor or Postprocessor()
        self.cache_enabled = cache_enabled
        self.device = device

        self.model = None
        self.model_loader = ModelLoader()
        self._cache = {} if cache_enabled else None

        # Statistics
        self._inference_count = 0
        self._total_time = 0.0

    def load_model(self, model_path: Optional[str] = None, **kwargs) -> None:
        """
        Load model into engine.

        Args:
            model_path: Path to model (uses init path if not provided)
            **kwargs: Additional arguments for model loading
        """
        model_path = model_path or self.model_path
        if not model_path:
            raise ValueError("Model path must be provided")

        self.model = self.model_loader.load(
            model_path,
            model_type=self.model_type,
            config={"device": self.device, **kwargs}
        )

    def run(
        self,
        image: Union[np.ndarray, Image.Image, str],
        confidence: Optional[float] = None,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on single image.

        Args:
            image: Input image
            confidence: Confidence threshold (uses postprocessor default if None)
            return_raw: Whether to include raw model output

        Returns:
            Dictionary with results:
                - detections: List of DetectionResult
                - metrics: Aggregate metrics
                - severity: Severity level
                - timing: Inference timing info
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Preprocess
        prep_start = time.time()
        processed_image = self.preprocessor.process(image)
        prep_time = time.time() - prep_start

        # Run inference
        infer_start = time.time()
        raw_results = self.model.predict(
            processed_image,
            confidence=confidence or self.postprocessor.confidence_threshold
        )
        infer_time = time.time() - infer_start

        # Postprocess
        post_start = time.time()
        detections = self.postprocessor.process(raw_results)
        metrics = self.postprocessor.calculate_metrics(detections)
        severity = self.postprocessor.calculate_severity(detections)
        post_time = time.time() - post_start

        total_time = time.time() - start_time

        # Update statistics
        self._inference_count += 1
        self._total_time += total_time

        result = {
            "detections": detections,
            "metrics": metrics,
            "severity": severity,
            "timing": {
                "preprocessing": prep_time,
                "inference": infer_time,
                "postprocessing": post_time,
                "total": total_time,
            },
        }

        if return_raw:
            result["raw_output"] = raw_results

        return result

    def run_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on batch of images.

        Args:
            images: List of input images
            confidence: Confidence threshold

        Returns:
            List of result dictionaries
        """
        return [
            self.run(img, confidence)
            for img in images
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with statistics
        """
        avg_time = self._total_time / self._inference_count if self._inference_count > 0 else 0

        return {
            "inference_count": self._inference_count,
            "total_time": self._total_time,
            "avg_time": avg_time,
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
        }

    def clear_cache(self) -> None:
        """Clear result cache."""
        if self._cache:
            self._cache.clear()

    def unload(self) -> None:
        """Unload model and clear resources."""
        if self.model:
            self.model.unload()
            self.model = None
        self.clear_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False
