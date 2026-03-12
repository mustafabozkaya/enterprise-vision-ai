"""
Model loader for loading models from various sources.

Supports:
- Local file system
- HuggingFace Hub
- S3/MinIO storage
- MLflow registry
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union

from enterprise_vision_ai.models.base import BaseModel
from enterprise_vision_ai.models.defect_detector import DefectDetector
from enterprise_vision_ai.models.ore_classifier import OreClassifier
from enterprise_vision_ai.models.yolo_adapter import YOLOAdapter


class ModelLoader:
    """Load models from various sources."""

    # Model type mapping
    MODEL_REGISTRY = {
        "defect": DefectDetector,
        "defect_detection": DefectDetector,
        "ore": OreClassifier,
        "ore_classification": OreClassifier,
        "yolo": YOLOAdapter,
        "yolo11": YOLOAdapter,
        "yolo11s": YOLOAdapter,
        "yolo11n": YOLOAdapter,
        "yolo11m": YOLOAdapter,
        "yolo11l": YOLOAdapter,
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        model_path: Union[str, Path],
        model_type: str = "auto",
        config: Optional[Dict] = None,
    ) -> BaseModel:
        """
        Load a model from path.

        Args:
            model_path: Path to model file or identifier
            model_type: Type of model (auto-detect if not specified)
            config: Additional configuration

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model type is not supported
            FileNotFoundError: If model file not found
        """
        model_path = Path(model_path)
        config = config or {}

        # Auto-detect model type from filename
        if model_type == "auto":
            model_type = self._detect_model_type(model_path.name)

        # Get model class
        model_class = self.MODEL_REGISTRY.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load model
        model = model_class(str(model_path), config)
        model.load()

        return model

    def load_from_hf(
        self,
        repo_id: str,
        filename: str = "best.pt",
        model_type: str = "auto",
        config: Optional[Dict] = None,
    ) -> BaseModel:
        """
        Load model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID
            filename: Model filename in repository
            model_type: Type of model
            config: Additional configuration

        Returns:
            Loaded model instance
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface-hub")

        # Download model to cache
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(self.cache_dir),
        )

        return self.load(model_path, model_type, config)

    def load_from_s3(
        self,
        bucket: str,
        key: str,
        model_type: str = "auto",
        config: Optional[Dict] = None,
    ) -> BaseModel:
        """
        Load model from S3/MinIO storage.

        Args:
            bucket: S3 bucket name
            key: S3 object key
            model_type: Type of model
            config: Additional configuration

        Returns:
            Loaded model instance
        """
        # TODO: Implement S3 loading
        raise NotImplementedError("S3 loading not yet implemented")

    def list_available_models(self) -> Dict[str, str]:
        """
        List available model types.

        Returns:
            Dictionary of model types and descriptions
        """
        return {
            "defect": "Defect detection model (YOLO-based)",
            "defect_detection": "Alias for defect",
            "ore": "Ore classification model (YOLO-based)",
            "ore_classification": "Alias for ore",
            "yolo": "Generic YOLO model adapter",
        }

    def _detect_model_type(self, filename: str) -> str:
        """
        Auto-detect model type from filename.

        Args:
            filename: Model filename

        Returns:
            Detected model type
        """
        filename_lower = filename.lower()

        # Check for defect detection keywords
        if any(kw in filename_lower for kw in ["defect", "defek", "kusur", "çatlak", "çizik"]):
            return "defect"

        # Check for ore classification keywords
        if any(kw in filename_lower for kw in ["ore", "cevher", "manyetit", "krom", "maden"]):
            return "ore"

        # Default to YOLO adapter
        if any(kw in filename_lower for kw in ["yolo", ".pt"]):
            return "yolo"

        return "yolo"
