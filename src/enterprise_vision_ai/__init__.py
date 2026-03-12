"""
Enterprise Vision AI - Core Package

This package provides the core functionality for the Enterprise Vision AI project,
including model management, data loading, and utilities for industrial AI applications.

Example usage:
    from enterprise_vision_ai import load_model, load_dataset

    # Load a defect detection model
    model = load_model("yolov8n-defect")

    # Load a dataset
    dataset = load_dataset("defect_detection", split="train")
"""

__version__ = "2.0.0"
__author__ = "Enterprise Vision AI Team"

from enterprise_vision_ai.datasets.data_loader import (
    DefectDetectionDataset,
    OreClassificationDataset,
    load_dataset,
)
from enterprise_vision_ai.models.model_manager import ModelManager, load_model

__all__ = [
    "__version__",
    "__author__",
    "ModelManager",
    "load_model",
    "DefectDetectionDataset",
    "OreClassificationDataset",
    "load_dataset",
]
