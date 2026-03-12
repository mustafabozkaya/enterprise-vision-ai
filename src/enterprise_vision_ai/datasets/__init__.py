"""
Enterprise Vision AI - Datasets Package

This package provides dataset utilities and data loaders for the
Enterprise Vision AI project.

Submodules:
- data_loader: Data loading and preprocessing utilities
"""

from enterprise_vision_ai.datasets.data_loader import (
    DefectDetectionDataset,
    OreClassificationDataset,
    load_dataset,
)

__all__ = [
    "DefectDetectionDataset",
    "OreClassificationDataset",
    "load_dataset",
]
