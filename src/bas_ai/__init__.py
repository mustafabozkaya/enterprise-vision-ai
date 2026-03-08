"""
Enterprise Vision AI - Core Package (Backward Compatibility Alias)

This module provides backward compatibility for the legacy 'bas_ai' package name.
All functionality is now available via 'enterprise_vision_ai'.

For new code, please use:
    from enterprise_vision_ai import load_model, load_dataset

For backward compatibility, you can still use:
    from bas_ai import load_model, load_dataset

Example usage:
    from bas_ai import load_model, load_dataset
    
    # Load a defect detection model
    model = load_model("yolov8n-defect")
    
    # Load a dataset
    dataset = load_dataset("defect_detection", split="train")
"""

# Backward compatibility: import everything from enterprise_vision_ai
# This allows both 'from bas_ai import ...' and 'from enterprise_vision_ai import ...'
import enterprise_vision_ai as _evai

# Re-export all public API
__version__ = _evai.__version__
__author__ = _evai.__author__

from enterprise_vision_ai import (
    ModelManager,
    load_model,
    DefectDetectionDataset,
    OreClassificationDataset,
    load_dataset,
)

__all__ = [
    # Version
    "__version__",
    # Model management
    "ModelManager",
    "load_model",
    # Dataset utilities
    "DefectDetectionDataset",
    "OreClassificationDataset",
    "load_dataset",
]
