"""
Enterprise Vision AI - Utilities Module

Utility functions for image processing, visualization, and metrics.
"""

from .image_utils import load_image, resize_image, preprocess_for_model
from .visualization import (
    draw_annotations,
    draw_severity_indicator,
    get_defect_colors,
    get_ore_class_colors,
    get_severity_color,
)
from .metrics import (
    calculate_anomaly_score,
    get_severity_level,
    get_maintenance_recommendation,
    calculate_ore_metrics,
    calculate_metal_ratio,
    get_diverter_recommendation,
    create_metrics_dataframe,
    create_ore_dataframe,
)
from .io_utils import format_time, create_dummy_results

__all__ = [
    # Image utils
    "load_image",
    "resize_image",
    "preprocess_for_model",
    # Visualization
    "draw_annotations",
    "draw_severity_indicator",
    "get_defect_colors",
    "get_ore_class_colors",
    "get_severity_color",
    # Metrics
    "calculate_anomaly_score",
    "get_severity_level",
    "get_maintenance_recommendation",
    "calculate_ore_metrics",
    "calculate_metal_ratio",
    "get_diverter_recommendation",
    "create_metrics_dataframe",
    "create_ore_dataframe",
    # IO Utils
    "format_time",
    "create_dummy_results",
]
