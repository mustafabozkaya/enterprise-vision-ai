"""
Enterprise Vision AI MVP - Ortak Fonksiyonlar
Defekt Tespiti ve Cevher Ön Seçimi için yardımcı fonksiyonlar

NOTE: This module is deprecated. Use enterprise_vision_ai.utils instead.
"""

import warnings

# Show deprecation warning
warnings.warn(
    "services.utils is deprecated. Use enterprise_vision_ai.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new modular structure
from enterprise_vision_ai.utils.image_utils import (
    load_image,
    preprocess_for_model,
    resize_image,
)
from enterprise_vision_ai.utils.metrics import (
    calculate_anomaly_score,
    calculate_metal_ratio,
    calculate_ore_metrics,
    create_metrics_dataframe,
    create_ore_dataframe,
    get_diverter_recommendation,
    get_maintenance_recommendation,
    get_severity_level,
)
from enterprise_vision_ai.utils.visualization import (
    draw_annotations,
    draw_severity_indicator,
    get_defect_colors,
    get_ore_class_colors,
    get_severity_color,
)

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
]
