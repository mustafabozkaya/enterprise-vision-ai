"""
Defect detection model.

Specialized model for industrial surface defect detection.
"""

from typing import Dict, List, Optional, Any

from enterprise_vision_ai.models.yolo_adapter import YOLOAdapter


class DefectDetector(YOLOAdapter):
    """Defect detection model for surface quality inspection."""

    # Default defect class names (Turkish)
    DEFAULT_CLASSES = {
        0: "çatlak",
        1: "çizik",
        2: "delik",
        3: "leke",
        4: "deformasyon",
    }

    # Class descriptions
    CLASS_DESCRIPTIONS = {
        "çatlak": "Yüzey çatlağı - kritik yapısal hasar",
        "çizik": "Yüzey çizigi - hafif kozmetik hasar",
        "delik": "Delik/patlatma - malzeme kaybı",
        "leke": "Renk lekesi - kontaminasyon",
        "deformasyon": "Şekil bozukluğu - fiziksel hasar",
    }

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize defect detector.

        Args:
            model_path: Path to YOLO weights
            config: Configuration dict with keys:
                - custom_classes: Override default class names
                - confidence_threshold: Detection threshold
                - severity_thresholds: (low, medium, high) tuple
        """
        super().__init__(model_path, config)

        # Set default task to segmentation
        self.task = "segment"

        # Override class names if provided
        if config and "custom_classes" in config:
            self._class_names = config["custom_classes"]
        else:
            self._class_names = self.DEFAULT_CLASSES.copy()

        self.confidence_threshold = config.get("confidence_threshold", 0.25)
        self.severity_thresholds = config.get("severity_thresholds", (0.3, 0.7))

    def get_info(self) -> Dict[str, Any]:
        """Get defect detector information."""
        info = super().get_info()
        info.update({
            "type": "defect_detector",
            "application": "surface_defect_detection",
            "severity_thresholds": self.severity_thresholds,
            "class_descriptions": self.CLASS_DESCRIPTIONS,
        })
        return info

    def get_class_description(self, class_name: str) -> str:
        """
        Get description for defect class.

        Args:
            class_name: Class name (e.g., 'çatlak')

        Returns:
            Description string
        """
        return self.CLASS_DESCRIPTIONS.get(class_name, "Unknown defect type")

    def calculate_defect_metrics(self, results: List[Any]) -> Dict[str, Any]:
        """
        Calculate defect-specific metrics from results.

        Args:
            results: Detection results

        Returns:
            Dictionary with defect metrics
        """
        if not results:
            return {
                "total_defects": 0,
                "defects_by_type": {},
                "severity_score": 0.0,
                "critical_defects": 0,
            }

        defects_by_type = {}
        critical_count = 0
        total_confidence = 0.0

        for det in results:
            class_name = getattr(det, 'class_name', 'unknown')
            confidence = getattr(det, 'confidence', 0)

            defects_by_type[class_name] = defects_by_type.get(class_name, 0) + 1
            total_confidence += confidence

            # Count critical defects (cracks and holes)
            if class_name in ["çatlak", "delik"]:
                critical_count += 1

        avg_confidence = total_confidence / len(results) if results else 0

        # Calculate severity score (0-100)
        severity_score = avg_confidence * 100

        return {
            "total_defects": len(results),
            "defects_by_type": defects_by_type,
            "severity_score": severity_score,
            "critical_defects": critical_count,
        }
