"""
Ore classification model.

Specialized model for mineral ore classification.
"""

from typing import Any, Dict, List, Optional

from enterprise_vision_ai.models.yolo_adapter import YOLOAdapter


class OreClassifier(YOLOAdapter):
    """Ore classification model for mineral identification."""

    # Default ore class names (Turkish)
    DEFAULT_CLASSES = {
        0: "manyetit",
        1: "krom",
        2: "atık",
        3: "düşük tenör",
    }

    # Ore type descriptions
    ORE_DESCRIPTIONS = {
        "manyetit": "Manyetit (Fe3O4) - Manyetik demir oksit",
        "krom": "Kromit (FeCr2O4) - Yüksek değerli krom cevheri",
        "atık": "Atık/Gang - Değersiz kayaç",
        "düşük tenör": "Düşük tenörlü - Düşük mineral içeriği",
    }

    # Typical metal ratios (for estimation)
    DEFAULT_METAL_RATIOS = {
        "manyetit": 0.72,  # 72% Fe
        "krom": 0.46,  # 46% Cr
        "atık": 0.0,
        "düşük tenör": 0.15,
    }

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize ore classifier.

        Args:
            model_path: Path to YOLO weights
            config: Configuration dict with keys:
                - custom_classes: Override default class names
                - metal_ratios: Custom metal content ratios
                - confidence_threshold: Classification threshold
        """
        super().__init__(model_path, config)

        # Set default task to detection (classification boxes)
        self.task = "detect"

        # Override class names if provided
        if config and "custom_classes" in config:
            self._class_names = config["custom_classes"]
        else:
            self._class_names = self.DEFAULT_CLASSES.copy()

        # Metal ratios for ore quality estimation
        if config and "metal_ratios" in config:
            self.metal_ratios = config["metal_ratios"]
        else:
            self.metal_ratios = self.DEFAULT_METAL_RATIOS.copy()

        self.confidence_threshold = config.get("confidence_threshold", 0.25)

    def get_info(self) -> Dict[str, Any]:
        """Get ore classifier information."""
        info = super().get_info()
        info.update(
            {
                "type": "ore_classifier",
                "application": "mineral_classification",
                "metal_ratios": self.metal_ratios,
                "ore_descriptions": self.ORE_DESCRIPTIONS,
            }
        )
        return info

    def get_ore_description(self, ore_type: str) -> str:
        """
        Get description for ore type.

        Args:
            ore_type: Ore class name (e.g., 'manyetit')

        Returns:
            Description string
        """
        return self.ORE_DESCRIPTIONS.get(ore_type, "Unknown ore type")

    def estimate_metal_ratio(self, ore_type: str) -> float:
        """
        Estimate metal content ratio for ore type.

        Args:
            ore_type: Ore class name

        Returns:
            Metal ratio (0-1)
        """
        return self.metal_ratios.get(ore_type, 0.0)

    def calculate_ore_metrics(self, results: List[Any]) -> Dict[str, Any]:
        """
        Calculate ore-specific metrics from results.

        Args:
            results: Classification results

        Returns:
            Dictionary with ore metrics
        """
        if not results:
            return {
                "total_ores": 0,
                "ores_by_type": {},
                "estimated_metal_ratio": 0.0,
                "waste_ratio": 0.0,
                "avg_confidence": 0.0,
            }

        ores_by_type = {}
        total_confidence = 0.0
        weighted_metal_ratio = 0.0

        for det in results:
            class_name = getattr(det, "class_name", "unknown")
            confidence = getattr(det, "confidence", 0)
            area = getattr(det, "area", 1.0)

            ores_by_type[class_name] = ores_by_type.get(class_name, 0) + 1
            total_confidence += confidence

            # Weight by area and confidence
            metal_ratio = self.estimate_metal_ratio(class_name)
            weighted_metal_ratio += metal_ratio * area * confidence

        total_area = sum(getattr(det, "area", 1.0) for det in results)
        avg_confidence = total_confidence / len(results) if results else 0

        # Normalize weighted ratio
        estimated_metal_ratio = weighted_metal_ratio / total_area if total_area > 0 else 0

        # Calculate waste ratio
        waste_count = ores_by_type.get("atık", 0)
        waste_ratio = waste_count / len(results) if results else 0

        return {
            "total_ores": len(results),
            "ores_by_type": ores_by_type,
            "estimated_metal_ratio": estimated_metal_ratio,
            "waste_ratio": waste_ratio,
            "avg_confidence": avg_confidence,
        }

    def get_diverter_action(self, results: List[Any]) -> str:
        """
        Determine sorting diverter action based on classification.

        Args:
            results: Classification results

        Returns:
            Action: 'accept', 'reject', or 'manual'
        """
        if not results:
            return "reject"

        # Get most confident prediction
        best = max(results, key=lambda x: getattr(x, "confidence", 0))
        class_name = getattr(best, "class_name", "unknown")
        confidence = getattr(best, "confidence", 0)

        # Decision logic
        if confidence < 0.5:
            return "manual"

        if class_name in ["manyetit", "krom"]:
            return "accept"
        elif class_name in ["atık"]:
            return "reject"
        else:
            return "manual"
