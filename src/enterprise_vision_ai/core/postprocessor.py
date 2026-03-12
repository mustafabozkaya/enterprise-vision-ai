"""
Postprocessing module for inference results.

Handles:
- Result filtering
- NMS (Non-Maximum Suppression)
- Coordinate transformation
- Annotation generation
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class DetectionResult:
    """Single detection result."""

    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: Optional[List[float]] = None,
        mask: Optional[np.ndarray] = None,
        area: Optional[float] = None,
    ):
        """
        Initialize detection result.

        Args:
            class_id: Class ID
            class_name: Human-readable class name
            confidence: Detection confidence (0-1)
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Segmentation mask (if applicable)
            area: Object area in pixels
        """
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.mask = mask
        self.area = area

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "area": self.area,
        }

    def __repr__(self) -> str:
        return f"DetectionResult({self.class_name}: {self.confidence:.2f})"


class Postprocessor:
    """Postprocessor for inference results."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize postprocessor.

        Args:
            confidence_threshold: Minimum confidence to keep detection
            iou_threshold: IoU threshold for NMS
            class_names: Mapping from class ID to name
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

    def process(
        self,
        raw_results: Any,
        original_image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[DetectionResult]:
        """
        Process raw model outputs.

        Args:
            raw_results: Raw model output
            original_image_shape: Original (h, w) of input image

        Returns:
            List of processed detection results
        """
        # This will be overridden by specific implementations
        # Default behavior for YOLO results
        if hasattr(raw_results, 'boxes'):
            return self._process_yolo_result(raw_results, original_image_shape)

        return []

    def _process_yolo_result(
        self,
        result,
        original_image_shape: Optional[Tuple[int, int]],
    ) -> List[DetectionResult]:
        """Process YOLO result object."""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])

            if conf < self.confidence_threshold:
                continue

            cls_id = int(boxes.cls[i])
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # Get bbox
            xyxy = boxes.xyxy[i].tolist()

            # Get mask if available
            mask = None
            if result.masks is not None and i < len(result.masks):
                mask = result.masks.data[i].cpu().numpy()

            detection = DetectionResult(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox=xyxy,
                mask=mask,
            )
            detections.append(detection)

        return detections

    def filter_by_class(
        self,
        detections: List[DetectionResult],
        class_ids: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
    ) -> List[DetectionResult]:
        """
        Filter detections by class.

        Args:
            detections: List of detections
            class_ids: Keep only these class IDs
            class_names: Keep only these class names

        Returns:
            Filtered detections
        """
        if class_ids is None and class_names is None:
            return detections

        filtered = []
        for det in detections:
            if class_ids and det.class_id in class_ids:
                filtered.append(det)
            elif class_names and det.class_name in class_names:
                filtered.append(det)

        return filtered

    def calculate_metrics(
        self,
        detections: List[DetectionResult],
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from detections.

        Args:
            detections: List of detections

        Returns:
            Dictionary with metrics
        """
        if not detections:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "class_distribution": {},
            }

        confidences = [d.confidence for d in detections]
        class_counts = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        return {
            "count": len(detections),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "class_distribution": class_counts,
        }

    @staticmethod
    def calculate_severity(
        detections: List[DetectionResult],
        thresholds: Tuple[float, float] = (0.3, 0.7),
    ) -> str:
        """
        Calculate severity level based on detections.

        Args:
            detections: List of detections
            thresholds: (low, high) thresholds

        Returns:
            Severity level: 'düşük', 'orta', or 'yüksek'
        """
        if not detections:
            return "düşük"

        # Calculate anomaly score from average confidence
        avg_conf = np.mean([d.confidence for d in detections])
        count_factor = min(len(detections) / 5, 1.0)  # Normalize to 0-1
        anomaly_score = (avg_conf * 0.7 + count_factor * 0.3) * 100

        low_threshold, high_threshold = thresholds

        if anomaly_score < low_threshold * 100:
            return "düşük"
        elif anomaly_score < high_threshold * 100:
            return "orta"
        else:
            return "yüksek"
