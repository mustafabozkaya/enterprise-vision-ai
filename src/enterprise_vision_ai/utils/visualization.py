"""
Visualization utilities for Enterprise Vision AI.
"""

from typing import Dict, Tuple

import cv2
import numpy as np


def get_ore_class_colors() -> Dict[str, Tuple[int, int, int]]:
    """
    Get ore class colors.

    Returns:
        Dict: Class name -> RGB color tuple
    """
    return {
        "manyetit": (220, 20, 60),  # Crimson
        "krom": (0, 255, 127),  # Spring green
        "atık": (128, 128, 128),  # Gray
        "düşük tenör": (255, 165, 0),  # Orange
        "defect": (255, 0, 255),  # Magenta
        "normal": (0, 255, 0),  # Light green
    }


def get_defect_colors() -> Dict[str, Tuple[int, int, int]]:
    """
    Get defect type colors.

    Returns:
        Dict: Defect type -> RGB color tuple
    """
    return {
        "çatlak": (255, 0, 0),  # Red
        "çizik": (0, 255, 255),  # Cyan
        "delik": (255, 165, 0),  # Orange
        "leke": (128, 0, 128),  # Purple
        "deformasyon": (0, 0, 255),  # Blue
    }


def get_severity_color(severity: str) -> Tuple[int, int, int]:
    """
    Get color based on severity level.

    Args:
        severity: 'düşük', 'orta', 'yüksek'

    Returns:
        Tuple: RGB color
    """
    colors = {
        "düşük": (0, 255, 0),  # Green
        "orta": (255, 165, 0),  # Orange
        "yüksek": (255, 0, 0),  # Red
    }
    return colors.get(severity.lower(), (128, 128, 128))


def draw_annotations(
    image: np.ndarray,
    results,
    class_colors: Dict[str, Tuple[int, int, int]],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw YOLO results on image.

    Args:
        image: Input image (BGR format)
        results: YOLO model results
        class_colors: Class colors dict
        show_labels: Show labels
        show_confidence: Show confidence scores
        line_thickness: Line thickness

    Returns:
        np.ndarray: Annotated image
    """
    result_image = image.copy()

    # Convert to RGB (OpenCV BGR -> RGB)
    if len(result_image.shape) == 3 and result_image.shape[2] == 3:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    if results is None or len(results) == 0:
        return result_image

    # Process each result
    for result in results:
        # Segmentation masks
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for mask, box in zip(masks, boxes):
                cls_id = int(box[5])
                conf = float(box[4])

                # Get class name
                if hasattr(result, "names") and result.names:
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                else:
                    class_name = f"class_{cls_id}"

                # Select color
                color = class_colors.get(class_name, (255, 255, 255))

                # Prepare mask
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Resize to original image size
                target_h, target_w = result_image.shape[:2]
                mask_resized = cv2.resize(
                    mask_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

                # Create colored overlay
                mask_colored = np.zeros_like(result_image)
                for i, c in enumerate(color):
                    mask_colored[:, :, i] = mask_resized * (c / 255.0)

                # Apply mask overlay
                result_image = cv2.addWeighted(result_image, 1, mask_colored, 0.3, 0)

        # Bounding boxes
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)

                # Get class name
                if hasattr(result, "names") and result.names:
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                else:
                    class_name = f"class_{cls_id}"

                # Select color
                color = class_colors.get(class_name, (255, 255, 255))

                # Draw box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_thickness)

                # Draw label
                if show_labels:
                    label = class_name
                    if show_confidence:
                        label += f" {conf:.2f}"

                    # Label background
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        result_image,
                        (x1, y1 - label_h - 10),
                        (x1 + label_w, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        result_image,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

    return result_image


def draw_severity_indicator(image: np.ndarray, severity: str, score: float) -> np.ndarray:
    """
    Add severity indicator to image.

    Args:
        image: Input image
        severity: Severity level
        score: Anomaly score

    Returns:
        np.ndarray: Image with indicator
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Get color
    color = get_severity_color(severity)

    # Indicator box
    box_w, box_h = 200, 80
    x, y = w - box_w - 20, 20

    # Semi-transparent background
    overlay = result.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), color, -1)
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

    # Border
    cv2.rectangle(result, (x, y), (x + box_w, y + box_h), color, 3)

    # Text
    cv2.putText(
        result,
        f"Severity: {severity.upper()}",
        (x + 10, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        result,
        f"Score: {score:.1f}",
        (x + 10, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return result
