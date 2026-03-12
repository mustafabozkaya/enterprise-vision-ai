"""
Metrics utilities for Enterprise Vision AI.
"""

from typing import Dict, List

import pandas as pd


def calculate_anomaly_score(results) -> float:
    """
    Calculate anomaly score from detection results.

    Args:
        results: YOLO model results

    Returns:
        float: Anomaly score (0-100)
    """
    if results is None or len(results) == 0:
        return 0.0

    total_score = 0.0
    count = 0

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                conf = float(box[4])
                total_score += conf * 100
                count += 1

    if count == 0:
        return 0.0

    return min(100.0, total_score / count)


def get_severity_level(score: float) -> str:
    """
    Get severity level based on anomaly score.

    Args:
        score: Anomaly score (0-100)

    Returns:
        str: 'düşük', 'orta', 'yüksek'
    """
    if score < 30:
        return "düşük"
    elif score < 70:
        return "orta"
    else:
        return "yüksek"


def get_maintenance_recommendation(score: float) -> str:
    """
    Get maintenance recommendation based on anomaly score.

    Args:
        score: Anomaly score (0-100)

    Returns:
        str: Maintenance recommendation
    """
    if score < 30:
        return "✅ Normal operating conditions. Routine maintenance is sufficient."
    elif score < 50:
        return "⚠️ Light anomaly detected. Close monitoring recommended."
    elif score < 70:
        return "🔶 Medium level anomaly. Planned maintenance scheduling recommended."
    elif score < 85:
        return "🔴 High anomaly. Urgent maintenance recommended."
    else:
        return "🚨 Critical! Production should be stopped for maintenance."


def calculate_ore_metrics(results) -> Dict[str, int]:
    """
    Calculate ore classification metrics.

    Args:
        results: YOLO model results

    Returns:
        Dict: Class name -> count
    """
    counts = {"manyetit": 0, "krom": 0, "atık": 0, "düşük tenör": 0}

    if results is None or len(results) == 0:
        return counts

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                cls_id = int(box[5])

                if hasattr(result, "names") and result.names:
                    class_name = result.names.get(cls_id, "").lower()

                    for key in counts.keys():
                        if key in class_name:
                            counts[key] += 1
                            break

    return counts


def calculate_metal_ratio(counts: Dict[str, int]) -> float:
    """
    Calculate metal ratio from ore counts.

    Args:
        counts: Class counts

    Returns:
        float: Metal ratio (0-100)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Valuable ores: magnetite, chrome
    valuable = counts.get("manyetit", 0) + counts.get("krom", 0)

    return (valuable / total) * 100


def get_diverter_recommendation(metal_ratio: float) -> str:
    """
    Get diverter recommendation based on metal ratio.

    Args:
        metal_ratio: Metal ratio (0-100)

    Returns:
        str: Diverter recommendation
    """
    if metal_ratio >= 60:
        return "⬆️ FORWARD PROCESSING: High metal ratio. Send to production."
    elif metal_ratio >= 30:
        return "⏸️ CONTROL: Medium metal ratio. Additional analysis required."
    else:
        return "⬇️ WASTE: Low metal ratio. Divert to waste line."


def create_metrics_dataframe(data: List[Dict]) -> pd.DataFrame:
    """
    Create metrics DataFrame.

    Args:
        data: List of metric data

    Returns:
        pd.DataFrame: Metrics DataFrame
    """
    if not data:
        return pd.DataFrame(columns=["Time", "Anomaly Score", "Severity"])

    return pd.DataFrame(data)


def create_ore_dataframe(counts: Dict[str, int]) -> pd.DataFrame:
    """
    Create ore distribution DataFrame.

    Args:
        counts: Class counts

    Returns:
        pd.DataFrame: Ore DataFrame
    """
    return pd.DataFrame(list(counts.items()), columns=["Class", "Count"])
