"""
IO utilities for Enterprise Vision AI.
"""

from typing import List


def format_time(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS.

    Args:
        seconds: Seconds

    Returns:
        str: Formatted time
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def create_dummy_results(class_name: str = "defect", count: int = 1) -> List:
    """
    Create dummy results when model is not available.

    Args:
        class_name: Class name
        count: Object count

    Returns:
        List: Dummy results
    """
    return []
