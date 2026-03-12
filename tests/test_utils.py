"""
Tests for services.utils module.
"""

import pytest
from services.utils import (
    get_ore_class_colors,
    get_defect_colors,
    calculate_anomaly_score,
    calculate_metal_ratio,
    get_severity_level,
)


class TestHelperFunctions:
    """Test cases for helper functions in services.utils module."""

    def test_get_ore_class_colors(self):
        """Test that get_ore_class_colors returns the correct dictionary."""
        colors = get_ore_class_colors()
        assert isinstance(colors, dict)
        assert 'manyetit' in colors
        assert 'krom' in colors
        assert 'atık' in colors
        assert 'düşük tenör' in colors
        assert 'defect' in colors
        assert 'normal' in colors

    def test_get_defect_colors(self):
        """Test that get_defect_colors returns the correct dictionary."""
        colors = get_defect_colors()
        assert isinstance(colors, dict)
        assert 'çatlak' in colors
        assert 'çizik' in colors
        assert 'delik' in colors
        assert 'leke' in colors
        assert 'deformasyon' in colors

    def test_calculate_anomaly_score(self):
        """Test that calculate_anomaly_score returns a valid score between 0 and 100."""
        class MockData:
            def __init__(self, data):
                self.data = data
                
            def cpu(self):
                return self
                
            def numpy(self):
                return self.data

        class MockBoxes:
            def __init__(self, confidences):
                import numpy as np
                self.data = MockData(np.array([[10, 10, 20, 20, conf, 0] for conf in confidences]))

        class MockResult:
            def __init__(self, confidences):
                self.boxes = MockBoxes(confidences)

        results = [MockResult([0.1, 0.2, 0.3, 0.4, 0.5])]
        score = calculate_anomaly_score(results)
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score == pytest.approx(30.0)

        score = calculate_anomaly_score([])
        assert score == 0.0

        results = [MockResult([0.0, 0.0, 0.0])]
        score = calculate_anomaly_score(results)
        assert score == 0.0

        results = [MockResult([1.0, 1.0, 1.0])]
        score = calculate_anomaly_score(results)
        assert score == pytest.approx(100.0)

    def test_calculate_metal_ratio(self):
        """Test that calculate_metal_ratio returns a valid metal ratio between 0 and 100."""
        ore_counts = {'manyetit': 2, 'krom': 3, 'atık': 1, 'düşük tenör': 4}
        ratio = calculate_metal_ratio(ore_counts)
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 100

        ore_counts = {'atık': 5, 'düşük tenör': 5}
        ratio = calculate_metal_ratio(ore_counts)
        assert ratio == 0.0

        ore_counts = {'manyetit': 5, 'krom': 5}
        ratio = calculate_metal_ratio(ore_counts)
        assert ratio == 100.0

        ratio = calculate_metal_ratio({})
        assert ratio == 0.0

    def test_get_severity_level(self):
        """Test that get_severity_level returns the correct severity level based on score."""
        assert get_severity_level(25) == 'düşük'
        assert get_severity_level(0) == 'düşük'
        assert get_severity_level(29.9) == 'düşük'

        assert get_severity_level(30) == 'orta'
        assert get_severity_level(50) == 'orta'
        assert get_severity_level(69.9) == 'orta'

        assert get_severity_level(70) == 'yüksek'
        assert get_severity_level(85) == 'yüksek'
        assert get_severity_level(100) == 'yüksek'
