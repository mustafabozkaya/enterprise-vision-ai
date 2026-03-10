"""
Tests for enterprise_vision_ai package imports and version.
"""

import pytest


class TestEnterpriseVisionAI:
    """Test cases for enterprise_vision_ai package."""

    def test_import_from_enterprise_vision_ai(self):
        """Test that imports from enterprise_vision_ai work."""
        from enterprise_vision_ai import load_dataset, load_model

        assert callable(load_model)
        assert callable(load_dataset)

    def test_version(self):
        """Test version is 2.0.0."""
        import enterprise_vision_ai

        assert enterprise_vision_ai.__version__ == "2.0.0"

    def test_author(self):
        """Test author is set correctly."""
        import enterprise_vision_ai

        assert "Enterprise Vision AI" in enterprise_vision_ai.__author__
