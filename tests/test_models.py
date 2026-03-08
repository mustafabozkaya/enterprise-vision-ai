"""
Tests for enterprise_vision_ai models package.
"""

import pytest
from enterprise_vision_ai.models.model_manager import ModelManager, load_model


class TestModelManager:
    """Test cases for ModelManager class."""
    
    def test_model_manager_init(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        assert manager is not None
        assert isinstance(manager.models, dict)
        assert len(manager.models) == 0
    
    def test_model_manager_with_cache_dir(self):
        """Test ModelManager with custom cache directory."""
        manager = ModelManager(cache_dir="/custom/path")
        assert manager.cache_dir == "/custom/path"
    
    def test_load_model(self):
        """Test model loading."""
        manager = ModelManager()
        model = manager.load("yolov8n-defect")
        assert model is not None
        assert model.name == "yolov8n-defect"
    
    def test_load_model_caching(self):
        """Test that models are cached."""
        manager = ModelManager()
        model1 = manager.load("yolov8n-defect")
        model2 = manager.load("yolov8n-defect")
        assert model1 is model2
    
    def test_unload_model(self):
        """Test model unloading."""
        manager = ModelManager()
        manager.load("yolov8n-defect")
        assert "yolov8n-defect" in manager.models
        manager.unload("yolov8n-defect")
        assert "yolov8n-defect" not in manager.models
    
    def test_list_models(self):
        """Test listing loaded models."""
        manager = ModelManager()
        manager.load("model1")
        manager.load("model2")
        models = manager.list_models()
        assert "model1" in models
        assert "model2" in models


class TestLoadModel:
    """Test cases for load_model convenience function."""
    
    def test_load_model_function(self):
        """Test load_model function."""
        model = load_model("yolov8n-defect")
        assert model is not None
    
    def test_load_model_returns_stub(self):
        """Test that load_model returns a stub."""
        model = load_model("test-model")
        # Stub should have predict method
        assert hasattr(model, 'predict')
        results = model.predict("test.jpg")
        assert results == []
