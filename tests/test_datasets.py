"""
Tests for enterprise_vision_ai datasets package.
"""

import pytest

from enterprise_vision_ai.datasets.data_loader import (
    DefectDetectionDataset,
    OreClassificationDataset,
    load_dataset,
)


class TestDefectDetectionDataset:
    """Test cases for DefectDetectionDataset class."""

    def test_dataset_init(self):
        """Test DefectDetectionDataset initialization."""
        dataset = DefectDetectionDataset("test_defect", split="train")
        assert dataset.name == "test_defect"
        assert dataset.split == "train"

    def test_dataset_with_custom_dir(self):
        """Test dataset with custom data directory."""
        dataset = DefectDetectionDataset("test", data_dir="/custom/data")
        assert dataset.data_dir == "/custom/data"

    def test_get_classes(self):
        """Test get_classes method."""
        dataset = DefectDetectionDataset("test")
        classes = dataset.get_classes()
        assert isinstance(classes, list)

    def test_get_num_classes(self):
        """Test get_num_classes method."""
        dataset = DefectDetectionDataset("test")
        num = dataset.get_num_classes()
        assert isinstance(num, int)
        assert num == 0


class TestOreClassificationDataset:
    """Test cases for OreClassificationDataset class."""

    def test_dataset_init(self):
        """Test OreClassificationDataset initialization."""
        dataset = OreClassificationDataset("test_ore", split="train")
        assert dataset.name == "test_ore"
        assert dataset.split == "train"

    def test_dataset_with_custom_dir(self):
        """Test dataset with custom data directory."""
        dataset = OreClassificationDataset("test", data_dir="/custom/ore")
        assert dataset.data_dir == "/custom/ore"

    def test_get_classes(self):
        """Test get_classes method."""
        dataset = OreClassificationDataset("test")
        classes = dataset.get_classes()
        assert isinstance(classes, list)

    def test_get_num_classes(self):
        """Test get_num_classes method."""
        dataset = OreClassificationDataset("test")
        num = dataset.get_num_classes()
        assert isinstance(num, int)
        assert num == 0


class TestLoadDataset:
    """Test cases for load_dataset convenience function."""

    def test_load_defect_dataset(self):
        """Test loading defect detection dataset."""
        dataset = load_dataset("defect_detection", split="train")
        assert isinstance(dataset, DefectDetectionDataset)

    def test_load_ore_dataset(self):
        """Test loading ore classification dataset."""
        dataset = load_dataset("ore_classification", split="train")
        assert isinstance(dataset, OreClassificationDataset)

    def test_load_dataset_auto_detect(self):
        """Test auto-detection of dataset type."""
        # Should auto-detect defect
        dataset = load_dataset("my_defect_dataset")
        assert isinstance(dataset, DefectDetectionDataset)

        # Should auto-detect ore
        dataset = load_dataset("my_ore_dataset")
        assert isinstance(dataset, OreClassificationDataset)

        # Should auto-detect mineral
        dataset = load_dataset("mineral_dataset")
        assert isinstance(dataset, OreClassificationDataset)

    def test_load_dataset_with_task(self):
        """Test loading dataset with explicit task."""
        dataset = load_dataset("custom", task="defect_detection")
        assert isinstance(dataset, DefectDetectionDataset)

        dataset = load_dataset("custom", task="ore_classification")
        assert isinstance(dataset, OreClassificationDataset)

    def test_load_dataset_invalid_task(self):
        """Test loading dataset with invalid task."""
        with pytest.raises(ValueError):
            load_dataset("test", task="invalid_task")
