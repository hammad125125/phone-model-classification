"""
Unit Tests for Dataset Scanning, Validation, and Leakage Checks
"""

from pathlib import Path
from PIL import Image
from src.data_loader import scan_and_validate_split, check_dataset_leakage, inspect_dataset
from src.dataset_setup import ensure_dataset_structure, create_verification_sample_dataset


class TestDatasetManagement:
    def test_ensure_dataset_structure(self, tmp_path):
        """Verifies that directory structure and subfolders are properly created."""
        classes = ["ClassA", "ClassB", "ClassC"]
        ensure_dataset_structure(tmp_path, classes)

        assert (tmp_path / "train" / "ClassA").exists()
        assert (tmp_path / "validation" / "ClassB").exists()
        assert (tmp_path / "test" / "ClassC").exists()
        assert (tmp_path / "sample_test_images").exists()

    def test_leakage_detection(self, tmp_path):
        """Tests that overlapping filenames between splits are flagged immediately."""
        train_p = tmp_path / "train" / "apple" / "img1.jpg"
        val_p = tmp_path / "validation" / "apple" / "img1.jpg"
        test_p = tmp_path / "test" / "apple" / "img2.jpg"

        train_dict = {"apple": [train_p]}
        val_dict = {"apple": [val_p]}
        test_dict = {"apple": [test_p]}

        warnings = check_dataset_leakage(train_dict, val_dict, test_dict)
        assert len(warnings) > 0
        assert "Filename overlap between train and validation" in warnings[0]

    def test_sample_dataset_generation(self, tmp_path):
        """Verifies synthetic calibration sample generator creates valid, readable images."""
        classes = ["BrandX", "BrandY"]
        create_verification_sample_dataset(tmp_path, classes=classes, samples_per_class=10)

        summary = inspect_dataset(tmp_path, allowed_extensions={".jpg", ".jpeg", ".png"})
        assert summary["status"] == "ready"
        assert summary["num_classes"] == 2
        assert summary["total_images"] > 0
        assert len(summary["leakage_warnings"]) == 0

