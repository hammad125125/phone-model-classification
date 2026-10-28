"""
Unit Tests for Image Preprocessing and Validation Pipeline
"""

import io
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from src.inference import SmartphoneInferenceEngine
from src.data_loader import validate_image_file


class TestPreprocessing:
    @pytest.fixture
    def engine(self):
        # Initialize engine without weights to test preprocessing methods
        return SmartphoneInferenceEngine()

    def test_valid_rgb_image_preprocessing(self, engine):
        """Tests that a standard RGB image is correctly resized to (1, 224, 224, 3) and scaled to [-1, 1]."""
        raw_img = Image.new("RGB", (640, 480), color=(128, 128, 128))
        batch = engine.preprocess_image(raw_img)

        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        # Pixel values in [-1, 1] range
        assert batch.min() >= -1.05
        assert batch.max() <= 1.05

    def test_rgba_to_rgb_conversion(self, engine):
        """Tests that 4-channel RGBA images are converted to 3-channel RGB without crashing."""
        raw_img = Image.new("RGBA", (300, 300), color=(100, 150, 200, 255))
        batch = engine.preprocess_image(raw_img)

        assert batch.shape == (1, 224, 224, 3)

    def test_grayscale_to_rgb_conversion(self, engine):
        """Tests that 1-channel Grayscale images are converted to 3-channel RGB."""
        raw_img = Image.new("L", (150, 150), color=200)
        batch = engine.preprocess_image(raw_img)

        assert batch.shape == (1, 224, 224, 3)

    def test_invalid_bytes_rejection(self, engine):
        """Tests that arbitrary non-image byte sequences raise ValueError cleanly."""
        corrupt_bytes = b"NOT_A_VALID_JPEG_HEADER_XYZ12345"
        with pytest.raises(ValueError, match="Image preprocessing failed"):
            engine.preprocess_image(corrupt_bytes)

    def test_validate_image_file_with_valid_and_invalid(self, tmp_path):
        """Tests validate_image_file filters valid files and rejects corrupt or bad extensions."""
        valid_file = tmp_path / "valid.jpg"
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        img.save(valid_file, "JPEG")

        bad_ext_file = tmp_path / "text.txt"
        bad_ext_file.write_text("Hello world")

        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"\x00\x01\x02\x03\x04")

        allowed = {".jpg", ".jpeg", ".png"}

        assert validate_image_file(valid_file, allowed) is True
        assert validate_image_file(bad_ext_file, allowed) is False
        assert validate_image_file(corrupt_file, allowed) is False

