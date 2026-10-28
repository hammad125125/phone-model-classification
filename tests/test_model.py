"""
Unit Tests for MobileNetV2 Architecture Construction and Lifecycle
"""

import pytest
from pathlib import Path


class TestModelArchitecture:
    def test_mobilenetv2_construction_and_shapes(self):
        """Tests that MobileNetV2 classifier compiles with correct input/output shapes and frozen base."""
        import tensorflow as tf
        from src.model import build_mobilenetv2_classifier

        num_classes = 5
        input_shape = (224, 224, 3)

        model, base_model = build_mobilenetv2_classifier(
            num_classes=num_classes,
            input_shape=input_shape,
            dropout_rate=0.3,
            trainable_base=False
        )

        assert model.input_shape == (None, 224, 224, 3)
        assert model.output_shape == (None, num_classes)
        assert base_model.trainable is False

        # Custom head layers must be trainable
        head_dense = model.get_layer("classification_output")
        assert head_dense.trainable is True

    def test_fine_tuning_unfreezing(self):
        """Tests that configure_fine_tuning correctly unfreezes upper layers from specified index."""
        from src.model import build_mobilenetv2_classifier, configure_fine_tuning

        model, base_model = build_mobilenetv2_classifier(
            num_classes=3,
            trainable_base=False
        )

        model = configure_fine_tuning(
            model=model,
            base_model=base_model,
            fine_tune_from_layer=100,
            learning_rate=1e-5
        )

        assert base_model.trainable is True
        # Layers before 100 should remain frozen
        assert base_model.layers[50].trainable is False
        # Non-BatchNormalization layers at/after 100 should be trainable
        assert any(l.trainable for l in base_model.layers[100:])

    def test_missing_weights_loading_raises_error(self, tmp_path):
        """Tests that load_classifier_model raises FileNotFoundError on non-existent weights."""
        from src.model import load_classifier_model

        fake_weights = tmp_path / "non_existent.weights.h5"
        fake_classes = tmp_path / "class_names.json"

        with pytest.raises(FileNotFoundError):
            load_classifier_model(fake_weights, fake_classes)

