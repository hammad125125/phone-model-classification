"""
MobileNetV2 Transfer Learning Model Architecture and Lifecycle
Mobile Phone Model Classification Using Deep Learning and Transfer Learning
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_mobilenetv2_classifier(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    dense_units: int = 128,
    learning_rate: float = 1e-3,
    trainable_base: bool = False,
    fine_tune_from_layer: Optional[int] = None
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs the MobileNetV2 Transfer Learning architecture.
    Returns:
        - model: Full functional model ready for training or inference.
        - base_model: The MobileNetV2 feature extractor backbone.
    """
    # 1. Base Pretrained Backbone
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # 2. Backbone Freezing Strategy
    if not trainable_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        if fine_tune_from_layer is not None:
            for layer in base_model.layers[:fine_tune_from_layer]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_from_layer:]:
                # Keep BatchNormalization in inference mode during fine-tuning for stability
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

    # 3. Functional Architecture Construction
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=trainable_base)

    # 4. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    if dense_units and dense_units > 0:
        x = layers.Dense(dense_units, activation="relu", name="head_dense")(x)
        x = layers.Dropout(dropout_rate * 0.7, name="head_dropout_2")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobile_phone_classifier")

    # 5. Compile with Adam and Categorical Crossentropy
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model, base_model


def configure_fine_tuning(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    fine_tune_from_layer: int = 100,
    learning_rate: float = 1e-5
) -> tf.keras.Model:
    """
    Unfreezes the upper layers of MobileNetV2 from `fine_tune_from_layer` onwards
    and recompiles the model with a conservative learning rate.
    """
    base_model.trainable = True

    # Freeze all layers before fine_tune_from_layer
    for layer in base_model.layers[:fine_tune_from_layer]:
        layer.trainable = False

    # Freeze BatchNormalization layers to prevent moving mean/variance destruction
    for layer in base_model.layers[fine_tune_from_layer:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_count = sum(bool(layer.trainable) for layer in model.layers)
    logger.info(
        f"Fine-tuning enabled from layer {fine_tune_from_layer}. "
        f"Model recompiled with lr={learning_rate}."
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def save_model_artifacts(
    model: tf.keras.Model,
    weights_path: Path,
    saved_model_path: Path,
    class_names: List[str],
    class_names_path: Path,
    model_config_path: Path,
    training_config_path: Path,
    training_history: Optional[Dict] = None,
    extra_config: Optional[Dict] = None
) -> None:
    """
    Saves model weights, full model archive, class labels, and complete configuration metadata.
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save Weights (.weights.h5)
    model.save_weights(str(weights_path))
    logger.info(f"Model weights saved to: {weights_path}")

    # 2. Save Complete Keras Model (.keras)
    try:
        model.save(str(saved_model_path))
        logger.info(f"Full Keras model saved to: {saved_model_path}")
    except Exception as e:
        logger.warning(f"Could not save .keras package directly: {e}")

    # 3. Save class names mapping
    class_mapping = {
        "classes": class_names,
        "num_classes": len(class_names),
        "class_to_idx": {cls: idx for idx, cls in enumerate(class_names)},
        "idx_to_class": {idx: cls for idx, cls in enumerate(class_names)}
    }
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    logger.info(f"Class names mapping saved to: {class_names_path}")

    # 4. Save model configuration
    model_config = {
        "architecture": "MobileNetV2",
        "weights_initialization": "ImageNet",
        "input_shape": list(model.input_shape[1:]),
        "num_classes": len(class_names),
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
        "non_trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
        "extra_info": extra_config or {}
    }
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Model config saved to: {model_config_path}")

    # 5. Save training configuration / history metadata
    training_meta = {
        "history": training_history or {},
        "framework": f"TensorFlow {tf.__version__}",
        "keras_version": getattr(tf.keras, "__version__", tf.__version__),
        "training_completed": bool(training_history)
    }
    with open(training_config_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)
    logger.info(f"Training metadata saved to: {training_config_path}")


def load_classifier_model(
    weights_path: Path,
    class_names_path: Path,
    model_config_path: Optional[Path] = None
) -> Tuple[tf.keras.Model, List[str], Dict]:
    """
    Safely reconstructs the deterministic MobileNetV2 architecture and loads the saved weights.
    Performs comprehensive validation against missing files, corrupted checkpoints, and dimension mismatches.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights file not found at: {weights_path}. "
            "Please train the model first by executing: python train.py"
        )

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names metadata not found at: {class_names_path}. "
            "Model cannot infer output classes without this file."
        )

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_data = json.load(f)

    class_names = class_data.get("classes", [])
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"Invalid class mapping: expected >= 2 classes, got {num_classes}")

    input_shape = (224, 224, 3)
    dense_units = 128
    dropout_rate = 0.3

    if model_config_path and model_config_path.exists():
        try:
            with open(model_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "input_shape" in cfg:
                    input_shape = tuple(cfg["input_shape"])
        except Exception as e:
            logger.warning(f"Could not read model config: {e}. Using defaults.")

    logger.info(f"Rebuilding MobileNetV2 architecture with {num_classes} classes...")
    model, _ = build_mobilenetv2_classifier(
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )

    logger.info(f"Loading weights from {weights_path}...")
    try:
        model.load_weights(str(weights_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load weights into MobileNetV2 architecture: {e}")

    logger.info("Model reconstructed and weights loaded successfully.")
    return model, class_names, class_data

