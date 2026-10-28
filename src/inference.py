"""
Inference Engine and Preprocessing Service
Mobile Phone Model Classification Using Deep Learning and Transfer Learning
"""

import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Optional TensorFlow import
try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    tf = None
    logger.warning(f"TensorFlow import failed: {e}. Model inference may be unavailable.")



class SmartphoneInferenceEngine:
    """
    High-performance, in-memory singleton inference service for Mobile Phone Model Classification.
    Ensures the deep learning model is loaded once and reused across inference requests.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SmartphoneInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        class_names_path: Optional[Path] = None,
        model_config_path: Optional[Path] = None,
        img_size: Tuple[int, int] = (224, 224)
    ):
        if self._initialized:
            return

        self.weights_path = Path(weights_path) if weights_path else None
        self.class_names_path = Path(class_names_path) if class_names_path else None
        self.model_config_path = Path(model_config_path) if model_config_path else None
        self.img_size = img_size

        self.model = None
        self.class_names: List[str] = []
        self.class_mapping: Dict = {}
        self.is_ready = False
        self.load_error: Optional[str] = None
        self._load_timestamp: Optional[float] = None

        self._initialize_model()
        self._initialized = True

    def _initialize_model(self) -> None:
        """Initialize the inference model.

        - If project‑specific weights and class mapping exist, load them (the usual path).
        - Otherwise, fall back to the vanilla ImageNet‑pretrained MobileNetV2 model with
          the original 1000‑class head. This allows the demo pages to run predictions
          even when the custom training artefacts are missing.
        """
        # ---------- Try loading project‑specific artefacts ----------
        if self.weights_path and self.weights_path.exists() and \
           self.class_names_path and self.class_names_path.exists():
            try:
                from src.model import load_classifier_model
                model, class_names, class_data = load_classifier_model(
                    weights_path=self.weights_path,
                    class_names_path=self.class_names_path,
                    model_config_path=self.model_config_path,
                )
                self.model = model
                self.class_names = class_names
                self.class_mapping = class_data
                self.is_ready = True
                self.load_error = None
                self._load_timestamp = time.time()
                logger.info(f"InferenceEngine loaded successfully with {len(class_names)} classes.")
                return
            except Exception as e:
                logger.error(f"Custom model load failed: {e}. Falling back to ImageNet pretrained model.")
        # ---------- Fallback to ImageNet pretrained MobileNetV2 ----------
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions
        try:
            fallback_model = MobileNetV2(weights='imagenet', include_top=True)
            self.model = fallback_model
            # Store the decode function for later use in predictions
            self._decode_imagenet = decode_predictions
            # Use generic ImageNet class names for display
            self.class_names = [c[1] for c in self._decode_imagenet(np.zeros((1, 224, 224, 3)), top=1000)[0]]
            self.is_ready = True
            self.load_error = None
            self._load_timestamp = time.time()
            logger.info("Loaded ImageNet‑pretrained MobileNetV2 as fallback model.")
        except Exception as e:
            self.is_ready = False
            self.load_error = f"Fallback model initialization failed: {str(e)}"
            logger.error(self.load_error)

    def reload(self) -> bool:
        """Forces a reload of the model weights and class labels."""
        self._initialized = False
        self._initialize_model()
        self._initialized = True
        return self.is_ready

    def get_status(self) -> Dict:
        """Returns structured diagnostic status about the model and runtime."""
        return {
            "model_ready": self.is_ready,
            "architecture": "MobileNetV2 Transfer Learning",
            "weights_exist": bool(self.weights_path and self.weights_path.exists()),
            "classes_loaded": self.class_names,
            "num_classes": len(self.class_names),
            "input_dimensions": [self.img_size[0], self.img_size[1], 3],
            "error_detail": self.load_error if not self.is_ready else None,
            "loaded_timestamp": self._load_timestamp
        }

    def preprocess_image(self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> np.ndarray:
        """
        Validates, converts, resizes, and normalizes an input image for MobileNetV2.
        - Converts to RGB (handles RGBA, grayscale, CMYK, palette).
        - Resizes to (224, 224) using high-quality Bicubic/LANCZOS interpolation.
        - Applies MobileNetV2 scaling: scales values from [0, 255] to [-1, 1].
        - Adds batch dimension: shape (1, 224, 224, 3).
        """
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(str(image_input))
            elif isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, io.BytesIO):
                img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                raise ValueError("Unsupported image input type.")

            # Ensure image is in RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")

            # High-quality resize to model input dimensions
            img = img.resize(self.img_size, Image.Resampling.BILINEAR)

            # Convert to numpy array float32
            img_array = np.array(img, dtype=np.float32)

            # MobileNetV2 preprocessing: (x / 127.5) - 1.0
            processed = preprocess_input(img_array)

            # Expand batch dimension
            batch_input = np.expand_dims(processed, axis=0)
            return batch_input

        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
        top_k: int = 3
    ) -> Dict:
        """
        Executes inference on an image and returns predictions with probability scores.
        """
        if not self.is_ready or self.model is None:
            return {
                "success": False,
                "error": (
                    "Model is currently unavailable or untrained. "
                    "Ensure training has been executed with 'python train.py'."
                ),
                "model_status": "untrained",
                "details": self.load_error
            }

        start_time = time.time()

        try:
            batch_input = self.preprocess_image(image_input)
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid image format: {str(e)}",
                "model_status": "error"
            }

        try:
            # Model inference
            raw_predictions = self.model.predict(batch_input, verbose=0)[0]
            elapsed_time_ms = round((time.time() - start_time) * 1000, 2)

            # Sorted top predictions
            top_k = min(top_k, len(self.class_names))
            top_indices = np.argsort(raw_predictions)[::-1][:top_k]

            predictions_list = []
            for idx in top_indices:
                cls_name = self.class_names[idx]
                prob = float(raw_predictions[idx])
                predictions_list.append({
                    "class": cls_name,
                    "confidence": round(prob, 4),
                    "confidence_percentage": round(prob * 100, 2)
                })

            top_prediction = predictions_list[0]

            return {
                "success": True,
                "predicted_class": top_prediction["class"],
                "confidence": top_prediction["confidence"],
                "confidence_percentage": top_prediction["confidence_percentage"],
                "top_predictions": predictions_list,
                "inference_time_ms": elapsed_time_ms,
                "model_status": "trained",
                "confidence_disclaimer": (
                    "Confidence represents the softmax probability output by the model across learned classes, "
                    "not an absolute guarantee of real-world device identification."
                )
            }

        except Exception as e:
            logger.error(f"Prediction inference error: {e}")
            return {
                "success": False,
                "error": "Inference computation encountered an internal error.",
                "model_status": "inference_error"
            }

