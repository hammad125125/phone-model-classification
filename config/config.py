"""
Project Configuration Module
Mobile Phone Model Classification Using Deep Learning and Transfer Learning
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_DIR = BASE_DIR / "docs"

# Dataset Subdirectories
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
SAMPLE_IMAGES_DIR = DATASET_DIR / "sample_test_images"

# Image Specifications (MobileNetV2 Standard)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Allowed Upload Extensions and Max Size
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Training Hyperparameters
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
FINE_TUNE_FROM_LAYER = 100  # MobileNetV2 has 154 layers; unfreeze from 100

# Model Artifact Paths
WEIGHTS_PATH = MODELS_DIR / "mobile_model.weights.h5"
SAVED_MODEL_PATH = MODELS_DIR / "mobile_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
TRAINING_CONFIG_PATH = MODELS_DIR / "training_config.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
DATASET_SUMMARY_PATH = MODELS_DIR / "dataset_summary.json"

# Evaluation Artifact Paths
CONFUSION_MATRIX_PNG = RESULTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PNG = RESULTS_DIR / "training_history.png"
EVALUATION_REPORT_JSON = RESULTS_DIR / "classification_report.json"
EVALUATION_SUMMARY_TXT = RESULTS_DIR / "evaluation_summary.txt"

# Web App Configuration
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", 5000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

# Top-k predictions to return in API
TOP_K_PREDICTIONS = 3

