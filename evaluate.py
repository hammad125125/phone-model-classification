"""
Independent Test Set Evaluation Script
Mobile Phone Model Classification Using Deep Learning and Transfer Learning

Execution:
    python evaluate.py [--test-dir dataset/test] [--batch-size 16]
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATASET_DIR,
    TEST_DIR,
    WEIGHTS_PATH,
    CLASS_NAMES_PATH,
    MODEL_CONFIG_PATH,
    RESULTS_DIR,
    METRICS_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Evaluator")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Mobile Phone Classifier on Test Dataset")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(TEST_DIR),
        help="Path to directory containing test class folders"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(WEIGHTS_PATH),
        help="Path to trained weights file (.weights.h5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    test_dir = Path(args.test_dir)
    weights_path = Path(args.weights)

    print("=" * 70)
    print("  MOBILE PHONE CLASSIFICATION - INDEPENDENT TEST EVALUATION")
    print("=" * 70)

    # 1. Validation checks
    if not weights_path.exists():
        logger.error(
            f"Trained model weights not found at: {weights_path}\n"
            "Blocker: The model has not been trained yet.\n"
            "To train the model, please execute:\n"
            "    python train.py"
        )
        sys.exit(1)

    if not CLASS_NAMES_PATH.exists():
        logger.error(f"Class names metadata missing at: {CLASS_NAMES_PATH}")
        sys.exit(1)

    if not test_dir.exists():
        logger.error(
            f"Test directory not found at: {test_dir}\n"
            "Please ensure test images are placed in: dataset/test/<class_name>/*.jpg"
        )
        sys.exit(1)

    import tensorflow as tf
    from src.model import load_classifier_model
    from src.evaluator import evaluate_model_on_dataset

    # 2. Reconstruct Model and Load Weights
    logger.info("Loading model and class mapping...")
    try:
        model, class_names, class_data = load_classifier_model(
            weights_path=weights_path,
            class_names_path=CLASS_NAMES_PATH,
            model_config_path=MODEL_CONFIG_PATH
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    logger.info(f"Target classes ({len(class_names)}): {class_names}")

    # 3. Build Test Dataset Pipeline
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    try:
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=args.batch_size,
            shuffle=False
        )
    except Exception as e:
        logger.error(f"Could not load test images from {test_dir}: {e}")
        sys.exit(1)

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = raw_test_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE)

    # 4. Execute Evaluation
    logger.info("Running evaluation across test dataset...")
    metrics = evaluate_model_on_dataset(
        model=model,
        test_ds=test_ds,
        class_names=class_names,
        results_dir=RESULTS_DIR,
        metrics_path=METRICS_PATH
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS:")
    print(f"  Accuracy:           {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score:     {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score:  {metrics['weighted_f1']:.4f}")
    print("=" * 70)
    print(f"Confusion Matrix:    {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"Detailed Report:     {RESULTS_DIR / 'classification_report.json'}")
    print(f"Summary Document:    {RESULTS_DIR / 'evaluation_summary.txt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

