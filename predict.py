"""
Command-Line Prediction Interface
Mobile Phone Model Classification Using Deep Learning and Transfer Learning

Execution:
    python predict.py --image path/to/phone.jpg [--top-k 3] [--json]
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
    WEIGHTS_PATH,
    CLASS_NAMES_PATH,
    MODEL_CONFIG_PATH,
    TOP_K_PREDICTIONS
)
from src.inference import SmartphoneInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Predictor")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict Mobile Phone Brand/Model from an Image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to smartphone image file (.jpg, .jpeg, .png, etc.)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_PREDICTIONS,
        help="Number of top class predictions to display"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    image_path = Path(args.image)

    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    # Initialize Engine
    engine = SmartphoneInferenceEngine(
        weights_path=WEIGHTS_PATH,
        class_names_path=CLASS_NAMES_PATH,
        model_config_path=MODEL_CONFIG_PATH
    )

    if not engine.is_ready:
        logger.error(f"Inference engine not ready: {engine.load_error}")
        logger.error("Please train the model first by executing: python train.py")
        sys.exit(1)

    # Execute Prediction
    result = engine.predict(image_path, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 60)
    print("  MOBILE PHONE CLASSIFICATION - PREDICTION RESULT")
    print("=" * 60)
    if not result.get("success"):
        print(f"Error: {result.get('error')}")
        sys.exit(1)

    print(f"Image:               {image_path.name}")
    print(f"Predicted Class:     {result['predicted_class']}")
    print(f"Confidence:          {result['confidence_percentage']:.2f}% (prob: {result['confidence']})")
    print(f"Inference Time:      {result['inference_time_ms']} ms")
    print("-" * 60)
    print("Top Predictions:")
    for i, pred in enumerate(result.get("top_predictions", []), 1):
        print(f"  {i}. {pred['class']:<20} {pred['confidence_percentage']:>6.2f}%")
    print("-" * 60)
    print("Note: Confidence represents the model's categorical softmax probability.")
    print("=" * 60)


if __name__ == "__main__":
    main()

