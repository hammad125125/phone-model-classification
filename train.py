"""
Model Training and Transfer Learning Pipeline
Mobile Phone Model Classification Using Deep Learning and Transfer Learning

Execution:
    python train.py [--epochs 15] [--fine-tune-epochs 10] [--batch-size 16]
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATASET_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    WEIGHTS_PATH,
    SAVED_MODEL_PATH,
    CLASS_NAMES_PATH,
    MODEL_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
    METRICS_PATH,
    DATASET_SUMMARY_PATH,
    TRAINING_CURVES_PNG,
    IMG_HEIGHT,
    IMG_WIDTH,
    INPUT_SHAPE,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    DROPOUT_RATE,
    FINE_TUNE_FROM_LAYER,
    ALLOWED_EXTENSIONS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Trainer")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for Mobile Phone Model Classification"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DATASET_DIR),
        help="Path to dataset root containing train/ and validation/ splits"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=INITIAL_EPOCHS,
        help="Initial training epochs for classification head"
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=FINE_TUNE_EPOCHS,
        help="Fine-tuning epochs for upper MobileNetV2 backbone"
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Skip fine-tuning phase and train head only"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=INITIAL_LEARNING_RATE,
        help="Learning rate for initial head training"
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=FINE_TUNE_LEARNING_RATE,
        help="Learning rate for fine-tuning phase"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT_RATE,
        help="Dropout probability in classification head"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset_path = Path(args.dataset_dir)

    print("=" * 70)
    print("  MOBILE PHONE CLASSIFICATION - DEEP LEARNING TRAINING PIPELINE")
    print("  Architecture: MobileNetV2 (ImageNet Pretrained) + Custom Dense Head")
    print("=" * 70)

    # 1. Inspect and Validate Dataset
    from src.data_loader import inspect_dataset, create_datasets
    from src.dataset_setup import ensure_dataset_structure, create_verification_sample_dataset

    train_folder = dataset_path / "train"
    if not train_folder.exists() or not any(train_folder.iterdir()):
        logger.warning(f"No training data found in {train_folder}.")
        logger.info("Initializing dataset structure with verification calibration samples...")
        ensure_dataset_structure(dataset_path)
        create_verification_sample_dataset(dataset_path, samples_per_class=20)

    logger.info(f"Inspecting dataset at: {dataset_path} ...")
    summary = inspect_dataset(
        dataset_dir=dataset_path,
        allowed_extensions=ALLOWED_EXTENSIONS,
        output_summary_path=DATASET_SUMMARY_PATH
    )

    if summary["status"] != "ready":
        logger.error(f"Dataset is not ready: {summary.get('warnings')}")
        logger.error("Please place phone brand/model images in dataset/train/<brand_or_model>/")
        sys.exit(1)

    classes = summary["classes"]
    num_classes = summary["num_classes"]
    train_count = summary["splits"]["train"]["total"]
    val_count = summary["splits"]["validation"]["total"]
    test_count = summary["splits"]["test"]["total"]

    print("\nDataset Summary:")
    print(f"  - Classes ({num_classes}): {', '.join(classes)}")
    print(f"  - Training Images:    {train_count}")
    print(f"  - Validation Images:  {val_count}")
    print(f"  - Test Images:        {test_count}")
    print(f"  - Total Images:       {summary['total_images']}")
    if summary["leakage_warnings"]:
        print(f"  [!] Leakage Warning: {summary['leakage_warnings']}")

    # 2. Build tf.data pipelines
    logger.info("Building optimized tf.data.Dataset pipelines...")
    train_ds, val_ds, test_ds, loaded_classes = create_datasets(
        dataset_dir=dataset_path,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=args.batch_size
    )

    # Fallback validation set if validation split is empty
    if val_ds is None:
        logger.warning("No validation split found. Splitting training data 80/20 for validation...")
        # tf.data fallback: takes 20%
        val_size = max(1, int(train_count * 0.20 // args.batch_size))
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)

    # 3. Build Model
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from src.model import build_mobilenetv2_classifier, configure_fine_tuning, save_model_artifacts
    from src.evaluator import plot_training_history, evaluate_model_on_dataset

    logger.info(f"Constructing MobileNetV2 transfer learning model for {num_classes} classes...")
    model, base_model = build_mobilenetv2_classifier(
        num_classes=num_classes,
        input_shape=INPUT_SHAPE,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        trainable_base=False
    )

    model.summary(print_fn=logger.info)

    # 4. Callbacks Setup
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temp_checkpoint_path = MODELS_DIR / "best_checkpoint.weights.h5"

    callbacks = [
        ModelCheckpoint(
            filepath=str(temp_checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Phase 1: Train Custom Classification Head
    print("\n" + "=" * 70)
    print(f"PHASE 1: Training Custom Classification Head ({args.epochs} Epochs)")
    print(f"Pretrained MobileNetV2 backbone is FROZEN. Learning Rate = {args.learning_rate}")
    print("=" * 70 + "\n")

    start_time = time.time()
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    combined_history = {
        "accuracy": list(map(float, history_phase1.history.get("accuracy", []))),
        "val_accuracy": list(map(float, history_phase1.history.get("val_accuracy", []))),
        "loss": list(map(float, history_phase1.history.get("loss", []))),
        "val_loss": list(map(float, history_phase1.history.get("val_loss", [])))
    }

    # 6. Phase 2: Fine-Tuning Upper Layers (Optional)
    if not args.no_fine_tune and args.fine_tune_epochs > 0:
        print("\n" + "=" * 70)
        print(f"PHASE 2: Fine-Tuning Upper Layers ({args.fine_tune_epochs} Epochs)")
        print(f"Unfreezing layers from index {FINE_TUNE_FROM_LAYER} onwards. LR = {args.fine_tune_lr}")
        print("=" * 70 + "\n")

        model = configure_fine_tuning(
            model=model,
            base_model=base_model,
            fine_tune_from_layer=FINE_TUNE_FROM_LAYER,
            learning_rate=args.fine_tune_lr
        )

        fine_tune_callbacks = [
            ModelCheckpoint(
                filepath=str(temp_checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=fine_tune_callbacks
        )

        combined_history["accuracy"].extend(list(map(float, history_phase2.history.get("accuracy", []))))
        combined_history["val_accuracy"].extend(list(map(float, history_phase2.history.get("val_accuracy", []))))
        combined_history["loss"].extend(list(map(float, history_phase2.history.get("loss", []))))
        combined_history["val_loss"].extend(list(map(float, history_phase2.history.get("val_loss", []))))

    total_duration = time.time() - start_time
    logger.info(f"Training completed in {total_duration:.1f} seconds.")

    # 7. Restore Best Weights
    if temp_checkpoint_path.exists():
        logger.info(f"Loading best weights from checkpoint: {temp_checkpoint_path}")
        model.load_weights(str(temp_checkpoint_path))

    # 8. Save Final Model Artifacts
    logger.info("Saving project-specific trained weights and configuration...")
    save_model_artifacts(
        model=model,
        weights_path=WEIGHTS_PATH,
        saved_model_path=SAVED_MODEL_PATH,
        class_names=classes,
        class_names_path=CLASS_NAMES_PATH,
        model_config_path=MODEL_CONFIG_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        training_history=combined_history,
        extra_config={
            "initial_epochs": args.epochs,
            "fine_tune_epochs": args.fine_tune_epochs if not args.no_fine_tune else 0,
            "batch_size": args.batch_size,
            "initial_learning_rate": args.learning_rate,
            "fine_tune_learning_rate": args.fine_tune_lr,
            "training_duration_seconds": round(total_duration, 2)
        }
    )

    # 9. Plot and Save Training Curves
    logger.info("Rendering training history curves...")
    plot_training_history(combined_history, TRAINING_CURVES_PNG)

    # 10. Automatic Test Set Evaluation
    if test_ds is not None and test_count > 0:
        print("\n" + "=" * 70)
        print("EVALUATING MODEL ON INDEPENDENT TEST SPLIT")
        print("=" * 70 + "\n")
        try:
            metrics = evaluate_model_on_dataset(
                model=model,
                test_ds=test_ds,
                class_names=classes,
                results_dir=RESULTS_DIR,
                metrics_path=METRICS_PATH
            )
            print(f"Test Set Evaluation Accuracy: {metrics['accuracy'] * 100:.2f}%")
        except Exception as e:
            logger.error(f"Test evaluation encountered an error: {e}")
    else:
        logger.info("No test split available. Run 'python evaluate.py' once test images are placed.")

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Model Weights:  {WEIGHTS_PATH}")
    print(f"Class Mapping:  {CLASS_NAMES_PATH}")
    print(f"Model Config:   {MODEL_CONFIG_PATH}")
    print(f"Training Plots: {TRAINING_CURVES_PNG}")
    print("=" * 70)


if __name__ == "__main__":
    main()

