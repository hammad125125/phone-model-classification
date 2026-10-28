"""
Dataset Management and Preprocessing Pipeline
Mobile Phone Model Classification Using Deep Learning and Transfer Learning
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_image_file(file_path: Path, allowed_extensions: set) -> bool:
    """
    Check if a file has a valid image extension and can be opened/verified with PIL.
    Filters out corrupt files, zero-byte files, and non-image artifacts (.DS_Store, thumbs.db).
    """
    if file_path.suffix.lower() not in allowed_extensions:
        return False
    if file_path.stat().st_size == 0:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Verify only tests headers; reopen to ensure data is loadable
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Corrupt or unreadable image found: {file_path} - {e}")
        return False


def scan_and_validate_split(
    split_dir: Path,
    allowed_extensions: set,
    expected_classes: Optional[List[str]] = None
) -> Tuple[Dict[str, List[Path]], List[str], List[Path]]:
    """
    Scans a dataset split folder (train/val/test).
    Returns:
        - class_to_images: mapping of class_name -> list of valid image Path objects
        - classes: sorted list of class names found
        - corrupted_files: list of corrupt/invalid files encountered
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {split_dir}")

    class_to_images: Dict[str, List[Path]] = {}
    corrupted_files: List[Path] = []

    subdirs = [p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subdirs:
        logger.warning(f"No class subdirectories found in {split_dir}")
        return {}, [], []

    classes = sorted([d.name for d in subdirs])

    if expected_classes is not None:
        missing = set(expected_classes) - set(classes)
        if missing:
            logger.warning(f"Split {split_dir.name} is missing expected classes: {missing}")

    for class_name in classes:
        class_folder = split_dir / class_name
        valid_images = []
        for file_path in class_folder.iterdir():
            if file_path.is_file():
                if validate_image_file(file_path, allowed_extensions):
                    valid_images.append(file_path)
                else:
                    corrupted_files.append(file_path)
        class_to_images[class_name] = valid_images

    return class_to_images, classes, corrupted_files


def check_dataset_leakage(
    train_images: Dict[str, List[Path]],
    val_images: Dict[str, List[Path]],
    test_images: Dict[str, List[Path]]
) -> List[str]:
    """
    Verifies that no identical filenames or files exist across train, validation, and test splits.
    Returns list of leakage warnings if found.
    """
    leakage_warnings = []
    train_names = {p.name: p for paths in train_images.values() for p in paths}
    val_names = {p.name: p for paths in val_images.values() for p in paths}
    test_names = {p.name: p for paths in test_images.values() for p in paths}

    train_val_overlap = set(train_names.keys()) & set(val_names.keys())
    if train_val_overlap:
        leakage_warnings.append(f"Filename overlap between train and validation: {train_val_overlap}")

    train_test_overlap = set(train_names.keys()) & set(test_names.keys())
    if train_test_overlap:
        leakage_warnings.append(f"Filename overlap between train and test: {train_test_overlap}")

    val_test_overlap = set(val_names.keys()) & set(test_names.keys())
    if val_test_overlap:
        leakage_warnings.append(f"Filename overlap between validation and test: {val_test_overlap}")

    return leakage_warnings


def inspect_dataset(
    dataset_dir: Path,
    allowed_extensions: set,
    output_summary_path: Optional[Path] = None
) -> Dict:
    """
    Comprehensive dataset inspection:
    - Scans train, validation, and test splits.
    - Validates class alignment across splits.
    - Computes class balance and image counts.
    - Checks for dataset leakage.
    - Generates and optionally persists summary JSON.
    """
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    summary = {
        "dataset_path": str(dataset_dir),
        "status": "ready",
        "splits": {},
        "classes": [],
        "num_classes": 0,
        "class_to_idx": {},
        "idx_to_class": {},
        "total_images": 0,
        "corrupted_images_count": 0,
        "class_balance": {},
        "leakage_warnings": [],
        "warnings": []
    }

    if not train_dir.exists():
        summary["status"] = "missing_train_split"
        summary["warnings"].append(f"Train directory not found at {train_dir}")
        return summary

    train_imgs, train_classes, train_corrupt = scan_and_validate_split(train_dir, allowed_extensions)
    if not train_classes:
        summary["status"] = "empty_train_split"
        summary["warnings"].append("No valid classes or images found in train split.")
        return summary

    summary["classes"] = train_classes
    summary["num_classes"] = len(train_classes)
    summary["class_to_idx"] = {cls_name: idx for idx, cls_name in enumerate(train_classes)}
    summary["idx_to_class"] = {idx: cls_name for idx, cls_name in enumerate(train_classes)}

    val_imgs, val_classes, val_corrupt = ({}, [], [])
    if val_dir.exists():
        val_imgs, val_classes, val_corrupt = scan_and_validate_split(
            val_dir, allowed_extensions, expected_classes=train_classes
        )

    test_imgs, test_classes, test_corrupt = ({}, [], [])
    if test_dir.exists():
        test_imgs, test_classes, test_corrupt = scan_and_validate_split(
            test_dir, allowed_extensions, expected_classes=train_classes
        )

    all_corrupt_count = len(train_corrupt) + len(val_corrupt) + len(test_corrupt)
    summary["corrupted_images_count"] = all_corrupt_count

    # Counts per split
    train_counts = {cls: len(train_imgs.get(cls, [])) for cls in train_classes}
    val_counts = {cls: len(val_imgs.get(cls, [])) for cls in train_classes}
    test_counts = {cls: len(test_imgs.get(cls, [])) for cls in train_classes}

    summary["splits"] = {
        "train": {"total": sum(train_counts.values()), "per_class": train_counts},
        "validation": {"total": sum(val_counts.values()), "per_class": val_counts},
        "test": {"total": sum(test_counts.values()), "per_class": test_counts}
    }
    summary["total_images"] = (
        summary["splits"]["train"]["total"] +
        summary["splits"]["validation"]["total"] +
        summary["splits"]["test"]["total"]
    )

    # Class balance analysis in training set
    counts = [c for c in train_counts.values() if c > 0]
    if counts:
        min_c, max_c = min(counts), max(counts)
        ratio = max_c / min_c if min_c > 0 else float("inf")
        summary["class_balance"] = {
            "min_class_count": min_c,
            "max_class_count": max_c,
            "imbalance_ratio": round(ratio, 2)
        }
        if ratio > 3.0:
            summary["warnings"].append(
                f"Class imbalance detected (ratio {ratio:.1f}:1). Consider weighting or augmentation."
            )

    # Data leakage check
    leakage = check_dataset_leakage(train_imgs, val_imgs, test_imgs)
    summary["leakage_warnings"] = leakage

    if output_summary_path:
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary written to {output_summary_path}")

    return summary


def get_data_augmentation_pipeline():
    """
    Returns TensorFlow Data Augmentation Sequential layer block.
    Applied exclusively to the training dataset.
    """
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", name="aug_random_flip"),
        tf.keras.layers.RandomRotation(0.12, fill_mode="nearest", name="aug_random_rotation"),
        tf.keras.layers.RandomZoom(0.10, fill_mode="nearest", name="aug_random_zoom"),
        tf.keras.layers.RandomContrast(0.10, name="aug_random_contrast")
    ], name="data_augmentation")


def create_datasets(
    dataset_dir: Path,
    img_height: int = 224,
    img_width: int = 224,
    batch_size: int = 16,
    seed: int = 42
):
    """
    Builds optimized tf.data.Dataset pipelines for train, val, and test splits.
    Applies MobileNetV2 preprocessing (scaled to [-1, 1]) and training augmentation.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training split missing at: {train_dir}")

    logger.info("Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {class_names}")

    val_ds = None
    if val_dir.exists():
        logger.info("Loading validation dataset...")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False
        )

    test_ds = None
    if test_dir.exists():
        logger.info("Loading test dataset...")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False
        )

    augmentation = get_data_augmentation_pipeline()

    # Preprocessing wrapper using tf.py_function or standard mapping
    def preprocess_train(x, y):
        x = augmentation(x, training=True)
        x = preprocess_input(x)
        return x, y

    def preprocess_val_or_test(x, y):
        x = preprocess_input(x)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_processed = (
        train_ds
        .map(preprocess_train, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size=AUTOTUNE)
    )

    val_ds_processed = None
    if val_ds is not None:
        val_ds_processed = (
            val_ds
            .map(preprocess_val_or_test, num_parallel_calls=AUTOTUNE)
            .prefetch(buffer_size=AUTOTUNE)
        )

    test_ds_processed = None
    if test_ds is not None:
        test_ds_processed = (
            test_ds
            .map(preprocess_val_or_test, num_parallel_calls=AUTOTUNE)
            .prefetch(buffer_size=AUTOTUNE)
        )

    return train_ds_processed, val_ds_processed, test_ds_processed, class_names

