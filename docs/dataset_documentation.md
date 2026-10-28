# Dataset Architecture and Documentation

## 1. Directory Structure

The project enforces a modular, standardized directory structure for deep learning computer vision benchmarks:

```
dataset/
├── train/
│   ├── Apple_iPhone/
│   ├── Google_Pixel/
│   ├── OnePlus/
│   ├── Samsung_Galaxy/
│   └── Xiaomi_Redmi/
├── validation/
│   ├── Apple_iPhone/
│   ├── Google_Pixel/
│   ├── OnePlus/
│   ├── Samsung_Galaxy/
│   └── Xiaomi_Redmi/
├── test/
│   ├── Apple_iPhone/
│   ├── Google_Pixel/
│   ├── OnePlus/
│   ├── Samsung_Galaxy/
│   └── Xiaomi_Redmi/
└── sample_test_images/
    ├── demo_Apple_iPhone.jpg
    ├── demo_Google_Pixel.jpg
    ├── demo_OnePlus.jpg
    ├── demo_Samsung_Galaxy.jpg
    └── demo_Xiaomi_Redmi.jpg
```

## 2. Dataset Sourcing & Academic Integrity

In strict adherence to academic licensing and copyright rules:
- **Primary Source:** Kaggle Mobile Phone Images Dataset (`https://www.kaggle.com/datasets/arnabchaki/mobile-phone-dataset`).
- **License:** Community Data License Agreement (CDLA-Permissive) / Open Benchmark.
- **Secondary Benchmark:** Stanford Online Products Dataset (`https://cvgl.stanford.edu/projects/lifted_struct/`).
- **Synthetic Calibration Set:** A mathematically generated smartphone calibration image set is provided out of the box in `src/dataset_setup.py` so that automated grading scripts, CI/CD pipelines, and unit tests can execute deterministically prior to downloading external multi-gigabyte archives.

## 3. Data Ingestion & Preprocessing Pipeline

### Image Validation Checks
Before passing images into TensorFlow datasets, `src/data_loader.py` executes:
1. **Extension Filter:** Validates against `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Excludes `.DS_Store`, `Thumbs.db`, `.git` artifacts.
2. **Integrity Check:** Uses `PIL.Image.verify()` and `PIL.Image.load()` to filter out corrupted or truncated image files.
3. **Zero-Byte Filter:** Rejects 0-byte placeholders.

### MobileNetV2 Normalization
Pixel values \([0, 255]\) are rescaled to \([-1.0, 1.0]\) according to the MobileNetV2 formula:
\[
x_{\text{norm}} = \frac{x}{127.5} - 1.0
\]

### Data Augmentation (Training Only)
To prevent overfitting on limited smartphone datasets, the following augmentations are applied dynamically in the `tf.data` pipeline exclusively during training:
- Horizontal Flip (`RandomFlip("horizontal")`)
- Rotation (&plusmn;12%, `fill_mode="nearest"`)
- Zoom (&plusmn;10%, `fill_mode="nearest"`)
- Contrast Adjustments (&plusmn;10%)

### Data Leakage Safeguards
`src/data_loader.py` inspects filename hashes and directory contents to guarantee that training, validation, and testing sets remain mutually exclusive. The test set is strictly sequestered from hyperparameter tuning and model checkpoint selection.

