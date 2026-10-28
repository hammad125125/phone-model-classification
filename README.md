# Mobile Phone Model Classification Using Deep Learning and Transfer Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Status](https://img.shields.io/badge/Project-Semester%20Deep%20Learning-green.svg)]()

An end-to-end Computer Vision and Deep Learning system designed to identify smartphone brands and models from uploaded photos. Built using **MobileNetV2** pretrained on ImageNet with custom inverted residual fine-tuning, a production-ready Flask REST API, an interactive web UI, comprehensive automated testing, and evaluation metrics.

---

## 1. System Architecture

```
[Uploaded Image]
       │
       ▼
[PIL Validation & RGB Normalization] ──> Filter Corrupt Files
       │
       ▼
[Bicubic Resizing: 224 x 224] ──> MobileNet Normalization [-1, 1]
       │
       ▼
[MobileNetV2 Pretrained Backbone] ──> 154 Layers (ImageNet Weights)
       │
       ▼
[GlobalAveragePooling2D] ──> Spatial Tensor Condensation (1280)
       │
       ▼
[Batch Normalization & Dropout (0.3)]
       │
       ▼
[Dense Head (128 units, ReLU) & Dropout (0.2)]
       │
       ▼
[Softmax Classification Output (num_classes)] ──> Top-3 Class Probabilities
```

---

## 2. Directory Structure

```
mobile-classification-project/
├── app.py                      # Flask web application and REST API server
├── train.py                    # Complete two-stage model training pipeline
├── evaluate.py                 # Independent test set evaluation script
├── predict.py                  # CLI inference utility
├── requirements.txt            # Project dependencies
├── Dockerfile                  # Containerized deployment specification
├── vercel.json                 # Frontend & serverless routing configuration
├── .gitignore                  # Git exclusions for environments and caches
├── .env.example                # Example environment variable template
├── config/
│   ├── __init__.py
│   └── config.py               # Central project parameters and paths
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion, tf.data pipelines, leakage checks
│   ├── model.py                # MobileNetV2 architecture, unfreezing & loading
│   ├── evaluator.py            # Confusion matrix & performance metric calculation
│   ├── inference.py            # Singleton in-memory inference engine
│   └── dataset_setup.py        # Dataset structure verification & sample generator
├── dataset/
│   ├── train/                  # Training images partitioned by class
│   ├── validation/             # Validation images partitioned by class
│   ├── test/                   # Independent test split for evaluation
│   └── sample_test_images/     # Legally usable sample images for quick testing
├── models/
│   ├── mobile_model.weights.h5 # Trained model weights
│   ├── class_names.json        # Class-to-index mapping
│   ├── model_config.json       # Architecture parameters
│   ├── training_config.json    # Optimizer and training metadata
│   └── metrics.json            # Quantitative evaluation metrics
├── results/
│   ├── confusion_matrix.png    # Evaluated confusion matrix heatmap
│   ├── training_history.png    # Accuracy and loss curves across epochs
│   ├── classification_report.json # Per-class precision, recall, and F1
│   └── evaluation_summary.txt  # Human-readable evaluation summary
├── templates/                  # Responsive HTML5 web templates
│   ├── base.html
│   ├── index.html              # Home overview & architecture highlights
│   ├── demo.html               # Interactive upload & top-3 prediction preview
│   ├── dataset.html            # Data distribution, sources & licensing
│   ├── model.html              # MobileNetV2 deep dive & metrics
│   ├── docs.html               # Academic methodology & limitations
│   └── resources.html          # Academic report, slides & repo placeholders
├── static/
│   ├── css/main.css            # Custom AI tech design system
│   └── js/main.js              # Drag-and-drop & API inference client
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Preprocessing & normalization tests
│   ├── test_model.py           # Model architecture & weight loading tests
│   ├── test_api.py             # Flask endpoints & validation tests
│   └── test_dataset.py         # Leakage detection & split checks
└── docs/                       # Academic reports & presentation guides
    ├── methodology.md
    ├── dataset_documentation.md
    ├── model_architecture.md
    ├── training_and_evaluation.md
    ├── deployment_guide.md
    ├── project_report_outline.md
    └── presentation_outline.md
```

---

## 3. Quickstart & Execution Instructions

### A. Environment Setup (Windows PowerShell)

```powershell
# 1. Clone repository
git clone <your-repository-url>
cd "mobile classification project"

# 2. Create Python 3.10 virtual environment
py -3.10 -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B. Dataset Preparation

1. **Option 1: Built-in Verification Benchmark:**
   The project includes a mathematical calibration dataset generator for immediate verification:
   ```powershell
   python -c "from src.dataset_setup import ensure_dataset_structure, create_verification_sample_dataset; from pathlib import Path; p = Path('dataset'); ensure_dataset_structure(p); create_verification_sample_dataset(p, samples_per_class=20)"
   ```

2. **Option 2: External Open-Access Smartphone Dataset:**
   - Download the [Kaggle Mobile Phone Dataset](https://www.kaggle.com/datasets/arnabchaki/mobile-phone-dataset) (CDLA-Permissive license).
   - Place raw brand/model folders inside `dataset/raw/`.
   - Partition into 70% Train, 15% Validation, 15% Test:
     ```powershell
     python -c "from src.dataset_setup import split_raw_dataset; from pathlib import Path; split_raw_dataset(Path('dataset/raw'), Path('dataset'))"
     ```

### C. Model Training

Execute the two-stage training pipeline (Phase 1: Feature Extraction, Phase 2: Fine-Tuning):

```powershell
python train.py --epochs 15 --fine-tune-epochs 10 --batch-size 16
```

### D. Model Evaluation

Evaluate the saved weights strictly on the independent `dataset/test` partition:

```powershell
python evaluate.py --test-dir dataset/test
```

Evaluation outputs are automatically stored in `results/`:
- `results/confusion_matrix.png`
- `results/training_history.png`
- `results/classification_report.json`
- `results/evaluation_summary.txt`

### E. CLI Single-Image Prediction

```powershell
python predict.py --image dataset/sample_test_images/demo_Apple_iPhone.jpg --top-k 3
```

### F. Running the Web Application

```powershell
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 4. REST API Specification

### 1. `POST /api/predict`
Accepts multipart form-data image upload.
- **Request Body:** `file` (image file: `.jpg`, `.png`, `.webp`, max 16MB)
- **Response Format:**
```json
{
  "success": true,
  "predicted_class": "Apple_iPhone",
  "confidence": 0.9412,
  "confidence_percentage": 94.12,
  "top_predictions": [
    { "class": "Apple_iPhone", "confidence": 0.9412, "confidence_percentage": 94.12 },
    { "class": "Samsung_Galaxy", "confidence": 0.0381, "confidence_percentage": 3.81 },
    { "class": "Xiaomi_Redmi", "confidence": 0.0124, "confidence_percentage": 1.24 }
  ],
  "inference_time_ms": 38.45,
  "model_status": "trained",
  "confidence_disclaimer": "Confidence represents the softmax probability output across learned classes."
}
```

### 2. `GET /api/model-status`
Returns model readiness, number of classes, input dimensions, and loaded state.

### 3. `GET /api/health`
System diagnostic health check.

---

## 5. Automated Unit & Integration Testing

Execute the test suite with pytest:

```powershell
pytest -v
```

Tests cover:
- Image resizing, bicubic interpolation, and normalization to `[-1, 1]`
- RGBA and Grayscale automatic 3-channel RGB conversion
- Handling of non-image and corrupted byte streams
- Zero-leakage verification between dataset splits
- Deterministic MobileNetV2 architecture compilation and shape checking
- REST API endpoint error handling and responses

---

## 6. Academic Integrity Statement

- **Pretrained Baseline:** MobileNetV2 with ImageNet weights serves solely as the starting feature extractor.
- **Project Adaptation:** All final classification weights, custom dense head parameters, and metrics are derived from the project-specific training and test pipeline.
- **No Hallucinated Metrics:** Accuracy, precision, recall, and F1 scores are computed only from genuine evaluation runs on isolated test data. If the model has not yet completed training, it is explicitly flagged as `untrained` across the UI and API.

