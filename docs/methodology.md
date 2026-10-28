# Methodology: Mobile Phone Model Classification Using Deep Learning

## 1. Research Formulation & Overview
Computer vision-based smartphone identification represents a fine-grained image classification challenge. Unlike generic object detection (distinguishing a phone from a laptop or coffee cup), fine-grained classification requires recognizing subtle inter-class visual variations across mobile brands (Apple, Samsung, Xiaomi, Google Pixel, OnePlus) and distinct phone models while accommodating substantial intra-class variance caused by lighting, angles, device colorways, and protective phone cases.

## 2. Transfer Learning Rationale
Training deep convolutional neural networks from scratch typically demands tens of thousands of labeled images per category to learn low-level spatial primitives (Gabor-like edge detectors, texture filters, color gradients).

By employing **Transfer Learning** with a backbone pretrained on ImageNet-1K (comprising over 1.4 million labeled images across 1,000 object categories), the model inherits generalized feature representations:
- **Early Layers (1–40):** Universal spatial primitives (edges, contours, corners, surface textures).
- **Middle Layers (41–100):** Generic geometric structures (curves, circles, concentric rings, rectangular boundaries).
- **Upper Layers (101–154):** High-level semantic features (camera modules, punch holes, brand-specific chassis bevels).

## 3. Architecture Selection: MobileNetV2
MobileNetV2 was selected as the optimal architecture for this semester project over heavier alternatives (ResNet-50, VGG-16, EfficientNet-B4) based on the following engineering criteria:

1. **Computational Efficiency:** 3.4M parameters compared to ResNet-50's 25.6M parameters.
2. **Inference Latency:** &lt; 50ms CPU execution time, enabling real-time client inference on standard hardware and university laboratory machines.
3. **Inverted Residual Topology:** Information bottlenecks preserve manifold geometry by expanding low-dimensional representations to higher-dimensional spaces before applying depthwise separable convolutions.
4. **Linear Bottlenecks:** Avoiding non-linear activations (ReLU) in narrow bottleneck projections prevents the destruction of manifold representations.

## 4. End-to-End Processing Workflow

```
[Uploaded Image]
       │
       ▼
[PIL Validation & RGB Normalization] ──> Reject Corrupted / Non-Images
       │
       ▼
[Bicubic Resizing: 224 x 224]
       │
       ▼
[MobileNetV2 Scaling: [-1, 1]]
       │
       ▼
[Pretrained MobileNetV2 Backbone] (Frozen in Phase 1 / Fine-Tuned in Phase 2)
       │
       ▼
[GlobalAveragePooling2D] ──> Condenses 7x7x1280 feature maps to 1280 vector
       │
       ▼
[Batch Normalization & Dropout (0.3)]
       │
       ▼
[Dense Layer (128 units, ReLU) & Dropout (0.2)]
       │
       ▼
[Dense Softmax Layer (num_classes)] ──> Categorical Probability Distribution
```

## 5. Training Protocol

### Phase 1: Feature Extraction (Classification Head Training)
- Pretrained MobileNetV2 backbone weights are frozen (`base_model.trainable = False`).
- Only newly initialized weights in the custom head are optimized.
- Optimizer: Adam (`learning_rate = 1e-3`).
- Loss Function: Categorical Crossentropy.
- Batch Size: 16.
- Prevents disruptive large gradients from distorting pretrained ImageNet weights.

### Phase 2: Fine-Tuning
- Backbone weights from layer 100 onwards are unfrozen (`fine_tune_from_layer = 100`).
- Batch Normalization layers remain frozen in inference mode (`layer.trainable = False`) to prevent moving mean and variance estimates from destabilizing on small target batches.
- Optimizer: Adam with conservative learning rate (`learning_rate = 1e-5`).
- Enables upper convolutional filters to adapt to smartphone-specific geometries (lens arrays, camera visors).

