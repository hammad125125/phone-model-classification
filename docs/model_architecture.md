# MobileNetV2 Model Architecture Specification

## 1. Mathematical Formulation of Inverted Residuals

Standard convolutional networks process inputs through sequential convolution filters with high computational complexity:
\[
\text{Cost}_{\text{standard}} = h \cdot w \cdot d_{\text{in}} \cdot d_{\text{out}} \cdot k^2
\]
where \(h, w\) are spatial dimensions, \(d_{\text{in}}\) is input channel depth, \(d_{\text{out}}\) is output channel depth, and \(k\) is kernel size (typically 3).

### Depthwise Separable Convolutions
MobileNetV2 decomposes this operation into two distinct stages:
1. **Depthwise Convolution:** A single \(k \times k\) filter applied per channel independently:
   \[
   \text{Cost}_{\text{depthwise}} = h \cdot w \cdot d_{\text{in}} \cdot k^2
   \]
2. **Pointwise Convolution:** A \(1 \times 1\) convolution linearly combining channel outputs:
   \[
   \text{Cost}_{\text{pointwise}} = h \cdot w \cdot d_{\text{in}} \cdot d_{\text{out}}
   \]
The total computational reduction ratio is:
\[
\frac{\text{Cost}_{\text{separable}}}{\text{Cost}_{\text{standard}}} = \frac{1}{d_{\text{out}}} + \frac{1}{k^2} \approx \frac{1}{9}
\]

### Linear Bottlenecks
MobileNetV2 uses linear bottlenecks in narrow layers. Non-linear activations like ReLU threshold negative values to 0, which can destroy manifold features when channel dimensions are low. Linear activation at the bottleneck projection preserves information capacity.

---

## 2. Project Classification Head Design

The custom head appended to MobileNetV2 is configured as follows:

```python
# Feature extraction from 7x7x1280 base output
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_output)
x = layers.BatchNormalization(name="head_batch_norm")(x)
x = layers.Dropout(0.3, name="head_dropout_1")(x)
x = layers.Dense(128, activation="relu", name="head_dense")(x)
x = layers.Dropout(0.2, name="head_dropout_2")(x)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="classification_output")(x)
```

### Purpose of Head Layers
- **GlobalAveragePooling2D:** Reduces spatial \(7 \times 7 \times 1280\) feature maps to a 1D vector of length 1280, drastically reducing parameter count compared to a Flatten layer and providing spatial translation invariance.
- **Batch Normalization:** Stabilizes feature variance entering the dense projection.
- **Dropout (0.3 & 0.2):** Prevents co-adaptation of hidden activations on training samples.
- **Dense(128, ReLU):** Adds task-specific non-linear capacity to model device features.
- **Dense(num_classes, Softmax):** Normalizes activations into a true categorical probability distribution \(\sum_{i} p_i = 1.0\).

---

## 3. Parameter Summary

| Component | Layer Count | Parameters | Trainable in Phase 1 | Trainable in Phase 2 |
|-----------|-------------|------------|----------------------|----------------------|
| MobileNetV2 Base | 154 | ~2,257,984 | Frozen (0) | ~1,200,000 (layers &ge; 100) |
| GAP + BatchNorm | 2 | ~2,560 | 2,560 | 2,560 (inference BN) |
| Dense Head (128) | 1 | ~163,968 | 163,968 | 163,968 |
| Softmax Head (5) | 1 | ~645 | 645 | 645 |
| **Total Model** | -- | **~2,425,157** | **~167,173** | **~1,367,173** |

