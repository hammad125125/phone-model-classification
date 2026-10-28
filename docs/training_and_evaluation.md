# Training & Evaluation Strategy

## 1. Optimization & Hyperparameters

| Hyperparameter | Head Training (Phase 1) | Fine-Tuning (Phase 2) | Rationale |
|----------------|--------------------------|-----------------------|-----------|
| **Optimizer** | Adam (\(\beta_1=0.9, \beta_2=0.999\)) | Adam (\(\beta_1=0.9, \beta_2=0.999\)) | Fast convergence with adaptive momentum |
| **Learning Rate** | \(1 \times 10^{-3}\) | \(1 \times 10^{-5}\) | Low LR in Phase 2 preserves low-level visual features |
| **Loss Function** | Categorical Crossentropy | Categorical Crossentropy | Standard convex surrogate for multi-class classification |
| **Batch Size** | 16 | 16 | Suitable for desktop/laptop CPU/GPU memory footprint |
| **Epochs** | 15 | 10 | Early stopping terminates dynamically |
| **Regularization** | Dropout (0.3 + 0.2) | Dropout + Frozen BN | Prevents co-adaptation and internal covariate shifts |

---

## 2. Callbacks & Checkpointing Protocol

1. **`ModelCheckpoint`:**
   - Monitored Metric: `val_accuracy` (maximization).
   - Mode: `save_best_only=True`, `save_weights_only=True`.
   - Filepath: `models/best_checkpoint.weights.h5`.
   - Ensures that final deployed weights represent the peak validation generalization rather than the final (possibly overfitted) epoch.

2. **`EarlyStopping`:**
   - Monitored Metric: `val_loss` (minimization).
   - Patience: 5 epochs in Phase 1, 4 epochs in Phase 2.
   - `restore_best_weights=True`: Automatically rolls back parameters to the epoch with lowest validation loss if validation error stagnates or diverges.

3. **`ReduceLROnPlateau`:**
   - Monitored Metric: `val_loss`.
   - Factor: 0.3.
   - Patience: 3 epochs.
   - Halts plateau stagnation by reducing learning rate when progress stalls.

---

## 3. Evaluation Metrics on Independent Test Set

Evaluation is performed exclusively on the isolated `dataset/test` partition:

### Quantitative Metrics
1. **Accuracy:** Overall proportion of correctly categorized smartphone images:
   \[
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
   \]
2. **Precision (Macro & Weighted):** Ratio of correct positive predictions to total predicted positives per class:
   \[
   \text{Precision} = \frac{TP}{TP + FP}
   \]
3. **Recall (Macro & Weighted):** Ratio of correctly identified devices to all actual instances of that brand/model:
   \[
   \text{Recall} = \frac{TP}{TP + FN}
   \]
4. **F1-Score (Macro & Weighted):** Harmonic mean balancing precision and recall:
   \[
   \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
   \]
5. **Confusion Matrix Heatmap:** Visualizes inter-class confusion (e.g., misclassification rates between Samsung Galaxy and Google Pixel).

