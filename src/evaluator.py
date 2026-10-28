"""
Model Evaluation and Metrics Visualization Module
Mobile Phone Model Classification Using Deep Learning and Transfer Learning
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments/servers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_training_history(
    history: Dict[str, List[float]],
    output_png_path: Path
) -> None:
    """
    Renders and saves high-resolution loss and accuracy curves across epochs.
    """
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("accuracy", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Accuracy Plot
    ax1.plot(epochs, history.get("accuracy", []), "b-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history and history["val_accuracy"]:
        ax1.plot(epochs, history.get("val_accuracy", []), "g-s", label="Validation Accuracy", linewidth=2)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs, history.get("loss", []), "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        ax2.plot(epochs, history.get("val_loss", []), "m-s", label="Validation Loss", linewidth=2)
    ax2.set_title("Categorical Crossentropy Loss", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(str(output_png_path), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training history curves to: {output_png_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_png_path: Path,
    normalize: bool = True
) -> None:
    """
    Generates and saves a publication-quality confusion matrix heatmap.
    """
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    data_to_plot = cm_norm if normalize else cm
    cax = ax.matshow(data_to_plot, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    # Set labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            val_norm = cm_norm[i, j]
            text = f"{val}\n({val_norm:.1%})" if normalize else f"{val}"
            color = "white" if data_to_plot[i, j] > (data_to_plot.max() / 2.0) else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix (Smartphone Classification)", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(str(output_png_path), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix heatmap to: {output_png_path}")


def evaluate_model_on_dataset(
    model,
    test_ds,
    class_names: List[str],
    results_dir: Path,
    metrics_path: Optional[Path] = None
) -> Dict:
    """
    Computes strict metrics on an independent test dataset:
    - Overall Accuracy, Precision, Recall, F1 (macro and weighted)
    - Full classification report
    - Confusion Matrix PNG
    - Generates JSON and human-readable text summaries
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Executing model inference across independent test dataset...")
    y_true_list = []
    y_pred_list = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true_indices = np.argmax(batch_y.numpy(), axis=1)
        y_pred_indices = np.argmax(preds, axis=1)

        y_true_list.extend(y_true_indices)
        y_pred_list.extend(y_pred_indices)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    if len(y_true) == 0:
        raise ValueError("Test dataset yielded 0 evaluation samples.")

    # Calculate standard metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "evaluation_dataset_size": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "classes": class_names,
        "confusion_matrix": cm,
        "classification_report": report_dict
    }

    # Plot Confusion Matrix
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

    # Save Classification Report JSON
    report_json_path = results_dir / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save to models directory if requested
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Write human-readable text summary
    summary_txt_path = results_dir / "evaluation_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("MOBILE PHONE CLASSIFICATION - MODEL EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Samples:        {len(y_true)}\n")
        f.write(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Macro Precision:     {prec_macro:.4f}\n")
        f.write(f"Macro Recall:        {rec_macro:.4f}\n")
        f.write(f"Macro F1-Score:      {f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score:   {f1_weighted:.4f}\n")
        f.write("-" * 60 + "\n")
        f.write("PER-CLASS CLASSIFICATION REPORT:\n")
        for cls in class_names:
            cls_stat = report_dict.get(cls, {})
            p = cls_stat.get("precision", 0.0)
            r = cls_stat.get("recall", 0.0)
            f1 = cls_stat.get("f1-score", 0.0)
            sup = cls_stat.get("support", 0)
            f.write(f"  {cls:<20} Prec: {p:.3f} | Rec: {r:.3f} | F1: {f1:.3f} | Support: {sup}\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Evaluation complete. Accuracy: {acc*100:.2f}%. Saved to {results_dir}")
    return metrics

