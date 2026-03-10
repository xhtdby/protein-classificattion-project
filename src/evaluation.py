"""
Evaluation utilities: metrics computation, confusion matrix plotting, comparison tables.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    classification_report,
)

from src.data_loading import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the four required metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    save_path: Path | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


def print_metrics_table(results_list: list[dict]) -> None:
    """Print a formatted comparison table for multiple model results."""
    header = f"{'Model':<30s} {'Accuracy':>10s} {'Macro F1':>10s} {'Bal. Acc.':>10s} {'MCC':>10s}"
    print(f"\n{'='*72}")
    print(header)
    print(f"{'-'*72}")
    for r in results_list:
        s = r["summary"]
        name = r["model_name"]
        print(
            f"{name:<30s} "
            f"{s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.3f} "
            f"{s['macro_f1_mean']:.4f}+/-{s['macro_f1_std']:.3f} "
            f"{s['balanced_accuracy_mean']:.4f}+/-{s['balanced_accuracy_std']:.3f} "
            f"{s['mcc_mean']:.4f}+/-{s['mcc_std']:.3f}"
        )
    print(f"{'='*72}")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print sklearn classification report with class names."""
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


def plot_metrics_comparison(
    results_list: list[dict],
    save_path: Path | None = None,
) -> None:
    """Bar chart comparing models across the four metrics."""
    metric_keys = ["accuracy_mean", "macro_f1_mean", "balanced_accuracy_mean", "mcc_mean"]
    metric_labels = ["Accuracy", "Macro F1", "Balanced Acc.", "MCC"]

    model_names = [r["model_name"] for r in results_list]
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results_list):
        vals = [r["summary"][k] for k in metric_keys]
        errs = [r["summary"][k.replace("_mean", "_std")] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=r["model_name"], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- 5-Fold CV Metrics")
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metrics comparison saved to {save_path}")
    plt.close(fig)


def plot_class_distribution(labels: np.ndarray, save_path: Path | None = None) -> None:
    """Bar chart of class sizes."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(CLASS_NAMES))
    bars = ax.bar(CLASS_NAMES, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_yscale("log")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Class distribution plot saved to {save_path}")
    plt.close(fig)


def plot_sequence_length_distribution(
    lengths: np.ndarray,
    labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Per-class histogram of sequence lengths."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_lengths = lengths[labels == cls_idx]
        ax.hist(cls_lengths, bins=50, alpha=0.5, label=f"{cls_name} (n={len(cls_lengths)})")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Length Distribution by Class")
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sequence length distribution saved to {save_path}")
    plt.close(fig)
