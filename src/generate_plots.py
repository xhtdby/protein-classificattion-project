"""
Generate combined model comparison and ablation study figures.

Uses hardcoded results from completed CV runs for reproducibility.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def make_result(name: str, acc_m: float, acc_s: float, f1_m: float, f1_s: float,
                ba_m: float, ba_s: float, mcc_m: float, mcc_s: float) -> dict:
    return {
        "model_name": name,
        "summary": {
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
            "macro_f1_mean": f1_m, "macro_f1_std": f1_s,
            "balanced_accuracy_mean": ba_m, "balanced_accuracy_std": ba_s,
            "mcc_mean": mcc_m, "mcc_std": mcc_s,
        },
    }


# ---- All model results from completed runs ----
ALL_RESULTS = [
    make_result("Logistic Regression\n(Handcrafted)",
                0.4573, 0.003, 0.1912, 0.002, 0.2997, 0.010, 0.1710, 0.003),
    make_result("Random Forest\n(Handcrafted)",
                0.8149, 0.000, 0.1283, 0.000, 0.1428, 0.000, -0.0032, 0.002),
    make_result("XGBoost\n(ESM-2)",
                0.8670, 0.004, 0.4655, 0.018, 0.3937, 0.012, 0.5223, 0.013),
    make_result("LightGBM\n(ESM-2)",
                0.8668, 0.003, 0.5162, 0.015, 0.4807, 0.013, 0.5756, 0.009),
    make_result("XGBoost+SMOTE\n(ESM-2)",
                0.8492, 0.005, 0.5269, 0.012, 0.5405, 0.010, 0.5685, 0.008),
    make_result("LightGBM+SMOTE\n(ESM-2)",
                0.8538, 0.004, 0.5284, 0.014, 0.5259, 0.015, 0.5689, 0.010),
]

# ---- Ablation results ----
ABLATION_RESULTS = [
    make_result("ESM-2 only",       0.8670, 0.004, 0.4655, 0.018, 0.3937, 0.012, 0.5223, 0.013),
    make_result("Handcrafted only", 0.8183, 0.003, 0.1679, 0.009, 0.1643, 0.009, 0.1627, 0.011),
    make_result("ESM-2 +\nHandcrafted", 0.8629, 0.004, 0.4314, 0.013, 0.3609, 0.009, 0.4980, 0.012),
    make_result("ESM-2 +\nPhysicochemical", 0.8682, 0.003, 0.4718, 0.017, 0.3987, 0.012, 0.5275, 0.012),
]


def plot_model_comparison(results: list[dict], save_path: Path) -> None:
    """Grouped bar chart comparing all models across four metrics."""
    metric_keys = ["accuracy_mean", "macro_f1_mean", "balanced_accuracy_mean", "mcc_mean"]
    metric_labels = ["Accuracy", "Macro F1", "Balanced Acc.", "MCC"]

    model_names = [r["model_name"] for r in results]
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = sns.color_palette("Set2", n_models)

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, r in enumerate(results):
        vals = [r["summary"][k] for k in metric_keys]
        errs = [r["summary"][k.replace("_mean", "_std")] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=r["model_name"],
               color=colors[i], capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.axhline(y=0, color="grey", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Model comparison plot saved to {save_path}")


def plot_ablation_study(results: list[dict], save_path: Path) -> None:
    """Grouped bar chart for ablation study."""
    metric_keys = ["accuracy_mean", "macro_f1_mean", "balanced_accuracy_mean", "mcc_mean"]
    metric_labels = ["Accuracy", "Macro F1", "Balanced Acc.", "MCC"]

    model_names = [r["model_name"] for r in results]
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = sns.color_palette("muted", n_models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results):
        vals = [r["summary"][k] for k in metric_keys]
        errs = [r["summary"][k.replace("_mean", "_std")] for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=r["model_name"],
               color=colors[i], capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Feature Ablation Study — XGBoost 5-Fold CV", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="grey", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation study plot saved to {save_path}")


if __name__ == "__main__":
    figures_dir = Path(__file__).resolve().parent.parent / "outputs" / "figures"
    plot_model_comparison(ALL_RESULTS, figures_dir / "model_comparison.png")
    plot_ablation_study(ABLATION_RESULTS, figures_dir / "ablation_study.png")
    print("All plots generated.")
