"""
Generate combined model comparison and ablation study figures.

Reads structured results from JSON files produced by baseline.py and advanced.py.
Falls back to hardcoded values only if JSON files are not available yet.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    """Load JSON file if it exists, else return None."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


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


def _load_all_results(outputs_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load baseline + advanced results from JSON, building plot-ready dicts."""
    all_models = []
    ablation = []

    # Baseline results
    bl = _load_json(outputs_dir / "baseline_results.json")
    if bl:
        for m in bl["models"]:
            all_models.append({"model_name": m["model_name"] + "\n(Handcrafted)",
                               "summary": m["summary"]})
        print(f"  Loaded {len(bl['models'])} baseline results from JSON")

    # Advanced results
    adv = _load_json(outputs_dir / "advanced_results.json")
    if adv:
        for m in adv["models"]:
            all_models.append({"model_name": m["model_name"], "summary": m["summary"]})
        for a in adv.get("ablation", []):
            ablation.append({"model_name": a["model_name"].replace("Ablation: ", ""),
                             "summary": a["summary"]})
        print(f"  Loaded {len(adv['models'])} advanced + {len(ablation)} ablation results from JSON")

    if not all_models:
        print("  WARNING: No JSON results found -- run baseline.py and advanced.py first")

    return all_models, ablation


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
    ax.set_title("Model Comparison -- 5-Fold Cross-Validation", fontsize=14, fontweight="bold")
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
    ax.set_title("Feature Ablation Study -- XGBoost 5-Fold CV", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="grey", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation study plot saved to {save_path}")


if __name__ == "__main__":
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    figures_dir = outputs_dir / "figures"

    all_models, ablation = _load_all_results(outputs_dir)

    if all_models:
        plot_model_comparison(all_models, figures_dir / "model_comparison.png")
    if ablation:
        plot_ablation_study(ablation, figures_dir / "ablation_study.png")

    print("All plots generated.")
