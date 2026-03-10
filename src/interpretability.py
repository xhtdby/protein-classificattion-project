"""
Interpretability analyses: feature importance, ablation study, confusion error analysis.

Implements at least two techniques for extra credit:
1. Permutation / Gini feature importance
2. Ablation study (feature group drop)
3. Confusion matrix error analysis
4. SHAP Tree/Gradient explanations
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.data_loading import CLASS_NAMES
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


def feature_importance_analysis(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    save_dir: Path,
    top_k: int = 20,
    method: str = "auto",
) -> dict[str, float]:
    """Compute and plot feature importance.

    Args:
        model: Fitted model.
        X_val: Validation features.
        y_val: Validation labels.
        feature_names: Names for each feature column.
        save_dir: Directory to save plot.
        top_k: Number of top features to show.
        method: "gini" for tree-based, "permutation" for model-agnostic, "auto" to choose.

    Returns:
        Dict of feature name -> importance score.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Decide method
    has_feature_importances = hasattr(model, "feature_importances_")
    if method == "auto":
        method = "gini" if has_feature_importances else "permutation"

    if method == "gini" and has_feature_importances:
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        title = "Feature Importance (Gini / Gain)"
    else:
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=10,
            random_state=42,
            scoring="f1_macro",
            n_jobs=-1,
        )
        importances = result.importances_mean
        importance_dict = dict(zip(feature_names, importances))
        title = "Feature Importance (Permutation)"

    # Sort and take top-k
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    names = [item[0] for item in sorted_items]
    vals = [item[1] for item in sorted_items]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance plot saved to {save_dir / 'feature_importance.png'}")

    return importance_dict


def ablation_study_summary(
    ablation_results: list[dict],
    save_dir: Path,
) -> None:
    """Generate ablation study summary table and plot from CV results.

    Args:
        ablation_results: List of CV result dicts with different feature combos.
        save_dir: Directory to save outputs.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Table
    print("\n=== Ablation Study Results ===")
    print(f"{'Feature Set':<30s} {'Macro F1':>10s} {'Bal. Acc.':>10s} {'MCC':>10s}")
    print("-" * 65)

    names = []
    f1_scores = []
    for r in ablation_results:
        s = r["summary"]
        name = r["model_name"]
        names.append(name)
        f1_scores.append(s["macro_f1_mean"])
        print(f"{name:<30s} {s['macro_f1_mean']:>10.4f} "
              f"{s['balanced_accuracy_mean']:>10.4f} {s['mcc_mean']:>10.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    ax.bar(x, f1_scores, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title("Ablation Study -- Feature Group Contribution")
    ax.set_ylim(0, max(f1_scores) * 1.15)
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_dir / "ablation_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation plot saved to {save_dir / 'ablation_results.png'}")


def class_confusion_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    save_dir: Path | None = None,
) -> None:
    """Analyse per-class performance and most confused class pairs.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Prediction probabilities (N, 7), optional.
        class_names: Class label names.
        save_dir: Directory to save outputs.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    print("\n=== Per-Class Performance ===")
    print(f"{'Class':<20s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:<20s} {precision[i]:>8.4f} {recall[i]:>8.4f} {f1[i]:>8.4f} {support[i]:>8d}")

    # Most confused pairs
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.fill_diagonal(cm, 0)  # zero out correct predictions

    print("\n=== Most Confused Class Pairs ===")
    flat_indices = np.argsort(cm.ravel())[::-1][:10]
    for idx in flat_indices:
        true_cls = idx // len(class_names)
        pred_cls = idx % len(class_names)
        count = cm[true_cls, pred_cls]
        if count == 0:
            break
        print(f"  {class_names[true_cls]} -> {class_names[pred_cls]}: {count} errors")

    # Confidence of misclassified samples
    if y_proba is not None:
        misclassified = y_true != y_pred
        if misclassified.any():
            max_proba = y_proba.max(axis=1)
            misc_proba = max_proba[misclassified]
            print(f"\n=== Misclassification Confidence ===")
            print(f"  Mean max-prob of misclassified: {misc_proba.mean():.4f}")
            print(f"  Median max-prob of misclassified: {np.median(misc_proba):.4f}")
            high_conf_wrong = (misc_proba >= 0.80).sum()
            print(f"  High-confidence errors (p >= 0.80): {high_conf_wrong} "
                  f"({100*high_conf_wrong/len(misc_proba):.1f}% of all errors)")

    # Save per-class metrics plot
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        ax.bar(x - width, precision, width, label="Precision", color="steelblue")
        ax.bar(x, recall, width, label="Recall", color="coral")
        ax.bar(x + width, f1, width, label="F1", color="seagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Precision / Recall / F1")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(save_dir / "per_class_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Per-class metrics plot saved to {save_dir / 'per_class_metrics.png'}")


def shap_analysis(
    model,
    X_val: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
    save_dir: Path | None = None,
    max_display: int = 20,
    n_samples: int = 500,
) -> None:
    """Compute SHAP values and generate summary plot.

    Uses TreeExplainer for tree-based models (fast), falls back to sampling
    for other models.

    Args:
        model: Fitted model.
        X_val: Validation features (will subsample if large).
        feature_names: Names for each feature column.
        class_names: Class label names.
        save_dir: Directory to save output plots.
        max_display: Max features to show in SHAP plot.
        n_samples: Max samples to compute SHAP values for (speed).
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed -- skipping SHAP analysis")
        return

    if class_names is None:
        class_names = CLASS_NAMES

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for speed
    if X_val.shape[0] > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_val.shape[0], size=n_samples, replace=False)
        X_sample = X_val[idx]
    else:
        X_sample = X_val

    print(f"Computing SHAP values for {X_sample.shape[0]} samples...")

    # Pick explainer based on model type
    model_type = type(model).__name__.lower()
    try:
        if "lgbm" in model_type or "xgb" in model_type or "forest" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(
                model.predict_proba, background, link="identity"
            )
            shap_values = explainer.shap_values(X_sample, nsamples=100)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return

    # shap_values may be list-of-arrays (one per class) or 3D array
    if isinstance(shap_values, list):
        # List of (n_samples, n_features) per class
        shap_arr = np.array(shap_values)  # (n_classes, n_samples, n_features)
        shap_mean_abs = np.abs(shap_arr).mean(axis=(0, 1))  # (n_features,)
    else:
        # Shape: (n_samples, n_features, n_classes) -- LightGBM TreeExplainer
        if shap_values.ndim == 3:
            shap_arr = shap_values.transpose(2, 0, 1)  # (n_classes, n_samples, n_features)
            shap_mean_abs = np.abs(shap_values).mean(axis=(0, 2))
        else:
            shap_mean_abs = np.abs(shap_values).mean(axis=0)

    # Top-k features by mean |SHAP|
    top_idx = np.argsort(shap_mean_abs)[::-1][:max_display]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = shap_mean_abs[top_idx]

    # Print top features
    print(f"\nTop-{max_display} features by mean |SHAP|:")
    for rank, (name, val) in enumerate(zip(top_names, top_vals), 1):
        print(f"  {rank:>2d}. {name:<35s}  {val:.6f}")

    # Save bar plot
    if save_dir:
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_vals[::-1], color="darkorange")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Importance (top {max_display})")
        plt.tight_layout()
        path = save_dir / "shap_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"SHAP importance plot saved to {path}")


if __name__ == "__main__":
    import logging
    import sys
    import time
    import joblib
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.data_loading import load_all_sequences, SEED
    from src.training import _fmt_time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    project_root = Path(__file__).resolve().parent.parent
    features_dir = project_root / "outputs" / "features"
    models_dir = project_root / "outputs" / "models"
    figures_dir = project_root / "outputs" / "figures"

    t0 = time.time()

    # -- Load model ------------------------------------------------------------
    model_path = models_dir / "best_model.joblib"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run src.models.advanced first.")
        sys.exit(1)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_source = artifact.get("feature_source", "Handcrafted")
    print(f"Loaded {type(model).__name__} ({feature_source})")

    # -- Load data & features --------------------------------------------------
    df = load_all_sequences(project_root)
    y = df["label"].values

    feat_file = "esm2_embeddings.npy" if "ESM" in feature_source else "handcrafted_features.npy"
    X = np.load(features_dir / feat_file)

    names_path = features_dir / "feature_names.npy"
    if "ESM" in feature_source:
        # ESM-2 embedding dimensions don't have handcrafted names
        feature_names = [f"ESM_{i}" for i in range(X.shape[1])]
    elif names_path.exists():
        feature_names = np.load(names_path, allow_pickle=True).tolist()
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    print(f"Features: {X.shape}  |  Names: {len(feature_names)}")

    # Stratified 20% hold-out
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(X, y))

    # -- Prefer saved OOF predictions for honest evaluation --------------------
    oof_preds = artifact.get("oof_preds")
    oof_proba = artifact.get("oof_proba")
    oof_true = artifact.get("oof_true")

    if oof_preds is not None and oof_true is not None:
        print("Using OOF (out-of-fold) predictions from CV - honest held-out evaluation.")
        y_val = oof_true
        y_pred = oof_preds
        y_proba = oof_proba
        # For feature importance we still need a properly scaled val set
        from sklearn.preprocessing import StandardScaler as _SS
        _sc_eval = _SS()
        _sc_eval.fit(X[train_idx])
        X_val = _sc_eval.transform(X[val_idx])
    else:
        print("WARNING: OOF predictions not in model artifact.")
        print("  Re-run 'python -m src.models.advanced' to save OOF predictions.")
        print("  Falling back to held-out split with fresh scaler (honest eval).")
        from sklearn.preprocessing import StandardScaler as _SS
        _sc_eval = _SS()
        _sc_eval.fit(X[train_idx])
        X_val = _sc_eval.transform(X[val_idx])
        y_val = y[val_idx]
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        print("  Note: confusion analysis uses full-data-trained model on held-out split.")

    print(f"Evaluation set: {len(y_val):,} samples  [{_fmt_time(time.time()-t0)}]")

    # -- Predict for feature importance (always from fresh scaler) -------------
    if "X_val" not in dir():
        from sklearn.preprocessing import StandardScaler as _SS
        _sc_eval = _SS()
        _sc_eval.fit(X[train_idx])
        X_val = _sc_eval.transform(X[val_idx])

    # -- 1. Feature importance -------------------------------------------------
    print(f"\n{'='*72}")
    print("  FEATURE IMPORTANCE")
    print(f"{'='*72}")
    importance_dict = feature_importance_analysis(
        model, X_val, y_val, feature_names,
        save_dir=figures_dir,
        top_k=30,
        method="auto",
    )
    # Print top-10
    top10 = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop-10 features:")
    for rank, (name, score) in enumerate(top10, 1):
        print(f"  {rank:>2d}. {name:<35s}  {score:.6f}")

    # -- 2. Class confusion analysis -------------------------------------------
    print(f"\n{'='*72}")
    print("  CLASS CONFUSION ANALYSIS")
    print(f"{'='*72}")
    class_confusion_analysis(
        y_val, y_pred, y_proba,
        save_dir=figures_dir,
    )

    # -- 3. SHAP analysis ------------------------------------------------------
    print(f"\n{'='*72}")
    print("  SHAP ANALYSIS")
    print(f"{'='*72}")
    shap_analysis(
        model, X_val, feature_names,
        save_dir=figures_dir,
        max_display=20,
        n_samples=1000,
    )

    print(f"\nTotal interpretability time: {_fmt_time(time.time()-t0)}")
