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


def high_confidence_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
    save_dir: Path | None = None,
) -> None:
    """Break down high-confidence errors by true class and predicted class.

    Answers: which classes produce the most over-confident wrong predictions?
    """
    if class_names is None:
        class_names = CLASS_NAMES

    max_p = y_proba.max(axis=1)
    wrong = y_true != y_pred
    high_conf = max_p >= 0.80
    hc_wrong = wrong & high_conf

    n_hc_wrong = hc_wrong.sum()
    n_wrong = wrong.sum()
    print(f"\n=== High-Confidence Error Analysis ===")
    print(f"  Total errors: {n_wrong}")
    print(f"  High-confidence errors (p>=0.80): {n_hc_wrong} "
          f"({100*n_hc_wrong/max(n_wrong,1):.1f}% of errors)")

    if n_hc_wrong == 0:
        print("  No high-confidence errors to analyse.")
        return

    # Breakdown by true class
    print(f"\n  By TRUE class (what was the sample actually?):")
    for c in range(len(class_names)):
        mask = hc_wrong & (y_true == c)
        n = mask.sum()
        if n > 0:
            total_c = (y_true == c).sum()
            print(f"    {class_names[c]:<20s}: {n:>4d} high-conf errors  "
                  f"({100*n/total_c:.1f}% of class samples)")

    # Breakdown by predicted class
    print(f"\n  By PREDICTED class (what did the model wrongly say?):")
    for c in range(len(class_names)):
        mask = hc_wrong & (y_pred == c)
        n = mask.sum()
        if n > 0:
            print(f"    {class_names[c]:<20s}: {n:>4d} wrongly predicted with high confidence")

    # Top confused pairs among high-conf errors
    from sklearn.metrics import confusion_matrix
    cm_hc = confusion_matrix(
        y_true[hc_wrong], y_pred[hc_wrong], labels=list(range(len(class_names)))
    )
    np.fill_diagonal(cm_hc, 0)
    print(f"\n  Top high-confidence confusion pairs:")
    flat = np.argsort(cm_hc.ravel())[::-1][:5]
    for idx in flat:
        tc = idx // len(class_names)
        pc = idx % len(class_names)
        cnt = cm_hc[tc, pc]
        if cnt == 0:
            break
        print(f"    {class_names[tc]} -> {class_names[pc]}: {cnt}")

    # Save plot
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # True class breakdown
        counts_true = [((hc_wrong) & (y_true == c)).sum() for c in range(len(class_names))]
        axes[0].barh(class_names, counts_true, color="tomato")
        axes[0].set_xlabel("Count")
        axes[0].set_title("High-Confidence Errors by True Class")

        # Predicted class breakdown
        counts_pred = [((hc_wrong) & (y_pred == c)).sum() for c in range(len(class_names))]
        axes[1].barh(class_names, counts_pred, color="steelblue")
        axes[1].set_xlabel("Count")
        axes[1].set_title("High-Confidence Errors by Predicted Class")

        plt.tight_layout()
        path = save_dir / "high_confidence_errors.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to {path}")


def shap_analysis(
    model,
    X_val: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
    save_dir: Path | None = None,
    max_display: int = 20,
    n_samples: int = 500,
) -> None:
    """Compute SHAP values and generate summary plot."""
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
        if shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes)
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
    from src.training import fmt_time

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

    esm_model_name = artifact.get("esm_model_name", "esm2_t6_8M_UR50D")

    # Build feature matrix matching feature_source
    if "Physicochemical" in feature_source and "ESM" in feature_source:
        # ESM-2 + Physicochemical (primary combo from advanced.py)
        from src.features.embeddings import get_cache_filename
        esm_file = get_cache_filename(esm_model_name)
        esm_X = np.load(features_dir / esm_file)
        hc_path = features_dir / "handcrafted_features.npy"
        if hc_path.exists():
            hc_X = np.load(hc_path)
            physico_X = hc_X[:, -8:]  # last 8 cols = physicochemical
        else:
            from src.features.physicochemical import extract_physicochemical_features
            seqs = df["sequence"].tolist()
            physico_X = extract_physicochemical_features(seqs)
        X = np.hstack([esm_X, physico_X])
        esm_dim = esm_X.shape[1]
        physico_names = ["charge", "hydro_mean", "hydro_std", "MW", "pI",
                         "arom", "instability", "gravy"]
        feature_names = [f"ESM_{i}" for i in range(esm_dim)] + physico_names
    elif "ESM" in feature_source:
        from src.features.embeddings import get_cache_filename
        esm_file = get_cache_filename(esm_model_name)
        X = np.load(features_dir / esm_file)
        feature_names = [f"ESM_{i}" for i in range(X.shape[1])]
    else:
        X = np.load(features_dir / "handcrafted_features.npy")
        names_path = features_dir / "feature_names.npy"
        if names_path.exists():
            feature_names = np.load(names_path, allow_pickle=True).tolist()
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
    print(f"Features: {X.shape}  |  Names: {len(feature_names)}")

    # Stratified 20% hold-out for feature importance & SHAP
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(X, y))

    # Scale validation features using the artifact scaler (matches training)
    X_val = scaler.transform(X[val_idx])
    y_val = y[val_idx]

    # -- Prefer saved OOF predictions for confusion analysis -------------------
    oof_preds = artifact.get("oof_preds")
    oof_proba = artifact.get("oof_proba")
    oof_true = artifact.get("oof_true")

    if oof_preds is not None and oof_true is not None:
        print("Using OOF (out-of-fold) predictions from CV - honest held-out evaluation.")
        y_eval = oof_true
        y_pred_eval = oof_preds
        y_proba_eval = oof_proba
    else:
        print("WARNING: OOF predictions not in model artifact. Using held-out split.")
        y_eval = y_val
        y_pred_eval = model.predict(X_val)
        y_proba_eval = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

    print(f"Evaluation set: {len(y_eval):,} samples  [{fmt_time(time.time()-t0)}]")

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
        y_eval, y_pred_eval, y_proba_eval,
        save_dir=figures_dir,
    )

    # -- 3. High-confidence error analysis ----------------------------------------
    print(f"\n{'='*72}")
    print("  HIGH-CONFIDENCE ERROR ANALYSIS")
    print(f"{'='*72}")
    if y_proba_eval is not None:
        high_confidence_error_analysis(
            y_eval, y_pred_eval, y_proba_eval,
            save_dir=figures_dir,
        )
    else:
        print("  Skipped -- no probability data available.")

    # -- 4. SHAP analysis ------------------------------------------------------
    print(f"\n{'='*72}")
    print("  SHAP ANALYSIS")
    print(f"{'='*72}")
    shap_analysis(
        model, X_val, feature_names,
        save_dir=figures_dir,
        max_display=20,
        n_samples=1000,
    )

    # -- Persist structured results to JSON ------------------------------------
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix as sk_cm
    from src.evaluation import save_results_json

    prec, rec, f1_arr, sup = precision_recall_fscore_support(
        y_eval, y_pred_eval, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    per_class = [
        {"class": CLASS_NAMES[i], "precision": float(prec[i]),
         "recall": float(rec[i]), "f1": float(f1_arr[i]), "support": int(sup[i])}
        for i in range(len(CLASS_NAMES))
    ]

    # Top-30 importance
    top30 = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:30]

    # High-confidence error counts
    hc_data = {}
    if y_proba_eval is not None:
        max_p = y_proba_eval.max(axis=1)
        wrong = y_eval != y_pred_eval
        hc_wrong = wrong & (max_p >= 0.80)
        hc_data = {
            "total_errors": int(wrong.sum()),
            "high_conf_errors": int(hc_wrong.sum()),
            "by_true_class": {
                CLASS_NAMES[c]: int((hc_wrong & (y_eval == c)).sum())
                for c in range(len(CLASS_NAMES))
            },
            "by_pred_class": {
                CLASS_NAMES[c]: int((hc_wrong & (y_pred_eval == c)).sum())
                for c in range(len(CLASS_NAMES))
            },
        }

    interp_data = {
        "feature_source": feature_source,
        "per_class_metrics": per_class,
        "top_features": [{"name": n, "importance": float(v)} for n, v in top30],
        "high_confidence_errors": hc_data,
    }
    save_results_json(interp_data, project_root / "outputs" / "interpretability_results.json")

    print(f"\nTotal interpretability time: {fmt_time(time.time()-t0)}")
