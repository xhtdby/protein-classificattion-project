"""
Confidence scoring: map predicted probabilities to High / Medium / Low levels.

Thresholds (from coursework spec):
  High:   p >= 0.80  -> +/-1 point
  Medium: 0.50 <= p < 0.80  -> +/-0.5 point
  Low:    p < 0.50  -> 0 points
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def assign_confidence(proba: np.ndarray) -> list[str]:
    """Map max predicted probability to confidence level.

    Args:
        proba: (N, 7) probability arrays or (N,) max-prob values.

    Returns:
        List of "High", "Medium", or "Low" strings.
    """
    if proba.ndim == 2:
        max_p = proba.max(axis=1)
    else:
        max_p = proba

    levels = np.where(max_p >= 0.80, "High",
                      np.where(max_p >= 0.50, "Medium", "Low"))
    return levels.tolist()


def confidence_calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    save_path: Path | None = None,
) -> dict[str, dict]:
    """Compute accuracy per confidence level and generate reliability diagram.

    Args:
        y_true: True labels.
        y_proba: (N, 7) predicted probabilities.
        y_pred: Predicted labels (derived from y_proba if None).
        save_path: Path to save calibration plot.

    Returns:
        Dict with stats per confidence level.
    """
    if y_pred is None:
        y_pred = y_proba.argmax(axis=1)

    max_p = y_proba.max(axis=1)
    levels = assign_confidence(y_proba)
    correct = y_true == y_pred

    report = {}
    print("\n=== Confidence Calibration Report ===")
    for level in ["High", "Medium", "Low"]:
        mask = np.array([l == level for l in levels])
        n = mask.sum()
        if n == 0:
            report[level] = {"count": 0, "accuracy": 0.0, "mean_prob": 0.0}
            continue
        acc = correct[mask].mean()
        mean_p = max_p[mask].mean()
        report[level] = {"count": int(n), "accuracy": float(acc), "mean_prob": float(mean_p)}
        print(f"  {level:>6s}: {n:>6d} predictions, "
              f"accuracy={acc:.4f}, mean_max_prob={mean_p:.4f}")

    total_correct = correct.sum()
    print(f"  {'Total':>6s}: {len(y_true):>6d} predictions, "
          f"accuracy={total_correct/len(y_true):.4f}")

    # Reliability diagram
    if save_path:
        _plot_reliability_diagram(max_p, correct, save_path)

    return report


def _plot_reliability_diagram(max_p: np.ndarray, correct: np.ndarray, save_path: Path) -> None:
    """Plot reliability (calibration) diagram."""
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (max_p >= bin_edges[i]) & (max_p < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (max_p >= bin_edges[i]) & (max_p <= bin_edges[i + 1])
        n = mask.sum()
        if n > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(max_p[mask].mean())
            bin_counts.append(n)
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})


    # Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="steelblue", label="Model")
    ax1.axvline(x=0.50, color="orange", linestyle=":", alpha=0.7, label="Medium threshold")
    ax1.axvline(x=0.80, color="red", linestyle=":", alpha=0.7, label="High threshold")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction Correct")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogram of predictions
    ax2.bar(bin_confs, bin_counts, width=0.08, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram saved to {save_path}")


if __name__ == "__main__":
    import logging
    import sys
    import joblib
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.data_loading import load_all_sequences, SEED

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

    # -- Load model ------------------------------------------------------------
    model_path = models_dir / "best_model.joblib"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run src.models.advanced first.")
        sys.exit(1)

    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_source = artifact.get("feature_source", "Handcrafted")
    print(f"Loaded {type(model).__name__} ({feature_source}) from {model_path}")

    # -- Load data & features --------------------------------------------------
    df = load_all_sequences(project_root)
    y = df["label"].values

    esm_model_name = artifact.get("esm_model_name", "esm2_t6_8M_UR50D")

    # Build feature matrix matching feature_source
    if "Physicochemical" in feature_source and "ESM" in feature_source:
        from src.features.embeddings import get_cache_filename
        esm_file = get_cache_filename(esm_model_name)
        esm_X = np.load(features_dir / esm_file)
        hc_path = features_dir / "handcrafted_features.npy"
        if hc_path.exists():
            hc_X = np.load(hc_path)
            physico_X = hc_X[:, -8:]
        else:
            from src.features.physicochemical import extract_physicochemical_features
            seqs = df["sequence"].tolist()
            physico_X = extract_physicochemical_features(seqs)
        X = np.hstack([esm_X, physico_X])
    elif "ESM" in feature_source:
        from src.features.embeddings import get_cache_filename
        esm_file = get_cache_filename(esm_model_name)
        X = np.load(features_dir / esm_file)
    else:
        X = np.load(features_dir / "handcrafted_features.npy")
    print(f"Features loaded: {X.shape}")

    # Stratified 20% hold-out -- indices for fresh scaler fitting
    from sklearn.preprocessing import StandardScaler
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss.split(X, y))

    # -- Prefer saved OOF probabilities for honest calibration -----------------
    oof_proba = artifact.get("oof_proba")
    oof_true = artifact.get("oof_true")

    if oof_proba is not None and oof_true is not None:
        print("Using OOF probabilities from CV -- honest held-out evaluation.")
        y_proba = oof_proba
        y_pred = oof_proba.argmax(axis=1)
        y_val = oof_true
    else:
        print("WARNING: OOF data not in model artifact.")
        print("  Re-run 'python -m src.models.advanced' to save OOF predictions.")
        print("  Falling back to held-out split with fresh scaler.")
        sc_eval = StandardScaler()
        sc_eval.fit(X[train_idx])
        X_val = sc_eval.transform(X[val_idx])
        y_val = y[val_idx]
        y_proba = model.predict_proba(X_val)
        y_pred = y_proba.argmax(axis=1)

    print(f"Calibration set: {len(y_val):,} samples")

    # -- Confidence calibration report -----------------------------------------
    report = confidence_calibration_report(
        y_val, y_proba, y_pred,
        save_path=figures_dir / "reliability_diagram.png",
    )

    # -- Persist structured results to JSON ------------------------------------
    from src.evaluation import save_results_json
    conf_data = {
        "feature_source": feature_source,
        "n_samples": len(y_val),
        "calibration": report,
    }
    save_results_json(conf_data, project_root / "outputs" / "confidence_results.json")
