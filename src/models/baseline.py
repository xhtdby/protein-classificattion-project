"""
Baseline models: Logistic Regression and Random Forest.

Both use class_weight='balanced' to handle class imbalance.
Run as standalone to execute 5-fold CV on handcrafted features.
"""

import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from src.data_loading import load_all_sequences, get_cv_splits, SEED
from src.features.composition import extract_composition_features
from src.features.physicochemical import extract_physicochemical_features
from src.training import cross_validate_model, print_cv_summary, fmt_time
from src.evaluation import (
    plot_confusion_matrix,
    print_metrics_table,
    print_classification_report,
    save_results_json,
)

logger = logging.getLogger(__name__)


def make_logistic_regression():
    """Create a fresh Logistic Regression model."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
    )


def make_random_forest():
    """Create a fresh Random Forest model."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def make_random_forest_pca(n_components: int = 50):
    """RF with PCA pre-reduction — avoids curse of dimensionality on sparse dipeptides."""
    return Pipeline([
        ("pca", PCA(n_components=n_components, random_state=SEED)),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def build_handcrafted_features(sequences: list[str], cache_dir: Path | None = None) -> np.ndarray:
    """Build combined handcrafted feature matrix (composition + physicochemical).

    Caches to disk if cache_dir is provided.
    """
    if cache_dir:
        cache_path = cache_dir / "handcrafted_features.npy"
        if cache_path.exists():
            data = np.load(cache_path)
            if data.shape[0] == len(sequences):
                logger.info("Loaded cached handcrafted features from %s", cache_path)
                return data

    comp_feats = extract_composition_features(sequences)
    phys_feats = extract_physicochemical_features(sequences)
    X = np.hstack([comp_feats, phys_feats])

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "handcrafted_features.npy", X)
        np.save(cache_dir / "feature_names.npy",
                np.array(
                    _get_all_feature_names(),
                    dtype=object,
                ))
        logger.info("Cached handcrafted features to %s", cache_dir)

    return X


def _get_all_feature_names() -> list[str]:
    """Get combined feature name list."""
    from src.features.composition import get_feature_names as comp_names
    from src.features.physicochemical import get_feature_names as phys_names
    return comp_names() + phys_names()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy DEBUG output from matplotlib font manager
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = project_root / "outputs" / "figures"
    features_dir = project_root / "outputs" / "features"

    t0 = time.time()

    # -- Load data -------------------------------------------------------------
    print("=" * 72)
    print("  BASELINE MODELS  --  Logistic Regression & Random Forest")
    print("=" * 72)
    print("Loading sequences...")
    df = load_all_sequences(project_root)
    sequences = df["sequence"].tolist()
    y = df["label"].values
    print(f"  Loaded {len(df):,} sequences  [{fmt_time(time.time() - t0)}]")
    print(f"  Class counts: {np.bincount(y).tolist()}")

    # -- Features --------------------------------------------------------------
    t_feat = time.time()
    print("\nExtracting handcrafted features (cached if already done)...")
    X = build_handcrafted_features(sequences, cache_dir=features_dir)
    print(f"  Feature matrix: {X.shape}  [{fmt_time(time.time() - t_feat)}]")

    cv_splits = get_cv_splits(y)

    # -- Logistic Regression ---------------------------------------------------
    try:
        lr_results = cross_validate_model(
            make_logistic_regression, X, y,
            cv_splits=cv_splits, model_name="Logistic Regression",
        )
        print_cv_summary(lr_results)
        plot_confusion_matrix(
            lr_results["oof_true"], lr_results["oof_preds"],
            save_path=figures_dir / "cm_logistic_regression.png",
            title="Logistic Regression -- Confusion Matrix",
        )
    except Exception:
        print("\nFATAL: Logistic Regression training failed:")
        traceback.print_exc()
        sys.exit(1)

    # -- Random Forest ---------------------------------------------------------
    try:
        rf_results = cross_validate_model(
            make_random_forest, X, y,
            cv_splits=cv_splits, model_name="Random Forest",
        )
        print_cv_summary(rf_results)
        plot_confusion_matrix(
            rf_results["oof_true"], rf_results["oof_preds"],
            save_path=figures_dir / "cm_random_forest.png",
            title="Random Forest -- Confusion Matrix",
        )
    except Exception:
        print("\nFATAL: Random Forest training failed:")
        traceback.print_exc()
        sys.exit(1)

    # -- Random Forest + PCA (fix sparse dipeptide curse) ----------------------
    try:
        rf_pca_results = cross_validate_model(
            make_random_forest_pca, X, y,
            cv_splits=cv_splits, use_scaler=False,  # PCA handles scaling internally
            model_name="Random Forest + PCA(50)",
        )
        print_cv_summary(rf_pca_results)
        plot_confusion_matrix(
            rf_pca_results["oof_true"], rf_pca_results["oof_preds"],
            save_path=figures_dir / "cm_random_forest_pca.png",
            title="Random Forest + PCA(50) -- Confusion Matrix",
        )
    except Exception:
        print("\nWARN: RF+PCA training failed:")
        traceback.print_exc()
        rf_pca_results = None

    # -- Summary ---------------------------------------------------------------
    baselines = [lr_results, rf_results]
    if rf_pca_results is not None:
        baselines.append(rf_pca_results)
    print_metrics_table(baselines)
    print("\n--- Logistic Regression Classification Report ---")
    print_classification_report(lr_results["oof_true"], lr_results["oof_preds"])
    print("\n--- Random Forest Classification Report ---")
    print_classification_report(rf_results["oof_true"], rf_results["oof_preds"])

    # -- Persist structured results to JSON ------------------------------------
    results_dir = project_root / "outputs"
    baseline_data = {
        "models": [
            {"model_name": r["model_name"], "summary": r["summary"]}
            for r in baselines
        ],
    }
    save_results_json(baseline_data, results_dir / "baseline_results.json")

    print(f"\nTotal wall time: {fmt_time(time.time() - t0)}")
