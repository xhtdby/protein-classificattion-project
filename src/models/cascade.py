"""
Hierarchical Cascade: two-stage classification for extreme class imbalance.

Stage 1 — Binary Filter:
    Separate "Enzyme" (classes 1–6 combined) from "Not Enzyme" (class 0).
    This isolates the 81.5% majority class so the latent space is not distorted.

Stage 2 — Multiclass Classifier:
    Among the enzyme subset, classify into 6 EC classes (1–6).
    Without the gravitational pull of class 0, separation is cleaner.

Final probability:
    P(class 0) = P(not-enzyme)
    P(class i) = P(enzyme) * P(class i | enzyme)   for i = 1..6

Usage:
    python -m src.models.cascade
    python -m src.models.cascade --esm-model 650M
    python -m src.models.cascade --tune
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loading import load_all_sequences, get_cv_splits, SEED, CLASS_NAMES
from src.training import fmt_time, optimize_thresholds, print_cv_summary
from src.evaluation import (
    plot_confusion_matrix,
    print_metrics_table,
    print_classification_report,
    save_results_json,
)

logger = logging.getLogger(__name__)

N_CLASSES = 7


# ---------------------------------------------------------------------------
# Cascade predictor
# ---------------------------------------------------------------------------

class CascadePredictor:
    """Two-stage hierarchical classifier.

    Attributes
    ----------
    stage1_model : sklearn-compatible binary classifier (0 vs 1).
    stage2_model : sklearn-compatible 6-class classifier (classes 1–6).
    stage1_scaler : StandardScaler fitted on full features for stage 1.
    stage2_scaler : StandardScaler fitted on enzyme-only features for stage 2.
    stage1_threshold : float, decision threshold for the binary stage.
    """

    def __init__(
        self,
        stage1_model,
        stage2_model,
        stage1_scaler: StandardScaler | None = None,
        stage2_scaler: StandardScaler | None = None,
        stage1_threshold: float = 0.5,
    ) -> None:
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.stage1_scaler = stage1_scaler
        self.stage2_scaler = stage2_scaler
        self.stage1_threshold = stage1_threshold

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 7) probability matrix via cascade.

        P(class 0) = P(not-enzyme)
        P(class i) = P(enzyme) × P(class i | enzyme)  for i in 1..6
        """
        # Stage 1: binary
        X1 = self.stage1_scaler.transform(X) if self.stage1_scaler else X
        p_binary = self.stage1_model.predict_proba(X1)  # (N, 2)
        p_not_enzyme = p_binary[:, 0]  # P(not enzyme)
        p_enzyme = p_binary[:, 1]      # P(enzyme)

        # Stage 2: multiclass among enzymes
        X2 = self.stage2_scaler.transform(X) if self.stage2_scaler else X
        p_ec = self.stage2_model.predict_proba(X2)  # (N, 6) for classes 1–6

        # Combine: scale enzyme sub-probabilities by P(enzyme)
        proba = np.zeros((len(X), N_CLASSES), dtype=np.float32)
        proba[:, 0] = p_not_enzyme
        for i in range(6):
            proba[:, i + 1] = p_enzyme * p_ec[:, i]

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return argmax predictions."""
        return self.predict_proba(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# Cross-validation with cascade
# ---------------------------------------------------------------------------

def cross_validate_cascade(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list,
    make_stage1_fn,
    make_stage2_fn,
    use_scaler: bool = True,
    model_name: str = "Cascade",
) -> dict:
    """Run stratified 5-fold CV with hierarchical cascade.

    Data-leakage-safe: scalers fitted on training fold only,
    class weights computed on training fold only.
    """
    n_folds = len(cv_splits)
    fold_metrics: list[dict] = []
    all_val_indices: list[np.ndarray] = []
    all_val_preds: list[np.ndarray] = []
    all_val_proba: list[np.ndarray] = []
    all_val_true: list[np.ndarray] = []

    print(f"\n{'-'*72}")
    print(f"  Training: {model_name}  |  Folds: {n_folds}  |  "
          f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"  Class distribution: {np.bincount(y).tolist()}")
    print(f"{'-'*72}")
    cv_start = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        fold_start = time.time()
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  [Fold {fold_i+1}/{n_folds}] "
              f"train={len(train_idx):,}  val={len(val_idx):,}  ", end="", flush=True)

        # --- Stage 1: Binary (enzyme=1, not-enzyme=0) ---
        y_binary_train = (y_train > 0).astype(int)

        scaler1 = StandardScaler() if use_scaler else None
        X1_train = scaler1.fit_transform(X_train) if scaler1 else X_train
        X1_val = scaler1.transform(X_val) if scaler1 else X_val

        sw1 = compute_sample_weight("balanced", y_binary_train)
        stage1 = make_stage1_fn()
        stage1.fit(X1_train, y_binary_train, sample_weight=sw1)

        # --- Stage 2: 6-class among enzymes only ---
        enzyme_mask_train = y_train > 0
        X_enzyme_train = X_train[enzyme_mask_train]
        # Remap labels 1–6 → 0–5 for stage 2
        y_enzyme_train = y_train[enzyme_mask_train] - 1

        scaler2 = StandardScaler() if use_scaler else None
        X2_train = scaler2.fit_transform(X_enzyme_train) if scaler2 else X_enzyme_train

        sw2 = compute_sample_weight("balanced", y_enzyme_train)
        stage2 = make_stage2_fn()
        stage2.fit(X2_train, y_enzyme_train, sample_weight=sw2)

        # --- Cascade prediction on validation ---
        # Stage 1 probabilities
        p_binary = stage1.predict_proba(X1_val)  # (N_val, 2)
        p_not_enzyme = p_binary[:, 0]
        p_enzyme = p_binary[:, 1]

        # Stage 2 probabilities
        X2_val = scaler2.transform(X_val) if scaler2 else X_val
        p_ec = stage2.predict_proba(X2_val)  # (N_val, 6)

        # Combine
        proba = np.zeros((len(X_val), N_CLASSES), dtype=np.float32)
        proba[:, 0] = p_not_enzyme
        for i in range(6):
            proba[:, i + 1] = p_enzyme * p_ec[:, i]

        y_pred = proba.argmax(axis=1)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "macro_f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
            "mcc": matthews_corrcoef(y_val, y_pred),
        }
        fold_metrics.append(metrics)
        all_val_indices.append(val_idx)
        all_val_preds.append(y_pred)
        all_val_proba.append(proba)
        all_val_true.append(y_val)

        elapsed = time.time() - fold_start
        print(
            f"Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}  "
            f"BA={metrics['balanced_accuracy']:.4f}  MCC={metrics['mcc']:.4f}  "
            f"[{fmt_time(elapsed)}]"
        )

    total_elapsed = time.time() - cv_start
    print(f"  Total CV time: {fmt_time(total_elapsed)}")

    # Aggregate
    metric_names = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]
    summary = {}
    for m in metric_names:
        vals = [fm[m] for fm in fold_metrics]
        summary[f"{m}_mean"] = float(np.mean(vals))
        summary[f"{m}_std"] = float(np.std(vals))

    # Concatenate OOF in original order
    oof_indices = np.concatenate(all_val_indices)
    oof_preds = np.concatenate(all_val_preds)
    oof_proba = np.concatenate(all_val_proba, axis=0)
    oof_true = np.concatenate(all_val_true)
    sort_order = np.argsort(oof_indices)
    oof_preds = oof_preds[sort_order]
    oof_proba = oof_proba[sort_order]
    oof_true = oof_true[sort_order]

    return {
        "model_name": model_name,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "oof_preds": oof_preds,
        "oof_proba": oof_proba,
        "oof_true": oof_true,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical Cascade: two-stage enzyme classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m src.models.cascade                      # 8M embeddings\n"
               "  python -m src.models.cascade --esm-model 650M     # 650M embeddings\n",
    )
    parser.add_argument(
        "--esm-model", default="8M", choices=["8M", "650M"],
        help="ESM-2 model size (must have embeddings pre-extracted). Default: 8M",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    import torch
    from xgboost import XGBClassifier

    # -- Hardware detection
    hw_cuda = torch.cuda.is_available()
    xgb_device = "cuda" if hw_cuda else "cpu"

    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = project_root / "outputs" / "figures"
    features_dir = project_root / "outputs" / "features"
    models_dir = project_root / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data
    t0 = time.time()
    print("Loading sequences...")
    df = load_all_sequences(project_root)
    y = df["label"].values
    cv_splits = get_cv_splits(y)
    print(f"  Loaded {len(df):,} sequences  [{fmt_time(time.time() - t0)}]")

    # -- Load ESM-2 embeddings + physicochemical features
    from src.features.embeddings import get_cache_filename

    ESM_ALIAS = {"8M": "esm2_t6_8M_UR50D", "650M": "esm2_t33_650M_UR50D"}
    esm_model_name = ESM_ALIAS[args.esm_model]
    esm_cache = features_dir / get_cache_filename(esm_model_name)

    if not esm_cache.exists():
        print(f"\nERROR: ESM-2 embeddings not found at {esm_cache.name}")
        print(f"  Run:  python -m src.features.embeddings --model {args.esm_model}")
        sys.exit(1)

    esm_X = np.load(esm_cache)
    print(f"  Loaded ESM-2 embeddings: shape={esm_X.shape}")

    # Physicochemical features
    hc_path = features_dir / "handcrafted_features.npy"
    if hc_path.exists():
        hc_X = np.load(hc_path)
        physico_X = hc_X[:, -8:]  # last 8 columns
        primary_X = np.hstack([esm_X, physico_X])
        feature_source = "ESM-2 + Physicochemical"
    else:
        primary_X = esm_X
        feature_source = "ESM-2"

    print(f"  Primary features: {feature_source}  shape={primary_X.shape}")

    # -- Model factories
    def make_stage1():
        """Binary XGBoost: enzyme vs not-enzyme."""
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=0,
            device=xgb_device,
        )

    def make_stage2():
        """6-class XGBoost: EC class classification."""
        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=0,
            device=xgb_device,
        )

    # -- Run cascade CV
    print(f"\n{'='*72}")
    print(f"  HIERARCHICAL CASCADE (Binary → 6-class)")
    print(f"{'='*72}")

    cascade_results = cross_validate_cascade(
        primary_X, y, cv_splits,
        make_stage1_fn=make_stage1,
        make_stage2_fn=make_stage2,
        model_name=f"Cascade ({feature_source})",
    )
    print_cv_summary(cascade_results)

    plot_confusion_matrix(
        cascade_results["oof_true"],
        cascade_results["oof_preds"],
        save_path=figures_dir / "cm_cascade.png",
        title=f"Hierarchical Cascade ({feature_source}) -- OOF",
    )

    print("\n--- OOF Classification Report ---")
    print_classification_report(cascade_results["oof_true"], cascade_results["oof_preds"])

    # -- Per-class threshold optimisation
    print("  Optimising per-class thresholds on OOF...", flush=True)
    thresholds, thresh_metrics = optimize_thresholds(
        cascade_results["oof_true"], cascade_results["oof_proba"],
    )
    print(f"  Thresholds: {np.round(thresholds, 3).tolist()}")
    print(f"  Threshold-opt:  Macro F1={thresh_metrics['macro_f1']:.4f}  "
          f"BA={thresh_metrics['balanced_accuracy']:.4f}  "
          f"MCC={thresh_metrics['mcc']:.4f}")

    # -- Save full-dataset cascade model
    print("\n  Retraining cascade on full dataset...")
    scaler1 = StandardScaler()
    X1 = scaler1.fit_transform(primary_X)
    y_binary = (y > 0).astype(int)
    sw1 = compute_sample_weight("balanced", y_binary)
    final_stage1 = make_stage1()
    final_stage1.fit(X1, y_binary, sample_weight=sw1)

    enzyme_mask = y > 0
    scaler2 = StandardScaler()
    X2 = scaler2.fit_transform(primary_X[enzyme_mask])
    y_enzyme = y[enzyme_mask] - 1
    sw2 = compute_sample_weight("balanced", y_enzyme)
    final_stage2 = make_stage2()
    final_stage2.fit(X2, y_enzyme, sample_weight=sw2)

    cascade_model = CascadePredictor(
        stage1_model=final_stage1,
        stage2_model=final_stage2,
        stage1_scaler=scaler1,
        stage2_scaler=scaler2,
    )

    artifact_path = models_dir / "cascade_model.joblib"
    joblib.dump({
        "model": cascade_model,
        "feature_source": feature_source,
        "esm_model_name": esm_model_name,
        "esm_embedding_dim": esm_X.shape[1],
        "thresholds": thresholds,
        "cv_scores": cascade_results["summary"],
    }, artifact_path)
    print(f"  Saved -> {artifact_path}")

    # -- Save results JSON
    results_data = {
        "model_name": cascade_results["model_name"],
        "feature_source": feature_source,
        "esm_model": esm_model_name,
        "summary": cascade_results["summary"],
        "threshold_metrics": thresh_metrics,
        "thresholds": thresholds.tolist(),
    }
    save_results_json(results_data, project_root / "outputs" / "cascade_results.json")

    print(f"\nTotal wall time: {fmt_time(time.time() - t0)}")
