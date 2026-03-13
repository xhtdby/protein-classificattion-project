"""
Soft-vote ensemble: fine-tuned ESM-2 8M + XGBoost-SMOTE on frozen ESM-2 650M.

Each model sees the same sequences from a different angle:
  - Fine-tuned 8M   : end-to-end task-specific fine-tuning; captures positional
                      and sequential context that mean-pooling discards.
  - XGBoost 650M    : frozen 1280-dim mean-pool embeddings from a much larger
                      PLM + 8 physicochemical features; richer representation.

Ensemble strategy
-----------------
Soft voting: weighted average of predicted probability vectors.
Default weights: proportional to each model's OOF Macro F1.
Per-class threshold optimisation applied to the blended probabilities.

OOF ensemble estimate (no GPU needed)
--------------------------------------
Blends the stored OOF probability arrays from finetune_results.json and a
cached XGBoost OOF run (outputs/models/xgb_oof_proba.npy).

    python -m src.models.ensemble

Override weights:
    python -m src.models.ensemble --w-finetune 0.4 --w-xgboost 0.6

Blind prediction
-----------------
Use EnsemblePredictor as a drop-in replace for the joblib artefact model in
predict_blind.py via the --model-finetune flag:

    python -m src.predict_blind --fasta blind.fasta \\
        --model outputs/models/best_model.joblib \\
        --model-finetune outputs/models/finetune_artifact.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from src.data_loading import CLASS_NAMES, SEED
from src.evaluation import plot_confusion_matrix, save_results_json
from src.training import fmt_time, optimize_thresholds

logger = logging.getLogger(__name__)

N_CLASSES = 7


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# XGBoost OOF re-computation (one-time, cached to disk)
# ---------------------------------------------------------------------------

def compute_xgb_oof_proba(
    project_root: Path,
    feature_source: str,
    esm_model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the best XGBoost model's CV to collect OOF probabilities.

    Results are cached to outputs/models/xgb_oof_proba.npy and
    outputs/models/xgb_oof_true.npy so subsequent calls are instant.

    Returns
    -------
    oof_proba : ndarray, shape (N, 7)
    oof_true  : ndarray, shape (N,)
    """
    cache_proba = project_root / "outputs" / "models" / "xgb_oof_proba.npy"
    cache_true  = project_root / "outputs" / "models" / "xgb_oof_true.npy"

    if cache_proba.exists() and cache_true.exists():
        print("  Loading cached XGBoost OOF probabilities...", flush=True)
        return np.load(cache_proba), np.load(cache_true)

    print("  XGBoost OOF cache not found -- re-running 5-fold CV...", flush=True)
    from src.data_loading import load_all_sequences, get_cv_splits
    from src.training import cross_validate_model
    from src.features.physicochemical import extract_physicochemical_features
    from src.features.embeddings import get_cache_filename

    df = load_all_sequences(project_root)
    sequences = df["sequence"].tolist()
    y = df["label"].values
    cv_splits = get_cv_splits(y)

    emb_cache = project_root / "outputs" / "features" / get_cache_filename(esm_model_name)
    if not emb_cache.exists():
        raise FileNotFoundError(
            f"ESM-2 embedding cache not found: {emb_cache}\n"
            f"Run: python -m src.features.embeddings --model 650M"
        )
    emb = np.load(emb_cache)
    print(f"  Loaded ESM-2 embeddings: {emb.shape}", flush=True)

    if "physicochemical" in feature_source.lower():
        phys = extract_physicochemical_features(sequences)
        X = np.hstack([emb, phys])
    else:
        X = emb

    artefact     = joblib.load(project_root / "outputs" / "models" / "best_model.joblib")
    trained_xgb  = artefact["model"]
    xgb_params   = {k: v for k, v in trained_xgb.get_params().items()
                    if k not in ("callbacks", "early_stopping_rounds")}

    from xgboost import XGBClassifier

    def model_fn() -> XGBClassifier:
        return XGBClassifier(**xgb_params)

    result = cross_validate_model(
        model_fn, X, y, cv_splits=cv_splits,
        use_scaler=True, use_smote=True,
        model_name="XGBoost+SMOTE (650M, OOF recompute)",
    )

    oof_proba = result["oof_proba"]
    oof_true  = result["oof_true"]

    cache_proba.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_proba, oof_proba)
    np.save(cache_true,  oof_true)
    print(f"  Cached XGBoost OOF to {cache_proba}", flush=True)
    return oof_proba, oof_true


# ---------------------------------------------------------------------------
# OOF-based ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_oof_ensemble(
    finetune_results_path: Path,
    xgb_oof_proba: np.ndarray,
    xgb_oof_true: np.ndarray,
    w_finetune: float | None = None,
    w_xgboost: float | None = None,
    figures_dir: Path | None = None,
) -> dict:
    """Blend stored OOF probabilities and compute ensemble metrics.

    Parameters
    ----------
    finetune_results_path : Path to finetune_results.json.
    xgb_oof_proba : OOF probability matrix from XGBoost model, shape (N, 7).
    xgb_oof_true  : True labels, shape (N,) -- must match finetune order.
    w_finetune    : Weight for fine-tuned 8M (default: F1-proportional).
    w_xgboost     : Weight for XGBoost 650M  (default: F1-proportional).
    figures_dir   : Directory to save confusion matrix PNG.

    Returns
    -------
    dict with ensemble metrics, weights used, thresholds, and a ``summary``
    key that matches the format used by generate_plots.py.
    """
    ft_data = _load_json(finetune_results_path)
    if ft_data is None:
        raise FileNotFoundError(
            f"finetune_results.json not found at {finetune_results_path}"
        )

    ft_proba = np.array(ft_data["oof_proba"], dtype=np.float32)
    ft_true  = np.array(ft_data["oof_true"],  dtype=np.int32)

    if ft_proba.shape != xgb_oof_proba.shape:
        raise ValueError(
            f"Shape mismatch: finetune OOF {ft_proba.shape} "
            f"vs XGBoost OOF {xgb_oof_proba.shape}"
        )
    if not np.array_equal(ft_true, xgb_oof_true):
        raise ValueError("OOF true labels differ between fine-tune and XGBoost")

    y_true = ft_true

    # Derive weights from individual OOF Macro F1 unless supplied
    ft_f1  = f1_score(y_true, ft_proba.argmax(axis=1),
                      average="macro", zero_division=0)
    xgb_f1 = f1_score(y_true, xgb_oof_proba.argmax(axis=1),
                      average="macro", zero_division=0)

    if w_finetune is None or w_xgboost is None:
        total = ft_f1 + xgb_f1
        w_finetune = ft_f1  / total
        w_xgboost  = xgb_f1 / total

    print(f"\n  Fine-tune OOF Macro F1 : {ft_f1:.4f}  (weight {w_finetune:.3f})")
    print(f"  XGBoost  OOF Macro F1  : {xgb_f1:.4f}  (weight {w_xgboost:.3f})")

    # Soft vote
    blend_proba  = w_finetune * ft_proba + w_xgboost * xgb_oof_proba
    y_pred_blend = blend_proba.argmax(axis=1)
    metrics_plain = _compute_metrics(y_true, y_pred_blend)

    print(f"\n  Ensemble (plain argmax):")
    print(f"    Accuracy:          {metrics_plain['accuracy']:.4f}")
    print(f"    Macro F1:          {metrics_plain['macro_f1']:.4f}")
    print(f"    Balanced Accuracy: {metrics_plain['balanced_accuracy']:.4f}")
    print(f"    MCC:               {metrics_plain['mcc']:.4f}")

    # Per-class threshold optimisation
    print("\n  Optimising per-class thresholds on blended OOF...", flush=True)
    best_thresholds, metrics_opt = optimize_thresholds(y_true, blend_proba)
    y_pred_opt = (blend_proba / best_thresholds[np.newaxis, :]).argmax(axis=1)

    print(f"\n  Ensemble (optimised thresholds):")
    print(f"    Accuracy:          {metrics_opt['accuracy']:.4f}")
    print(f"    Macro F1:          {metrics_opt['macro_f1']:.4f}")
    print(f"    Balanced Accuracy: {metrics_opt['balanced_accuracy']:.4f}")
    print(f"    MCC:               {metrics_opt['mcc']:.4f}")
    print(f"    Thresholds: {np.round(best_thresholds, 3)}")

    print("\n--- OOF Classification Report (optimised thresholds) ---")
    print(classification_report(y_true, y_pred_opt,
                                target_names=CLASS_NAMES, zero_division=0))

    if figures_dir is not None:
        plot_confusion_matrix(
            y_true, y_pred_opt,
            save_path=figures_dir / "cm_ensemble.png",
            title="Ensemble (Fine-tuned 8M + XGBoost 650M) -- OOF",
        )

    return {
        "model_name":         "Ensemble (Fine-tuned 8M + XGBoost 650M)",
        "w_finetune":         float(w_finetune),
        "w_xgboost":          float(w_xgboost),
        "ft_oof_macro_f1":    float(ft_f1),
        "xgb_oof_macro_f1":   float(xgb_f1),
        "metrics_plain":      metrics_plain,
        "metrics_optimised":  metrics_opt,
        "thresholds":         best_thresholds.tolist(),
        # summary key matches format used by generate_plots.py
        "summary": {
            "accuracy_mean":          metrics_opt["accuracy"],
            "accuracy_std":           0.0,
            "macro_f1_mean":          metrics_opt["macro_f1"],
            "macro_f1_std":           0.0,
            "balanced_accuracy_mean": metrics_opt["balanced_accuracy"],
            "balanced_accuracy_std":  0.0,
            "mcc_mean":               metrics_opt["mcc"],
            "mcc_std":                0.0,
        },
    }


# ---------------------------------------------------------------------------
# EnsemblePredictor -- sklearn-style predict_proba for predict_blind.py
# ---------------------------------------------------------------------------

class EnsemblePredictor:
    """Soft-vote ensemble of fine-tuned ESM-2 8M and XGBoost-SMOTE 650M.

    Both models are loaded lazily on the first call to predict_proba.

    Parameters
    ----------
    xgb_artifact_path      : Path to best_model.joblib (XGBoost artefact).
    finetune_artifact_path : Path to finetune_artifact.joblib.
    w_finetune             : Weight for fine-tune probabilities (0-1).
    w_xgboost              : Weight for XGBoost probabilities  (0-1).
        If both are None, weights are loaded from ensemble_results.json;
        if that file is also absent, equal weights (0.5 / 0.5) are used.
    thresholds             : Per-class decision thresholds (7-element array).
    """

    def __init__(
        self,
        xgb_artifact_path: Path | str,
        finetune_artifact_path: Path | str,
        w_finetune: float | None = None,
        w_xgboost:  float | None = None,
        thresholds: np.ndarray | list | None = None,
    ) -> None:
        self.xgb_artifact_path      = Path(xgb_artifact_path)
        self.finetune_artifact_path = Path(finetune_artifact_path)
        self.w_finetune  = w_finetune
        self.w_xgboost   = w_xgboost
        self.thresholds  = np.asarray(thresholds) if thresholds is not None else None

    def _resolve_weights(self) -> tuple[float, float]:
        if self.w_finetune is not None and self.w_xgboost is not None:
            return self.w_finetune, self.w_xgboost
        project_root = Path(__file__).resolve().parent.parent.parent
        ens_path     = project_root / "outputs" / "ensemble_results.json"
        ens          = _load_json(ens_path)
        if ens:
            return float(ens["w_finetune"]), float(ens["w_xgboost"])
        logger.warning("No stored ensemble weights -- using equal weights (0.5 / 0.5)")
        return 0.5, 0.5

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        """Run both models and return blended probability matrix (N, 7)."""
        w_ft, w_xgb = self._resolve_weights()
        print(f"  [Ensemble] weights: finetune={w_ft:.3f}  xgboost={w_xgb:.3f}",
              flush=True)

        # Fine-tuned 8M
        print("  [Ensemble] Running fine-tuned ESM-2 8M...", flush=True)
        ft_artefact = joblib.load(self.finetune_artifact_path)
        proba_ft    = ft_artefact["model"].predict_proba(sequences)

        # XGBoost 650M
        print("  [Ensemble] Running XGBoost 650M...", flush=True)
        xgb_artefact = joblib.load(self.xgb_artifact_path)
        xgb_model    = xgb_artefact["model"]
        xgb_scaler   = xgb_artefact["scaler"]
        feature_src  = xgb_artefact.get("feature_source", "ESM-2 + Physicochemical")
        esm_name     = xgb_artefact.get("esm_model_name", "esm2_t33_650M_UR50D")

        from src.features.embeddings import extract_esm2_embeddings
        from src.features.physicochemical import extract_physicochemical_features

        emb = extract_esm2_embeddings(sequences, model_name=esm_name)
        if "physicochemical" in feature_src.lower():
            phys = extract_physicochemical_features(sequences)
            X    = np.hstack([emb, phys])
        else:
            X = emb
        X = xgb_scaler.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*",
                                    category=UserWarning)
            proba_xgb = xgb_model.predict_proba(X)

        blend = w_ft * proba_ft + w_xgb * proba_xgb

        if self.thresholds is not None:
            blend = blend / self.thresholds[np.newaxis, :]

        return blend


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate soft-vote ensemble of fine-tuned ESM-2 8M + XGBoost 650M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--w-finetune", type=float, default=None,
        help="Weight for fine-tuned 8M (default: auto from OOF Macro F1)",
    )
    parser.add_argument(
        "--w-xgboost", type=float, default=None,
        help="Weight for XGBoost 650M (default: auto from OOF Macro F1)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir  = project_root / "outputs" / "figures"
    models_dir   = project_root / "outputs" / "models"

    t0 = time.time()

    print("\n" + "=" * 72)
    print("  Ensemble: Fine-tuned ESM-2 8M + XGBoost-SMOTE 650M")
    print("=" * 72)

    xgb_artifact_path = models_dir / "best_model.joblib"
    if not xgb_artifact_path.exists():
        print(f"ERROR: {xgb_artifact_path} not found. Run src.models.advanced first.")
        sys.exit(1)

    xgb_artefact   = joblib.load(xgb_artifact_path)
    feature_source = xgb_artefact.get("feature_source", "ESM-2 + Physicochemical")
    esm_model_name = xgb_artefact.get("esm_model_name", "esm2_t33_650M_UR50D")
    print(f"\n  XGBoost feature source : {feature_source}")
    print(f"  ESM-2 model            : {esm_model_name}")

    print("\n  Step 1: Obtaining XGBoost OOF probabilities...")
    xgb_oof_proba, xgb_oof_true = compute_xgb_oof_proba(
        project_root, feature_source, esm_model_name
    )
    print(f"  XGBoost OOF shape: {xgb_oof_proba.shape}")

    finetune_results_path = project_root / "outputs" / "finetune_results.json"
    if not finetune_results_path.exists():
        print(f"ERROR: {finetune_results_path} not found. Run src.models.finetune first.")
        sys.exit(1)

    print("\n  Step 2: Blending OOF probabilities and evaluating ensemble...")
    results = evaluate_oof_ensemble(
        finetune_results_path=finetune_results_path,
        xgb_oof_proba=xgb_oof_proba,
        xgb_oof_true=xgb_oof_true,
        w_finetune=args.w_finetune,
        w_xgboost=args.w_xgboost,
        figures_dir=figures_dir,
    )

    out_path = project_root / "outputs" / "ensemble_results.json"
    save_results_json(results, out_path)

    print(f"\n  Total time: {fmt_time(time.time() - t0)}")
    print("=" * 72)
    print(f"  Results saved to {out_path}")
    print(f"  Confusion matrix : {figures_dir / 'cm_ensemble.png'}")
    print("=" * 72)
