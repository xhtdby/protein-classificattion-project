"""
Cross-validation training loop with data-leakage-safe preprocessing.

CRITICAL RULES (enforced here):
- Scaler .fit() only on training fold
- SMOTE only on training fold
- Metrics computed on unmodified validation fold
"""

import random
import logging
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef

from src.data_loading import get_cv_splits

logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss or Xh Xm."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def cross_validate_model(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    n_splits: int = 5,
    use_scaler: bool = True,
    use_smote: bool = False,
    model_name: str = "model",
) -> dict:
    """Run stratified K-fold cross-validation with leak-safe preprocessing."""
    if cv_splits is None:
        cv_splits = get_cv_splits(y, n_splits=n_splits)

    n_folds = len(cv_splits)
    fold_metrics: list[dict] = []
    all_val_indices: list[np.ndarray] = []
    all_val_preds: list[np.ndarray] = []
    all_val_proba: list[np.ndarray] = []
    all_val_true: list[np.ndarray] = []
    fold_times: list[float] = []

    print(f"\n{'-'*72}")
    print(f"  Training: {model_name}  |  Folds: {n_folds}  |  "
          f"Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"  Class distribution: {np.bincount(y).tolist()}")
    print(f"{'-'*72}")
    cv_start = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        fold_start = time.time()
        print(f"  [Fold {fold_i+1}/{n_folds}] "
              f"train={len(train_idx):,}  val={len(val_idx):,}  ", end="", flush=True)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale -- fit on train only
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # SMOTE -- train fold only
        if use_smote:
            from imblearn.over_sampling import SMOTE
            smote_start = time.time()
            print(f"SMOTE... ", end="", flush=True)
            smote = SMOTE(random_state=SEED)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"({_fmt_time(time.time()-smote_start)}, "
                  f"resampled to {len(y_train):,})  ", end="", flush=True)

        # Train
        try:
            model = model_fn()
            model.fit(X_train, y_train)
        except Exception:
            print("\n  ERROR during model.fit():")
            traceback.print_exc()
            raise

        # Predict -- suppress the LightGBM feature-names UserWarning from sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
            y_pred = model.predict(X_val)
            y_proba = (
                model.predict_proba(X_val)
                if hasattr(model, "predict_proba")
                else np.zeros((len(X_val), 7))
            )

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
        all_val_proba.append(y_proba)
        all_val_true.append(y_val)

        fold_elapsed = time.time() - fold_start
        fold_times.append(fold_elapsed)
        folds_done = fold_i + 1
        avg_fold_time = np.mean(fold_times)
        eta = avg_fold_time * (n_folds - folds_done)

        print(
            f"Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}  "
            f"BA={metrics['balanced_accuracy']:.4f}  MCC={metrics['mcc']:.4f}  "
            f"[{_fmt_time(fold_elapsed)} | ETA {_fmt_time(eta)}]"
        )
        logger.debug(
            "Fold %d | Acc=%.4f  F1=%.4f  BA=%.4f  MCC=%.4f  time=%.1fs",
            fold_i, metrics["accuracy"], metrics["macro_f1"],
            metrics["balanced_accuracy"], metrics["mcc"], fold_elapsed,
        )

    total_elapsed = time.time() - cv_start
    print(f"  Total CV time: {_fmt_time(total_elapsed)}")

    # Aggregate
    metric_names = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]
    summary = {}
    for m in metric_names:
        vals = [fm[m] for fm in fold_metrics]
        summary[f"{m}_mean"] = np.mean(vals)
        summary[f"{m}_std"] = np.std(vals)

    # Concatenate all OOF predictions in original order
    oof_indices = np.concatenate(all_val_indices)
    oof_preds = np.concatenate(all_val_preds)
    oof_proba = np.concatenate(all_val_proba, axis=0) if all_val_proba[0].ndim == 2 else None
    oof_true = np.concatenate(all_val_true)

    # Sort back to original order
    sort_order = np.argsort(oof_indices)
    oof_preds = oof_preds[sort_order]
    oof_true = oof_true[sort_order]
    if oof_proba is not None:
        oof_proba = oof_proba[sort_order]

    return {
        "model_name": model_name,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "oof_preds": oof_preds,
        "oof_proba": oof_proba,
        "oof_true": oof_true,
    }


def print_cv_summary(results: dict) -> None:
    """Print formatted cross-validation results."""
    name = results["model_name"]
    s = results["summary"]
    print(f"\n{'='*60}")
    print(f"  {name} -- 5-Fold Cross-Validation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:          {s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}")
    print(f"  Macro F1:          {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f}")
    print(f"  Balanced Accuracy: {s['balanced_accuracy_mean']:.4f} +/- {s['balanced_accuracy_std']:.4f}")
    print(f"  MCC:               {s['mcc_mean']:.4f} +/- {s['mcc_std']:.4f}")
    print(f"{'='*60}")
