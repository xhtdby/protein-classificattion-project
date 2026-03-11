"""
Advanced models: XGBoost, LightGBM, and optional MLP.

Supports ESM-2 embeddings, handcrafted features, or combinations.
Includes feature-combination ablation study.
"""

import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import joblib
import torch
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_loading import load_all_sequences, get_cv_splits, SEED
from src.training import cross_validate_model, print_cv_summary, fmt_time
from src.evaluation import (
    plot_confusion_matrix,
    print_metrics_table,
    plot_metrics_comparison,
    print_classification_report,
    save_results_json,
)

logger = logging.getLogger(__name__)


# -- Hardware detection --------------------------------------------------------

def _detect_hardware() -> dict:
    """Detect available compute hardware and return a device-info dict."""
    info: dict = {"cuda": False, "mps": False, "device_str": "cpu", "gpu_name": None}
    if torch.cuda.is_available():
        info["cuda"] = True
        info["device_str"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps"] = True
        info["device_str"] = "mps"
        info["gpu_name"] = "Apple MPS"
    return info


def _print_hardware(hw: dict) -> None:
    print("\n" + "=" * 72)
    print("  HARDWARE")
    print("=" * 72)
    print(f"  CUDA          : {hw['cuda']}")
    print(f"  MPS           : {hw['mps']}")
    if hw["gpu_name"]:
        print(f"  GPU           : {hw['gpu_name']}")
        if "vram_gb" in hw:
            print(f"  VRAM          : {hw['vram_gb']:.1f} GB")
    print(f"  XGBoost dev   : {'cuda' if hw['cuda'] else 'cpu'}")
    print("=" * 72 + "\n")


# -- Model factories -----------------------------------------------------------

def make_xgboost(hw: dict | None = None, **kwargs) -> XGBClassifier:
    """XGBoost with GPU acceleration if CUDA is available."""
    if hw is None:
        hw = _detect_hardware()
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "verbosity": 0,
        "device": "cuda" if hw["cuda"] else "cpu",
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def make_lightgbm(hw: dict | None = None, **kwargs) -> LGBMClassifier:
    """LightGBM with class_weight='balanced' for imbalance handling."""
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)


# -- Ablation ------------------------------------------------------------------

def run_feature_ablation(
    feature_groups: dict[str, np.ndarray],
    y: np.ndarray,
    cv_splits: list,
    hw: dict | None = None,
) -> list[dict]:
    """Ablation over single groups and all-combined."""
    if hw is None:
        hw = _detect_hardware()

    def _xgb():
        return make_xgboost(hw)

    results = []
    for name, X in feature_groups.items():
        r = cross_validate_model(_xgb, X, y, cv_splits=cv_splits, model_name=f"{name} only")
        print_cv_summary(r)
        results.append(r)

    if len(feature_groups) > 1:
        X_all = np.hstack(list(feature_groups.values()))
        combined = " + ".join(feature_groups.keys())
        r = cross_validate_model(_xgb, X_all, y, cv_splits=cv_splits, model_name=combined)
        print_cv_summary(r)
        results.append(r)

    return results


# -- Hyperparameter tuning -----------------------------------------------------

def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list,
    hw: dict | None = None,
    n_iter: int = 20,
) -> dict:
    """Random search over XGBoost hyperparameters, scored by Macro F1.

    Returns dict with best params and CV results.
    """
    import itertools
    if hw is None:
        hw = _detect_hardware()

    param_grid = {
        "max_depth": [4, 6, 8],
        "n_estimators": [300, 500, 800],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }
    # Generate all combos, then sample n_iter
    keys = list(param_grid.keys())
    all_combos = [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]
    rng = np.random.default_rng(SEED)
    if len(all_combos) > n_iter:
        indices = rng.choice(len(all_combos), size=n_iter, replace=False)
        candidates = [all_combos[i] for i in indices]
    else:
        candidates = all_combos

    print(f"\n{'='*72}")
    print(f"  HYPERPARAMETER TUNING — {len(candidates)} XGBoost configurations")
    print(f"{'='*72}")

    best_f1 = -1.0
    best_result = None
    best_params = None

    for i, params in enumerate(candidates):
        label = " | ".join(f"{k}={v}" for k, v in params.items())
        r = cross_validate_model(
            lambda p=params: make_xgboost(hw, **p),
            X, y, cv_splits=cv_splits, use_class_weight=True,
            model_name=f"Tune [{i+1}/{len(candidates)}]",
        )
        f1 = r["summary"]["macro_f1_mean"]
        marker = " ** NEW BEST" if f1 > best_f1 else ""
        print(f"  [{i+1:>2d}/{len(candidates)}] F1={f1:.4f}  {label}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_result = r
            best_params = params

    print(f"\n  Best params: {best_params}  (Macro F1={best_f1:.4f})")
    return {"params": best_params, "result": best_result, "best_f1": best_f1}


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Advanced model training",
        epilog="Examples:\n"
               "  python -m src.models.advanced                    # train with 8M ESM-2\n"
               "  python -m src.models.advanced --esm-model 650M   # train with 650M ESM-2\n"
               "  python -m src.models.advanced --tune              # add hyperparam search\n"
               "  python -m src.models.advanced --esm-model 650M --tune --tune-iter 30\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--esm-model", default="8M", choices=["8M", "650M"],
        help="ESM-2 model size to use (must have embeddings pre-extracted). Default: 8M",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter tuning before final model comparison",
    )
    parser.add_argument(
        "--tune-iter", type=int, default=20,
        help="Number of random search iterations for tuning (default: 20)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    hw = _detect_hardware()
    _print_hardware(hw)

    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = project_root / "outputs" / "figures"
    features_dir = project_root / "outputs" / "features"
    models_dir = project_root / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data -------------------------------------------------------------
    t0 = time.time()
    print("Loading sequences...")
    df = load_all_sequences(project_root)
    y = df["label"].values
    cv_splits = get_cv_splits(y)
    print(f"  Loaded {len(df):,} sequences  [{fmt_time(time.time() - t0)}]")

    # -- Load features ---------------------------------------------------------
    from src.features.embeddings import get_cache_filename

    ESM_ALIAS = {"8M": "esm2_t6_8M_UR50D", "650M": "esm2_t33_650M_UR50D"}
    esm_model_name = ESM_ALIAS[args.esm_model]
    esm_cache = get_cache_filename(esm_model_name)

    feature_groups: dict[str, np.ndarray] = {}
    for tag, fname in [("Handcrafted", "handcrafted_features.npy"),
                       ("ESM-2", esm_cache)]:
        p = features_dir / fname
        if p.exists():
            arr = np.load(p)
            feature_groups[tag] = arr
            print(f"  Loaded {tag} features: shape={arr.shape}")
        elif tag == "ESM-2":
            print(f"\nERROR: ESM-2 embeddings not found at {p.name}")
            print(f"  Run:  python -m src.features.embeddings --model {args.esm_model}")
            sys.exit(1)

    if not feature_groups:
        print("\nERROR: No pre-computed features found.")
        print("  Run:  python -m src.models.baseline   (handcrafted)")
        print(f"        python -m src.features.embeddings --model {args.esm_model}")
        sys.exit(1)

    # -- Determine primary features (ESM-2 preferred) -------------------------
    esm_X = feature_groups.get("ESM-2")
    hc_X = feature_groups.get("Handcrafted")

    if esm_X is None:
        print(f"\nERROR: ESM-2 embeddings required. Run:  python -m src.features.embeddings --model {args.esm_model}")
        sys.exit(1)

    # -- Build feature combos for model training & ablation --------------------
    # Physicochemical = last 8 cols of Handcrafted (composition:421 + physico:8)
    physico_X = hc_X[:, -8:] if hc_X is not None else None

    # Primary training features: ESM-2 + Physicochemical when available
    # (ablation showed this beats ESM-2 alone)
    if physico_X is not None:
        primary_X = np.hstack([esm_X, physico_X])
        primary_label = "ESM-2 + Physico"
        feature_source = "ESM-2 + Physicochemical"
    else:
        primary_X = esm_X
        primary_label = "ESM-2"
        feature_source = "ESM-2"
    print(f"  Primary features: {primary_label}  shape={primary_X.shape}")

    combos: dict[str, np.ndarray] = {"ESM-2": esm_X}
    if hc_X is not None:
        combos["Handcrafted"] = hc_X
        combos["ESM-2 + Handcrafted"] = np.hstack([esm_X, hc_X])
    if physico_X is not None:
        combos["ESM-2 + Physicochemical"] = primary_X

    # === MODEL TRAINING (all on primary features: ESM-2 + Physico) =============
    all_results: list[dict] = []

    # -- Optional hyperparameter tuning ----------------------------------------
    best_xgb_params: dict = {}
    if args.tune:
        tune_result = tune_xgboost(
            primary_X, y, cv_splits, hw=hw, n_iter=args.tune_iter,
        )
        best_xgb_params = tune_result["params"]
        all_results.append(tune_result["result"])
        tune_result["result"]["model_name"] = f"XGBoost-Tuned ({primary_label})"
        print_cv_summary(tune_result["result"])
        plot_confusion_matrix(
            tune_result["result"]["oof_true"], tune_result["result"]["oof_preds"],
            save_path=figures_dir / "cm_xgboost_tuned.png",
            title=f"XGBoost-Tuned ({primary_label}) -- Confusion Matrix",
        )

    # -- XGBoost (GPU) ---------------------------------------------------------
    try:
        xgb_results = cross_validate_model(
            lambda: make_xgboost(hw, **best_xgb_params), primary_X, y,
            cv_splits=cv_splits, use_class_weight=True,
            model_name=f"XGBoost ({primary_label})",
        )
        print_cv_summary(xgb_results)
        plot_confusion_matrix(
            xgb_results["oof_true"], xgb_results["oof_preds"],
            save_path=figures_dir / "cm_xgboost.png",
            title=f"XGBoost ({primary_label}) -- Confusion Matrix",
        )
        all_results.append(xgb_results)
    except Exception:
        print("\nFATAL: XGBoost training failed:")
        traceback.print_exc()
        sys.exit(1)

    # -- LightGBM (class_weight='balanced') ------------------------------------
    try:
        lgbm_results = cross_validate_model(
            lambda: make_lightgbm(hw), primary_X, y,
            cv_splits=cv_splits, model_name=f"LightGBM ({primary_label})",
        )
        print_cv_summary(lgbm_results)
        plot_confusion_matrix(
            lgbm_results["oof_true"], lgbm_results["oof_preds"],
            save_path=figures_dir / "cm_lightgbm.png",
            title=f"LightGBM ({primary_label}) -- Confusion Matrix",
        )
        all_results.append(lgbm_results)
    except Exception:
        print("\nFATAL: LightGBM training failed:")
        traceback.print_exc()
        sys.exit(1)

    # -- LightGBM + SMOTE ------------------------------------------------------
    try:
        lgbm_smote_results = cross_validate_model(
            lambda: make_lightgbm(hw), primary_X, y,
            cv_splits=cv_splits, use_smote=True,
            model_name=f"LightGBM+SMOTE ({primary_label})",
        )
        print_cv_summary(lgbm_smote_results)
        plot_confusion_matrix(
            lgbm_smote_results["oof_true"], lgbm_smote_results["oof_preds"],
            save_path=figures_dir / "cm_lightgbm_smote.png",
            title=f"LightGBM+SMOTE ({primary_label}) -- Confusion Matrix",
        )
        all_results.append(lgbm_smote_results)
    except Exception:
        print("\nWARN: LightGBM+SMOTE failed (skipping):")
        traceback.print_exc()

    # -- XGBoost + SMOTE -------------------------------------------------------
    try:
        xgb_smote_results = cross_validate_model(
            lambda: make_xgboost(hw, **best_xgb_params), primary_X, y,
            cv_splits=cv_splits, use_smote=True,
            model_name=f"XGBoost+SMOTE ({primary_label})",
        )
        print_cv_summary(xgb_smote_results)
        plot_confusion_matrix(
            xgb_smote_results["oof_true"], xgb_smote_results["oof_preds"],
            save_path=figures_dir / "cm_xgboost_smote.png",
            title=f"XGBoost+SMOTE ({primary_label}) -- Confusion Matrix",
        )
        all_results.append(xgb_smote_results)
    except Exception:
        print("\nWARN: XGBoost+SMOTE failed (skipping):")
        traceback.print_exc()

    # === ABLATION STUDY (4 feature combos with XGBoost) =======================
    print("\n" + "=" * 72)
    print("  FEATURE ABLATION STUDY")
    print("=" * 72)

    ablation_results: list[dict] = []
    for combo_name, combo_X in combos.items():
        r = cross_validate_model(
            lambda: make_xgboost(hw, **best_xgb_params), combo_X, y,
            cv_splits=cv_splits, use_class_weight=True,
            model_name=f"Ablation: {combo_name}",
        )
        print_cv_summary(r)
        ablation_results.append(r)

    print_metrics_table(ablation_results)
    plot_metrics_comparison(ablation_results, save_path=figures_dir / "ablation_results.png")

    # === SAVE BEST MODEL ======================================================
    best = max(all_results, key=lambda r: r["summary"]["macro_f1_mean"])
    best_name = best["model_name"]
    print(f"\nBest: {best_name}  Macro F1={best['summary']['macro_f1_mean']:.4f}")

    from sklearn.preprocessing import StandardScaler
    t_fit = time.time()
    print("Retraining best model on full dataset...")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(primary_X)
    y_train_final = y

    fit_kwargs = {}
    if "SMOTE" in best_name:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=SEED)
        X_sc, y_train_final = smote.fit_resample(X_sc, y)
    elif "XGBoost" in best_name:
        from sklearn.utils.class_weight import compute_sample_weight
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train_final)

    if "XGBoost" in best_name:
        final = make_xgboost(hw, **best_xgb_params)
    else:
        final = make_lightgbm(hw)
    final.fit(X_sc, y_train_final, **fit_kwargs)

    # -- Per-class threshold optimization on OOF probabilities ----------------
    from src.training import optimize_thresholds
    thresholds, thresh_metrics = optimize_thresholds(
        best["oof_true"], best["oof_proba"], n_classes=7,
    )
    print(f"\n  Optimized per-class thresholds: {np.round(thresholds, 3).tolist()}")
    print(f"  Threshold-opt metrics:  Macro F1={thresh_metrics['macro_f1']:.4f}  "
          f"BA={thresh_metrics['balanced_accuracy']:.4f}  MCC={thresh_metrics['mcc']:.4f}")

    save_path = models_dir / "best_model.joblib"
    joblib.dump({
        "model": final, "scaler": scaler, "feature_source": feature_source,
        "esm_model_name": esm_model_name,
        "esm_embedding_dim": esm_X.shape[1],
        "xgb_params": best_xgb_params if "XGBoost" in best_name else {},
        "thresholds": thresholds,
        "oof_preds": best["oof_preds"], "oof_proba": best["oof_proba"],
        "oof_true": best["oof_true"], "cv_scores": best["summary"],
    }, save_path)
    print(f"  Saved -> {save_path}  [{fmt_time(time.time() - t_fit)}]")

    # === FINAL SUMMARY ========================================================
    print_metrics_table(all_results)
    print("\n--- Best Model Classification Report ---")
    print_classification_report(best["oof_true"], best["oof_preds"])

    # -- Persist structured results to JSON ------------------------------------
    results_dir = project_root / "outputs"
    advanced_data = {
        "esm_model": esm_model_name,
        "feature_source": feature_source,
        "primary_features_shape": list(primary_X.shape),
        "models": [
            {"model_name": r["model_name"], "summary": r["summary"]}
            for r in all_results
        ],
        "ablation": [
            {"model_name": r["model_name"], "summary": r["summary"]}
            for r in ablation_results
        ],
        "best_model": best_name,
        "best_cv_scores": best["summary"],
        "xgb_params": best_xgb_params if "XGBoost" in best_name else {},
        "thresholds": thresholds.tolist(),
        "threshold_metrics": thresh_metrics,
    }
    save_results_json(advanced_data, results_dir / "advanced_results.json")

    print(f"\nTotal wall time: {fmt_time(time.time() - t0)}")
