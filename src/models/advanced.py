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
from src.training import cross_validate_model, print_cv_summary, _fmt_time
from src.evaluation import (
    plot_confusion_matrix,
    print_metrics_table,
    plot_metrics_comparison,
    print_classification_report,
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
    """XGBoost with GPU if CUDA is available (XGBoost 2.0+ device param)."""
    if hw is None:
        hw = _detect_hardware()
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "verbosity": 0,
        "device": "cuda" if hw["cuda"] else "cpu",
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def make_lightgbm(hw: dict | None = None, **kwargs) -> LGBMClassifier:
    """LightGBM; uses GPU device if available and built with GPU support."""
    if hw is None:
        hw = _detect_hardware()
    device_type = "gpu" if hw["cuda"] else "cpu"
    params = {
        "objective": "multiclass",
        "num_class": 7,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "class_weight": "balanced",   # correct imbalance for multiclass
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
        "device_type": device_type,
    }
    params.update(kwargs)
    clf = LGBMClassifier(**params)
    # Silently fall back to CPU if the GPU build is unavailable
    if device_type == "gpu":
        import lightgbm as lgb
        try:
            lgb.train(
                {"objective": "multiclass", "num_class": 2,
                 "device_type": "gpu", "verbose": -1},
                lgb.Dataset(np.zeros((10, 2)), label=[0] * 5 + [1] * 5),
                num_boost_round=1,
            )
        except Exception:
            logger.info("LightGBM GPU not available -- falling back to CPU.")
            clf.set_params(device_type="cpu")
    return clf


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


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced model training")
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
    print(f"  Loaded {len(df):,} sequences  [{_fmt_time(time.time() - t0)}]")

    # -- Load features ---------------------------------------------------------
    feature_groups: dict[str, np.ndarray] = {}
    for tag, fname in [("Handcrafted", "handcrafted_features.npy"),
                       ("ESM-2", "esm2_embeddings.npy")]:
        p = features_dir / fname
        if p.exists():
            arr = np.load(p)
            feature_groups[tag] = arr
            print(f"  Loaded {tag} features: shape={arr.shape}")

    if not feature_groups:
        print("\nERROR: No pre-computed features found.")
        print("  Run:  python -m src.models.baseline   (handcrafted)")
        print("        python -m src.features.embeddings (ESM-2)")
        sys.exit(1)

    # -- Determine primary features (ESM-2 preferred) -------------------------
    esm_X = feature_groups.get("ESM-2")
    hc_X = feature_groups.get("Handcrafted")

    if esm_X is None:
        print("\nERROR: ESM-2 embeddings required. Run:  python -m src.features.embeddings")
        sys.exit(1)

    # -- Build feature combos for model training & ablation --------------------
    # Physicochemical = last 8 cols of Handcrafted (composition:421 + physico:8)
    physico_X = hc_X[:, -8:] if hc_X is not None else None

    combos: dict[str, np.ndarray] = {"ESM-2": esm_X}
    if hc_X is not None:
        combos["Handcrafted"] = hc_X
        combos["ESM-2 + Handcrafted"] = np.hstack([esm_X, hc_X])
    if physico_X is not None:
        combos["ESM-2 + Physicochemical"] = np.hstack([esm_X, physico_X])

    # === MODEL TRAINING (all on ESM-2 features) ===============================
    all_results: list[dict] = []

    # -- XGBoost (GPU) ---------------------------------------------------------
    try:
        xgb_results = cross_validate_model(
            lambda: make_xgboost(hw), esm_X, y,
            cv_splits=cv_splits, model_name="XGBoost (ESM-2)",
        )
        print_cv_summary(xgb_results)
        plot_confusion_matrix(
            xgb_results["oof_true"], xgb_results["oof_preds"],
            save_path=figures_dir / "cm_xgboost.png",
            title="XGBoost (ESM-2) -- Confusion Matrix",
        )
        all_results.append(xgb_results)
    except Exception:
        print("\nFATAL: XGBoost training failed:")
        traceback.print_exc()
        sys.exit(1)

    # -- LightGBM (class_weight='balanced') ------------------------------------
    try:
        lgbm_results = cross_validate_model(
            lambda: make_lightgbm(hw), esm_X, y,
            cv_splits=cv_splits, model_name="LightGBM (ESM-2)",
        )
        print_cv_summary(lgbm_results)
        plot_confusion_matrix(
            lgbm_results["oof_true"], lgbm_results["oof_preds"],
            save_path=figures_dir / "cm_lightgbm.png",
            title="LightGBM (ESM-2) -- Confusion Matrix",
        )
        all_results.append(lgbm_results)
    except Exception:
        print("\nFATAL: LightGBM training failed:")
        traceback.print_exc()
        sys.exit(1)

    # -- LightGBM + SMOTE ------------------------------------------------------
    try:
        lgbm_smote_results = cross_validate_model(
            lambda: make_lightgbm(hw), esm_X, y,
            cv_splits=cv_splits, use_smote=True,
            model_name="LightGBM+SMOTE (ESM-2)",
        )
        print_cv_summary(lgbm_smote_results)
        plot_confusion_matrix(
            lgbm_smote_results["oof_true"], lgbm_smote_results["oof_preds"],
            save_path=figures_dir / "cm_lightgbm_smote.png",
            title="LightGBM+SMOTE (ESM-2) -- Confusion Matrix",
        )
        all_results.append(lgbm_smote_results)
    except Exception:
        print("\nWARN: LightGBM+SMOTE failed (skipping):")
        traceback.print_exc()

    # -- XGBoost + SMOTE -------------------------------------------------------
    try:
        xgb_smote_results = cross_validate_model(
            lambda: make_xgboost(hw), esm_X, y,
            cv_splits=cv_splits, use_smote=True,
            model_name="XGBoost+SMOTE (ESM-2)",
        )
        print_cv_summary(xgb_smote_results)
        plot_confusion_matrix(
            xgb_smote_results["oof_true"], xgb_smote_results["oof_preds"],
            save_path=figures_dir / "cm_xgboost_smote.png",
            title="XGBoost+SMOTE (ESM-2) -- Confusion Matrix",
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
            lambda: make_xgboost(hw), combo_X, y,
            cv_splits=cv_splits, model_name=f"Ablation: {combo_name}",
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
    X_sc = scaler.fit_transform(esm_X)
    y_train_final = y

    if "SMOTE" in best_name:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=SEED)
        X_sc, y_train_final = smote.fit_resample(X_sc, y)

    if "XGBoost" in best_name:
        final = make_xgboost(hw)
    else:
        final = make_lightgbm(hw)
    final.fit(X_sc, y_train_final)

    save_path = models_dir / "best_model.joblib"
    joblib.dump({
        "model": final, "scaler": scaler, "feature_source": "ESM-2",
        "oof_preds": best["oof_preds"], "oof_proba": best["oof_proba"],
        "oof_true": best["oof_true"], "cv_scores": best["summary"],
    }, save_path)
    print(f"  Saved -> {save_path}  [{_fmt_time(time.time() - t_fit)}]")

    # === FINAL SUMMARY ========================================================
    print_metrics_table(all_results)
    print("\n--- Best Model Classification Report ---")
    print_classification_report(best["oof_true"], best["oof_preds"])
    print(f"\nTotal wall time: {_fmt_time(time.time() - t0)}")
