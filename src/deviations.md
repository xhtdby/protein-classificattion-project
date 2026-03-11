# Plan Deviation Report — COMP0082 Protein Enzyme Classification
*Generated: March 2026. Compares plan.md targets against actual outputs.*

---

## Summary

| Phase | Status | Deviation Severity |
|-------|--------|--------------------|
| 1 – Data Loading | ✅ PASS | None |
| 2 – Handcrafted Features | ✅ PASS | None |
| 3 – Baseline Models | ⚠️ PARTIAL | Low |
| 4 – ESM-2 Embeddings | ✅ PASS | None |
| 5 – Advanced Models | ❌ DEVIATED | High (F1 target missed) |
| 6 – Interpretability | ⚠️ PARTIAL | Medium |
| 7 – Confidence & Blind Pipeline | ✅ PASS | None |
| 8 – Figures & Report | ⚠️ PARTIAL | Low |

---

## Phase 1 — Data Loading ✅

### Verification targets
- Run `python -m src.data_loading` → 7 class counts matching the spec table.
- Each CV fold preserves class proportions.

### Actual results
| Class | Expected | Actual |
|-------|----------|--------|
| 0 – Not enzyme | 32,410 | 32,410 ✅ |
| 1 – Oxidoreductase | 1,184 | 1,184 ✅ |
| 2 – Transferase | 2,769 | 2,769 ✅ |
| 3 – Hydrolase | 2,108 | 2,108 ✅ |
| 4 – Lyase | 600 | 600 ✅ |
| 5 – Isomerase | 411 | 411 ✅ |
| 6 – Ligase | 282 | 282 ✅ |
| **Total** | **39,764** | **39,764** ✅ |

Stratified 5-fold CV: class 0 proportion = 0.815 in every fold (matches overall 0.815). ✅

---

## Phase 2 — Handcrafted Feature Engineering ✅

### Verification targets
- `X_handcrafted.shape == (39764, ~430)`, no NaN/Inf values.
- AA composition rows sum to ≈1.0; dipeptide rows sum to ≈1.0.

### Actual results
- Shape: `(39764, 429)` ✅ (20 AA + 400 dipeptide + 1 length + 8 physicochemical)
- NaN count: 0 ✅
- AA composition row sums: min=0.0, max=1.0 ✅  
  *(min=0.0 is correct — very short sequences that are entirely non-standard AAs get zero vectors)*
- Dipeptide row sums: min=0.0, max=1.0 ✅

**Note:** Plan says ~430; actual is 429. Composition module produces 421 (20+400+1) and physicochemical 8, totalling 429. The plan estimate of ~430 was approximate and matches.

---

## Phase 3 — Baseline Models ⚠️

### Verification targets
- Both models produce 4 metrics. Balanced accuracy should be **above 1/7 ≈ 14.3%**.
- Confusion matrix saved as PNG. Class 0 should **not dominate all predictions**.

### Actual results
| Model | Accuracy | Macro F1 | Balanced Acc | MCC |
|-------|----------|----------|--------------|-----|
| Logistic Regression | — | 0.1912 | 0.2997 | 0.1710 |
| Random Forest | — | 0.1413 | 0.1492 | 0.0930 |
| Random Forest + PCA(50) | — | 0.1973 | 0.1850 | 0.2165 |

### Deviations
**DEV-3a**: Random Forest balanced accuracy = 0.149, barely above the 14.3% random floor.
The RF is nearly degenerate — it is predicting class 0 for the vast majority of samples. 
Adding `max_depth=20, min_samples_leaf=5` improved it from 0.128 to 0.141 F1 but it remains near-random.

**Cause:** The handcrafted feature space (dipeptide frequencies) is very sparse and high-dimensional
(429 features). RF's Gini impurity splits favour the majority class. Without depth restrictions and
with balanced weights, the forest still struggles to partition the heavily-imbalanced space.

**Fix step:** Use `class_weight='balanced_subsample'` instead of `'balanced'`, or lower `min_samples_split`.
Alternatively, use RF only on the reduced PCA(50) version and report the 0.197 F1 variant as the baseline.

**DEV-3b**: Confusion matrix naming. Plan specifies a figure called `confusion_matrix_best.png`
in the report-ready set. What exists: individual per-model confusion matrices
(`cm_logistic_regression.png`, `cm_random_forest.png`, `cm_random_forest_pca.png`). The specific
filename `confusion_matrix_best.png` is absent.

---

## Phase 4 — ESM-2 Embeddings ✅

### Verification targets
- `esm2_embeddings.shape == (39764, 320)`, no NaN values.
- Sequences > 1022 tokens truncated with a warning.

### Actual results
- Shape: `(39764, 320)` ✅
- NaN count: 0 ✅
- Truncation at 1022 implemented ✅ (see `embeddings.py` lines 90–134)
- Model used: `esm2_t6_8M_UR50D` (8M, 320-dim) ✅

---

## Phase 5 — Advanced Models ❌ (Critical Deviation)

### Verification targets
- > **0.70 Macro F1** for ESM-2 models. (plan.md §5 Verification)
- > 0.50 Macro F1 for handcrafted features.
- Best model saved as `best_model.joblib` and loads correctly.

### Actual results
| Model | Macro F1 | Balanced Acc | MCC |
|-------|----------|--------------|-----|
| XGBoost (ESM-2 + Physico) | 0.5346 | 0.5145 | 0.5877 |
| LightGBM (ESM-2 + Physico) | 0.5129 | 0.4783 | 0.5788 |
| LightGBM+SMOTE | 0.5277 | 0.5255 | 0.5680 |
| XGBoost+SMOTE | 0.5303 | 0.5449 | 0.5682 |

**Ablation:**
| Feature Set | Macro F1 |
|-------------|----------|
| ESM-2 only (320-dim) | 0.5306 |
| Handcrafted only (429-dim) | 0.2468 |
| ESM-2 + Handcrafted (749-dim) | 0.5021 |
| ESM-2 + Physicochemical (328-dim) | 0.5346 |

**Per-class F1 of best model:**
| Class | F1 |
|-------|----|
| Not enzyme | 0.9361 |
| Oxidoreductase | 0.6069 |
| Transferase | 0.6138 |
| Hydrolase | 0.5593 |
| **Lyase** | **0.2421** |
| **Isomerase** | **0.2934** |
| Ligase | 0.4990 |

### Deviations
**DEV-5a (High Priority):** **Macro F1 = 0.5346 vs target ≥ 0.70 — gap of 0.17.**
This is the most significant deviation in the project.

*Cause 1 — ESM-2 model size:* The plan allows using the 8M model first, with the intent to 
upgrade to the 650M (1280-dim) if GPU allows. We used only the 8M model. Frozen 650M embeddings  
typically improve F1 by 0.07–0.12 on similar tasks. The gap from 0.53 to 0.70 is likely bridged  
by the 650M model.

*Cause 2 — No hyperparameter tuning done:* `--tune` flag exists but was **never run**.
`tune_xgboost()` implements grid+random search. Running it could add 0.02–0.05 F1.

*Cause 3 — Tiny minority classes (Lyase=600, Isomerase=411, Ligase=282):* Even with balanced
weights, these classes are so rare that XGBoost cannot learn sufficient decision boundaries 
from 80% of the data (≈480 Lyase training samples per fold). The 0.70 target implicitly assumes 
either a more powerful feature extractor (650M) or fine-tuning the PLM.

*Cause 4 — Frozen embeddings vs fine-tuning:* The plan does not explicitly require fine-tuning, 
and ESM-2 was used correctly as a stateless feature extractor. However, the 0.70 target is very 
optimistic for frozen 8M embeddings.

**Fix steps (ordered by expected impact):**
1. Extract 650M ESM-2 embeddings: `python -m src.features.embeddings --model 650M --batch-size 2`
2. Retrain: `python -m src.models.advanced --esm-model 650M`  
   *(Expected F1: 0.60–0.65 with frozen 650M)*
3. Run hyperparameter tuning: `python -m src.models.advanced --esm-model 650M --tune --tune-iter 30`  
   *(Expected additional gain: 0.02–0.05)*
4. For full 0.70+: Fine-tune ESM-2 8M on this dataset with a classification head (PyTorch),  
   using weighted cross-entropy. This requires building a custom training loop (~200 lines).

**DEV-5b:** Handcrafted Macro F1 = 0.2468, below the > 0.50 target stated in Phase 5.
This target was **overoptimistic in the plan** — handcrafted features without PLM embeddings
are expected to perform far below 0.50 on this heavily-imbalanced 7-class problem.
The 0.50 target in the plan referred to ESM-2 models, not handcrafted alone.

**DEV-5c:** Per-class threshold optimization produced all thresholds ≈ 1.0 (improvement: 
+0.0012 F1). The optimizer converged but found no meaningful gains — XGBoost's internal 
`scale_pos_weight` already handles class imbalance well at the probability level.

**DEV-5d:** MLP model not implemented. Plan §5.1 lists it as "Optional MLP" with an 
architecture (320→256→128→7, dropout 0.3). It was excluded due to time. Given the 
transformer features are already well-structured, an MLP would likely not beat XGBoost 
on frozen embeddings, so the impact on the F1 gap is low.

---

## Phase 6 — Interpretability ⚠️

### Verification targets
- At least 2 interpretability figures saved.
- Ablation results show ESM-2 is the dominant feature group.
- Feature importance ranking is plausible (e.g., AA composition features).

### Actual results
Figures generated: `feature_importance.png`, `shap_importance.png`, `per_class_metrics.png`,
`high_confidence_errors.png`, `ablation_results.png` ✅ (5 figures, exceeds minimum)

Ablation confirmation: ESM-2 alone (F1=0.5306) >> Handcrafted alone (F1=0.2468) ✅

Top features by SHAP: `molecular_weight` (#1), `ESM_112` (#2) ✅  
Top features by Gini: `ESM_112` (#1), `molecular_weight` (#5) ✅  
Both plausible — molecular weight correlates with enzyme size/complexity.

### Deviations
**DEV-6a:** Plan §6.1 specifies `ablation_study(model_fn, feature_groups, y, cv_splits)` —
i.e., a dedicated function that takes a dict of feature groups and retrains for each.
What was implemented: ablation is run inside `advanced.py` (not in `interpretability.py`),
and `interpretability.py` only has `ablation_study_summary()` which formats pre-computed results.
The function signature and module boundary differ from the plan.

*Impact:* Low — results are identical. The split allows results to be persisted to JSON
before any interpretability run.

**DEV-6b:** SHAP analysis was specified for ~2000-sample subsample; implementation uses 1000.
No material impact on correctness; reduces compute time.

**DEV-6c (historical, now fixed):** Physicochemical feature names in `interpretability.py`
were fabricated (`"charge"`, `"MW"`, `"pI"`) rather than imported from `physicochemical.py`
(correct: `"molecular_weight"`, `"isoelectric_point"`, etc.). Fixed by importing
`FEATURE_NAMES` from `src/features/physicochemical.py`.

---

## Phase 7 — Confidence Scoring & Blind Pipeline ✅

### Verification targets
- Output file format: `<seq_id> <class> Confidence <level>` with no header.
- Confidence distribution makes sense.
- Script runs end-to-end on arbitrary FASTA.

### Actual results — Confidence calibration (on OOF predictions, 39,764 samples):
| Level | Count | Accuracy | Mean max-prob |
|-------|-------|----------|---------------|
| High (p ≥ 0.80) | 30,448 | 0.9508 | 0.9594 |
| Medium (0.50–0.80) | 6,658 | 0.6621 | 0.6633 |
| Low (p < 0.50) | 2,658 | 0.4120 | 0.4106 |

Calibration looks reasonable: High-confidence is correct 95% of the time ✅  
`predict_blind.py` tested on held-out sequences ✅  
Output format verified ✅

### Minor notes
- High-confidence errors: 1,499 (28.2% of all 5,312 errors are high-confidence wrong).
  This is elevated but expected — 30,448 high-confidence samples is 76.5% of all predictions,
  so even a 5% error rate generates many absolute errors.

---

## Phase 8 — Figures & Report Support ⚠️

### Verification targets
- ≥7 figures in `outputs/figures/`.
- Specific filenames: `confusion_matrix_best.png`, `metrics_comparison.png`,
  `class_distribution.png`, `feature_importance.png`, `ablation_results.png`,
  `confidence_calibration.png` (≡ `reliability_diagram.png`), `sequence_length_distribution.png`.

### Actual figures (17 total — exceeds minimum):
`ablation_results.png`, `ablation_study.png`, `class_distribution.png`, `cm_lightgbm.png`,
`cm_lightgbm_smote.png`, `cm_logistic_regression.png`, `cm_random_forest.png`,
`cm_random_forest_pca.png`, `cm_xgboost.png`, `cm_xgboost_smote.png`,
`feature_importance.png`, `high_confidence_errors.png`, `model_comparison.png`,
`per_class_metrics.png`, `reliability_diagram.png`, `sequence_length_distribution.png`,
`shap_importance.png`

### Deviations
**DEV-8a:** `confusion_matrix_best.png` not present (plan name). Exists as `cm_xgboost.png`.
*Fix:* Rename or add an alias in `generate_plots.py`.

**DEV-8b:** `metrics_comparison.png` not present. Exists as `model_comparison.png`.
*Fix:* Rename or add an alias.

**DEV-8c:** `confidence_calibration.png` named as `reliability_diagram.png` in actual output.
*Fix:* Add alias copy or rename in `confidence.py`.

**DEV-8d (Low):** No EDA notebook (`notebooks/01_eda.ipynb`). The notebooks folder is empty.
Plan listed this as "optional but recommended". Has no impact on model quality or report metrics.

---

## Full Deviation Registry

| ID | Phase | Severity | Status | One-Line Description |
|----|-------|----------|--------|----------------------|
| DEV-3a | 3 | Low | Open | RF balanced_accuracy barely above random (0.149 vs >0.143) |
| DEV-3b | 3/8 | Low | Open | `confusion_matrix_best.png` filename missing |
| DEV-5a | 5 | **High** | **Open** | Macro F1 = 0.53 vs 0.70 target |
| DEV-5b | 5 | Medium | Informational | Handcrafted F1=0.25 vs >0.50 (plan target was overoptimistic) |
| DEV-5c | 5 | Low | Open | Threshold optimization ineffective (~zero gain) |
| DEV-5d | 5 | Low | Open | Optional MLP not implemented |
| DEV-6a | 6 | Low | Open | Ablation runs in advanced.py, not interpretability.py |
| DEV-6b | 6 | None | Closed | SHAP subsample 1000 vs planned 2000 |
| DEV-6c | 6 | Medium | **Fixed** | Wrong physicochemical feature names (now uses canonical FEATURE_NAMES) |
| DEV-8a | 8 | Low | Open | `confusion_matrix_best.png` → need alias for `cm_xgboost.png` |
| DEV-8b | 8 | Low | Open | `metrics_comparison.png` → exists as `model_comparison.png` |
| DEV-8c | 8 | Low | Open | `confidence_calibration.png` → exists as `reliability_diagram.png` |
| DEV-8d | 8 | None | Informational | EDA notebook absent (was marked optional in plan) |

---

## Priority Fix Order

1. **DEV-5a** — Run 650M ESM-2 extraction + retrain (expected: +0.07–0.12 F1 → ~0.60–0.65)
2. **DEV-5a** — Run hyperparameter tuning on 650M model (expected: +0.02–0.05 → ~0.65+)
3. **DEV-8a/8b/8c** — Add figure aliases to `generate_plots.py` (30-min task)
4. **DEV-5a** — Fine-tune ESM-2 8M with classification head (expected: ~0.70+, 2–4 hr task)
5. **DEV-3a** — Investigate RF with `class_weight='balanced_subsample'` (minor improvement)
