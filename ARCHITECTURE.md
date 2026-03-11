# Architecture Guide — Protein Enzyme Classification

> **Audience**: A developer who needs to understand, debug, or extend this codebase.
> Read this before touching any file.

---

## 1. What This Project Does

Classifies protein amino-acid sequences into **7 enzyme classes** (EC 0–6) using
supervised machine learning. The dataset is heavily imbalanced (~82% class 0 "Not
an enzyme"), so every component is designed around that constraint.

| Class | Label          | Count  | % of Total |
|-------|----------------|--------|------------|
| 0     | Not an enzyme  | 32,410 | 81.5%      |
| 1     | Oxidoreductase | 1,184  | 3.0%       |
| 2     | Transferase    | 2,769  | 7.0%       |
| 3     | Hydrolase      | 2,108  | 5.3%       |
| 4     | Lyase          | 600    | 1.5%       |
| 5     | Isomerase      | 411    | 1.0%       |
| 6     | Ligase         | 282    | 0.7%       |

---

## 2. Project Layout

```
protein-classification/
│
├── class0_rep_seq.fasta.txt       ← Raw data (7 FASTA files, one per class)
├── ec1_rep_seq.fasta.txt          ←   ...
│   ...
├── ec6_rep_seq.fasta.txt          ←   ...
│
├── src/                           ← ALL source code lives here
│   ├── data_loading.py            ← FASTA I/O, labels, CV splits
│   │
│   ├── features/                  ← Feature extraction (3 strategies)
│   │   ├── composition.py         ←   AA composition (20d) + dipeptide (400d) + length (1d)
│   │   ├── physicochemical.py     ←   MW, pI, GRAVY, charge, aromaticity, SS fractions (8d)
│   │   └── embeddings.py          ←   ESM-2 protein language model (320d or 1280d)
│   │
│   ├── models/                    ← Model definitions + training scripts
│   │   ├── baseline.py            ←   LogReg + Random Forest (handcrafted features)
│   │   └── advanced.py            ←   XGBoost + LightGBM + ablation (ESM-2 features)
│   │
│   ├── training.py                ← Cross-validation loop (leak-safe)
│   ├── evaluation.py              ← Metrics, confusion matrices, comparison charts
│   ├── interpretability.py        ← Feature importance, SHAP, confusion analysis
│   ├── confidence.py              ← Probability → High/Medium/Low + calibration
│   ├── predict_blind.py           ← Blind challenge prediction pipeline
│   └── generate_plots.py          ← Combined comparison & ablation figures
│
├── outputs/
│   ├── features/                  ← Cached .npy feature matrices (gitignored)
│   ├── models/                    ← Saved model artifacts (.joblib)
│   ├── figures/                   ← All generated plots (.png)
│   └── predictions/               ← Blind challenge output files
│
├── notebooks/                     ← Exploratory analysis (optional)
├── requirements.txt               ← Pinned dependencies
├── plan.md                        ← Original phased execution plan
└── README.md                      ← Quick-start + results summary
```

---

## 3. Data Flow (Pipeline Order)

```
FASTA files ──► data_loading.py ──► DataFrame (seq_id, sequence, label, length)
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            composition.py      physicochemical.py      embeddings.py
            (421-dim)            (8-dim)                (320-dim ESM-2)
                    │                    │                    │
                    └───── np.hstack ────┘                    │
                         = "Handcrafted"                  = "ESM-2"
                          (429-dim)                       (320-dim)
                              │                              │
                              ▼                              ▼
                        baseline.py                    advanced.py
                     (LogReg, RF on 429d)          (XGB, LGBM on 320d)
                              │                              │
                              └──────────┬───────────────────┘
                                         ▼
                                    training.py
                              (stratified 5-fold CV,
                               scaler on train, SMOTE
                               on train, sample_weight)
                                         │
                                         ▼
                                   evaluation.py
                              (Acc, F1, BA, MCC, CMs)
                                         │
                        ┌────────────────┼────────────────┐
                        ▼                ▼                ▼
               interpretability.py  confidence.py   predict_blind.py
                (importance, SHAP,  (calibration,   (load model +
                 error analysis)     reliability)    FASTA → output)
```

### Execution Order

```bash
# 1. Verify data
python -m src.data_loading

# 2. Extract ESM-2 embeddings (GPU, ~50 min on RTX 3060)
python -m src.features.embeddings

# 3. Run baseline models (handcrafted features, ~2 min)
python -m src.models.baseline

# 4. Run advanced models + ablation (~35 min)
python -m src.models.advanced

# 5. Interpretability (feature importance, SHAP, confusion, ~10s)
python -m src.interpretability

# 6. Confidence calibration (~5s)
python -m src.confidence

# 7. Generate combined plots (~1s)
python -m src.generate_plots

# 8. Blind predictions (on actual test FASTA when available)
python -m src.predict_blind --fasta <blind_test.fasta>
```

---

## 4. Module-by-Module Reference

### `src/data_loading.py`

| Symbol | Purpose |
|--------|---------|
| `SEED = 42` | Global random seed, imported by all other modules |
| `FASTA_LABEL_MAP` | Maps filename → integer label (0–6) |
| `CLASS_NAMES` | Human-readable labels for the 7 classes |
| `load_all_sequences(root)` | Parse all 7 FASTA files → DataFrame |
| `get_cv_splits(labels, n_splits=5)` | Stratified K-Fold indices |

**Key rule**: This is the single source of truth for `SEED`. Every other module
imports it from here.

### `src/features/composition.py`

Extracts sequence-derived numeric features:
- **AA composition** (20-dim): normalised frequency of each canonical amino acid
- **Dipeptide frequencies** (400-dim): normalised frequency of each AA pair
- **Sequence length** (1-dim): raw integer length

Total: 421 features per sequence. Cached to `outputs/features/handcrafted_features.npy`.

### `src/features/physicochemical.py`

Extracts BioPython `ProteinAnalysis` properties:
- Molecular weight, isoelectric point, aromaticity, GRAVY
- Net charge at pH 7.0
- Secondary structure fractions (helix, turn, sheet)

Total: 8 features. Combined with composition → "Handcrafted" = 429 features.

### `src/features/embeddings.py`

Runs ESM-2 (`esm2_t6_8M_UR50D`, 8M params) protein language model:
- Tokenizes each sequence, runs forward pass, mean-pools last-layer representations
- Produces 320-dim embedding per sequence
- Handles empty sequences (→ zero vector), truncates >1022 tokens
- GPU-accelerated with batch processing

Cached to `outputs/features/esm2_embeddings.npy`.

### `src/training.py`

The leak-safe cross-validation engine. **All model training flows through here.**

| Symbol | Purpose |
|--------|---------|
| `fmt_time(seconds)` | Human-readable duration formatting |
| `cross_validate_model(...)` | Stratified K-Fold CV with options |

**`cross_validate_model` parameters:**
- `model_fn`: Callable that returns a fresh model instance
- `use_scaler`: Fit `StandardScaler` on train fold only (default: True)
- `use_smote`: Apply SMOTE oversampling on train fold only
- `use_class_weight`: Compute balanced `sample_weight` and pass to `.fit()` — use for XGBoost

**Data-leakage guarantees:**
1. Scaler `.fit()` on training fold only
2. SMOTE on training fold only
3. `sample_weight` computed from training fold labels only
4. Metrics on unmodified validation fold

**Returns** a dict with: `model_name`, `fold_metrics`, `summary` (mean ± std),
`oof_preds`, `oof_proba`, `oof_true` (out-of-fold predictions in original order).

### `src/models/baseline.py`

Two baseline classifiers on handcrafted features (429-dim):
- **Logistic Regression** — `class_weight='balanced'`, L-BFGS solver
- **Random Forest** — 300 trees, `class_weight='balanced'`

These establish the lower bound. Handcrafted features alone are very weak for
minority classes (RF gets ~0 F1 on classes 1–6).

### `src/models/advanced.py`

Four model configurations on ESM-2 embeddings (320-dim):
- **XGBoost** with balanced `sample_weight` (GPU-accelerated)
- **LightGBM** with `class_weight='balanced'`
- **XGBoost + SMOTE**
- **LightGBM + SMOTE**

Plus a **feature ablation study** (4 combos: ESM-2 only, Handcrafted only,
ESM-2 + Handcrafted, ESM-2 + Physicochemical).

Selects the best model by Macro F1, retrains on full data, and saves to
`outputs/models/best_model.joblib`.

**Saved artifact contents:**
```python
{
    "model": <fitted model>,
    "scaler": <StandardScaler fitted on full data>,
    "feature_source": "ESM-2",
    "oof_preds": np.ndarray,    # out-of-fold predictions (honest)
    "oof_proba": np.ndarray,    # out-of-fold probabilities
    "oof_true": np.ndarray,     # true labels
    "cv_scores": dict,          # mean ± std for 4 metrics
}
```

### `src/evaluation.py`

Plotting and metrics utilities used by baseline and advanced scripts:
- `compute_metrics(y_true, y_pred)` → dict with accuracy, macro_f1, balanced_accuracy, MCC
- `plot_confusion_matrix(...)` → normalised heatmap PNG
- `print_metrics_table(results_list)` → formatted comparison table
- `plot_metrics_comparison(results_list)` → grouped bar chart
- `plot_class_distribution(labels)` → class balance bar chart (log scale)
- `plot_sequence_length_distribution(lengths, labels)` → per-class histograms

### `src/interpretability.py`

Three interpretability analyses (loaded model from artifact):
1. **Feature importance** — Gini (tree-based) or permutation importance
2. **Class confusion analysis** — per-class P/R/F1, most confused pairs, high-confidence errors
3. **SHAP values** — TreeExplainer for tree models, bar plot of top features

Uses the artifact scaler for feature scaling and OOF predictions for confusion
analysis (honest evaluation).

### `src/confidence.py`

Maps `max(predict_proba)` to confidence levels:
- **High** (p ≥ 0.80) → challenge score ±1
- **Medium** (0.50 ≤ p < 0.80) → challenge score ±0.5
- **Low** (p < 0.50) → challenge score 0

Also generates a **reliability diagram** (calibration plot) showing prediction
probability vs actual accuracy across bins.

### `src/predict_blind.py`

End-to-end prediction pipeline for the blind challenge:
1. Loads saved model artifact
2. Parses input FASTA
3. Extracts the correct features based on `feature_source` in the artifact
4. Validates feature dimensions match the model
5. Scales, predicts, assigns confidence
6. Writes output in required format: `{seq_id} {class} Confidence {level}`

```bash
python -m src.predict_blind --fasta blind_test.fasta
```

### `src/generate_plots.py`

Standalone script that generates two summary figures from recorded CV results:
- **Model comparison** — all 6 models (2 baseline + 4 advanced) across 4 metrics
- **Ablation study** — 4 feature combos with XGBoost

Uses hardcoded results for reproducibility (update if models are retrained).

---

## 5. Key Design Decisions

### Why ESM-2 dominates handcrafted features
ESM-2 is a protein language model pre-trained on millions of sequences. Its 320-dim
embeddings capture evolutionary, structural, and functional information that simple
composition/physicochemical features cannot. Ablation confirms: ESM-2 alone (F1~0.52)
vastly outperforms handcrafted alone (F1~0.17). Adding handcrafted features to ESM-2
actually hurts due to noise dilution.

### Class imbalance handling
Three complementary strategies:
1. **`class_weight='balanced'`** — LightGBM's built-in parameter
2. **`sample_weight`** — Balanced weights passed to XGBoost's `.fit()`
3. **SMOTE** — Synthetic oversampling on training fold only

SMOTE boosts balanced accuracy (+0.13) at a small accuracy tradeoff.

### Data leakage prevention
The CV loop in `training.py` is the single enforcement point. All preprocessing
(scaling, SMOTE, sample_weight) happens **inside** the fold loop, fitted on
training data only. ESM-2 embeddings are pre-computed (stateless forward pass,
no label info) so they're safe to cache once.

---

## 6. Adding a New Model

1. Define a factory function in `src/models/advanced.py`:
   ```python
   def make_my_model(**kwargs) -> MyClassifier:
       params = {"random_state": SEED, ...}
       params.update(kwargs)
       return MyClassifier(**params)
   ```

2. Add a CV call in the `__main__` block:
   ```python
   my_results = cross_validate_model(
       make_my_model, esm_X, y,
       cv_splits=cv_splits,
       use_class_weight=True,  # if model doesn't have built-in class_weight
       model_name="MyModel (ESM-2)",
   )
   ```

3. If the model has built-in class weighting (like LightGBM), set it in the
   factory and use `use_class_weight=False`.

4. If using SMOTE, pass `use_smote=True` instead of `use_class_weight`.

---

## 7. Adding New Features

1. Create `src/features/my_features.py` with:
   - `extract_my_features(sequences: list[str]) -> np.ndarray`
   - `get_feature_names() -> list[str]`

2. Cache to `outputs/features/my_features.npy` in its `__main__` block.

3. Load in `advanced.py`'s feature_groups dict for ablation testing.

4. Update `predict_blind.py`'s `extract_features()` if using for predictions.

---

## 8. Common Pitfalls

| Pitfall | How to avoid |
|---------|-------------|
| Forgetting to activate venv | Use `.venv\Scripts\Activate.ps1` or full path `.venv\Scripts\python.exe` |
| Stale `.pyc` after edits | Delete `__pycache__/` dirs: `Get-ChildItem -Recurse __pycache__ \| Remove-Item -Recurse` |
| ESM-2 OOM on GPU | Reduce batch_size (16 works for RTX 3060 6GB). Sequences >1022 tokens are truncated. |
| NaN in features | 86 empty sequences exist in class 0. All feature extractors handle them (zero vectors). |
| `generate_plots.py` out of sync | Values are hardcoded. Update them after retraining models. |
| `sample_weight` + SMOTE | Don't use both — `cross_validate_model` already skips sample_weight when SMOTE is active. |

---

## 9. Output Files Reference

| File | Generated by | Description |
|------|-------------|-------------|
| `outputs/features/esm2_embeddings.npy` | `embeddings.py` | (39764, 320) float32 |
| `outputs/features/handcrafted_features.npy` | `baseline.py` | (39764, 429) float64 |
| `outputs/features/feature_names.npy` | `baseline.py` | 429 string names |
| `outputs/models/best_model.joblib` | `advanced.py` | Best model + scaler + OOF data |
| `outputs/figures/cm_*.png` | `baseline.py`, `advanced.py` | Confusion matrices |
| `outputs/figures/model_comparison.png` | `generate_plots.py` | All models bar chart |
| `outputs/figures/ablation_study.png` | `generate_plots.py` | Feature ablation chart |
| `outputs/figures/feature_importance.png` | `interpretability.py` | Gini importance top-30 |
| `outputs/figures/shap_importance.png` | `interpretability.py` | SHAP top-20 |
| `outputs/figures/per_class_metrics.png` | `interpretability.py` | P/R/F1 per class |
| `outputs/figures/reliability_diagram.png` | `confidence.py` | Calibration plot |
| `outputs/figures/class_distribution.png` | `evaluation.py` | Class balance chart |
| `outputs/figures/sequence_length_distribution.png` | `evaluation.py` | Length histograms |
| `outputs/predictions/blind_predictions.txt` | `predict_blind.py` | Challenge output |

---

## 10. Environment

- **Python** 3.10+ (developed on 3.13)
- **GPU**: NVIDIA RTX 3060 Laptop (6GB VRAM, CUDA 12.4)
- **PyTorch** 2.6.0+cu124 (for ESM-2 only)
- **XGBoost** 3.2.0 (GPU-accelerated training)
- **LightGBM** 4.6.0 (CPU, pip build)
- All dependencies pinned in `requirements.txt`
