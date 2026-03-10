# Plan: COMP0082 Protein Enzyme Classification — Maximum Marks

## TL;DR

Build a 7-class protein sequence classifier (EC 0–6) that progresses from a handcrafted-feature baseline through ESM-2 PLM embeddings to an optimised ensemble, with full interpretability, proper CV, and a ready-to-go blind prediction pipeline. Structured in 8 phases — each independently verifiable.

---

## Phase 1: Project Scaffolding & Data Loading

**Goal:** Reproducible project skeleton, clean data pipeline, exploratory statistics.

### Steps

1.1. Create the full directory tree defined in `copilot-instructions.md` (`src/`, `src/features/`, `src/models/`, `outputs/models/`, `outputs/figures/`, `outputs/predictions/`), plus `__init__.py` files.

1.2. Create `requirements.txt` with pinned versions:
  - `biopython`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`, `fair-esm`, `matplotlib`, `seaborn`, `shap`, `joblib`, `scipy`, `imbalanced-learn`

1.3. **`src/data_loading.py`** — implement:
  - `load_all_sequences(root: Path) -> pd.DataFrame` — iterate 7 FASTA files via `Bio.SeqIO`, return DataFrame with columns: `seq_id`, `sequence`, `label` (int 0–6), `length`.
  - File-to-label mapping: `class0_rep_seq.fasta.txt → 0`, `ec{n}_rep_seq.fasta.txt → n`.
  - `get_cv_splits(labels: np.ndarray, n_splits=5, seed=42) -> list[tuple[ndarray, ndarray]]` using `StratifiedKFold`.
  - Print class distribution summary when run standalone.

1.4. **`notebooks/01_eda.ipynb`** (optional but recommended) — sequence length distribution per class, class balance bar chart, amino acid composition heatmap.

### Verification
- Run `python -m src.data_loading` → prints 7 class counts matching the table (32410, 1184, 2769, 2108, 600, 411, 282 = 39764 total).
- Each CV fold preserves class proportions (check with `np.bincount` on fold labels).

### Files Created
- `src/__init__.py`, `src/data_loading.py`
- `src/features/__init__.py`, `src/models/__init__.py`
- `requirements.txt`

---

## Phase 2: Handcrafted Feature Engineering

**Goal:** Four feature groups ready as numpy arrays, each extractable independently.

*Depends on: Phase 1*

### Steps

2.1. **`src/features/composition.py`** — implement:
  - `amino_acid_composition(seq: str) -> np.ndarray` — 20-dim normalised frequency vector (one per canonical AA).
  - `dipeptide_frequencies(seq: str) -> np.ndarray` — 400-dim vector (20×20).
  - `extract_composition_features(sequences: list[str]) -> np.ndarray` — batch wrapper returning `(N, 421)` array (20 AA + 400 dipeptide + 1 length).

2.2. **`src/features/physicochemical.py`** — implement:
  - `physicochemical_features(seq: str) -> np.ndarray` — returns vector of:
    - Molecular weight (`Bio.SeqUtils.molecular_weight`)
    - Isoelectric point (`Bio.SeqUtils.IsoelectricPoint`)
    - Aromaticity (fraction of F+W+Y)
    - GRAVY (grand average of hydropathicity, Kyte-Doolittle)
    - Charge at pH 7 (net charge from pKa tables)
    - Fraction of each secondary-structure-propensity group (helix/sheet/coil based on Chou-Fasman propensities — simple lookup, no external tool)
  - `extract_physicochemical_features(sequences: list[str]) -> np.ndarray` — batch wrapper.

2.3. Build combined handcrafted feature matrix: `X_handcrafted = np.hstack([composition, physicochemical])`. Cache to `outputs/features/handcrafted_features.npy` and labels to `outputs/features/labels.npy`.

### Verification
- `X_handcrafted.shape == (39764, ~430)`, no NaN/Inf values.
- AA composition rows sum to ≈1.0; dipeptide rows sum to ≈1.0.
- Run on a single known protein and manually verify AA counts.

### Files Created
- `src/features/composition.py`
- `src/features/physicochemical.py`

---

## Phase 3: Baseline Models

**Goal:** Establish performance floor with simple classifiers on handcrafted features.

*Depends on: Phase 2*

### Steps

3.1. **`src/training.py`** — implement the core CV training loop:
  - `cross_validate_model(model, X, y, cv_splits, scaler=StandardScaler) -> dict` that:
    - For each fold: fit scaler on train, transform both, fit model, predict + predict_proba on val.
    - Computes per-fold: accuracy, macro F1, balanced accuracy, MCC.
    - Returns mean ± std for all four metrics, plus aggregated predictions for confusion matrix.
  - **CRITICAL:** scaler `.fit()` only on train fold. SMOTE only on train fold (if used).

3.2. **`src/evaluation.py`** — implement:
  - `compute_metrics(y_true, y_pred) -> dict` — accuracy, macro F1, balanced accuracy, MCC.
  - `plot_confusion_matrix(y_true, y_pred, class_names, save_path)` — normalised confusion matrix heatmap.
  - `print_metrics_table(results: dict)` — formatted comparison table.

3.3. **`src/models/baseline.py`** — implement:
  - Logistic Regression with `class_weight='balanced'`, `max_iter=1000`, `multi_class='multinomial'`.
  - Random Forest with `class_weight='balanced'`, `n_estimators=300`.
  - Run both through `cross_validate_model` on handcrafted features.
  - Log all 4 metrics for each. Save confusion matrix plot to `outputs/figures/`.

### Verification
- Both models produce 4 metrics each. Balanced accuracy should be meaningfully above 1/7 ≈ 14.3% (random).
- Confusion matrix saved as PNG. Class 0 should not dominate all predictions.
- No data leakage: check that scaler was fit inside the fold loop.

### Files Created
- `src/training.py`, `src/evaluation.py`, `src/models/baseline.py`

---

## Phase 4: PLM Embedding Extraction

**Goal:** ESM-2 embeddings for all ~40k sequences, cached to disk.

*Parallel with Phase 3 (independent, can run simultaneously if GPU available)*

### Steps

4.1. **`src/features/embeddings.py`** — implement:
  - `extract_esm2_embeddings(sequences: list[str], model_name='esm2_t6_8M_UR50D', batch_size=32, device='cuda') -> np.ndarray`:
    - Load ESM-2 model and alphabet via `esm.pretrained.esm2_t6_8M_UR50D()`.
    - Batch sequences, tokenise, forward pass, mean-pool over sequence dimension (exclude BOS/EOS tokens) → `(N, 320)` array.
    - Show progress bar (`tqdm`).
    - **GPU fallback:** try CUDA → MPS → CPU.
  - Save to `outputs/features/esm2_embeddings.npy`.
  - If file already exists and shape matches, skip re-computation (caching).

4.2. **Model size decision:**
  - `esm2_t6_8M_UR50D`: 8M params, 320-dim, fast — **use this first**.
  - `esm2_t33_650M_UR50D`: 650M params, 1280-dim, much better — **upgrade if time/GPU allows**.
  - Document the choice in code comments.

4.3. Handle long sequences: ESM-2 max length is 1022 tokens. **Truncate sequences longer than 1022** with a warning log. (Most proteins are < 1000 AA based on EDA.)

### Verification
- `esm2_embeddings.shape == (39764, 320)`, no NaN values.
- Embeddings for different classes should show some separation (quick t-SNE/PCA viz in notebook).
- Forward pass produces non-zero, non-constant vectors.

### Files Created
- `src/features/embeddings.py`

---

## Phase 5: Advanced Models & Model Selection

**Goal:** Best-performing model using combined features, with hyperparameter tuning.

*Depends on: Phase 3 + Phase 4*

### Steps

5.1. **`src/models/advanced.py`** — implement:
  - **XGBoost** classifier with:
    - `objective='multi:softprob'`, `num_class=7`, `eval_metric='mlogloss'`
    - `scale_pos_weight` or sample weights derived from class frequencies
    - Features: ESM-2 embeddings only (320-dim) — fast and powerful
    - Hyperparameter search via `RandomizedSearchCV` or manual grid over: `max_depth=[4,6,8]`, `n_estimators=[300,500,800]`, `learning_rate=[0.01,0.05,0.1]`, `subsample=[0.7,0.8,0.9]`
  - **LightGBM** as alternative (often better on imbalanced data with `is_unbalance=True`)
  - **Optional MLP** on ESM-2 embeddings:
    - Architecture: 320 → 256 → 128 → 7, ReLU, dropout 0.3, weighted cross-entropy loss
    - Train with early stopping on validation loss

5.2. **Feature combinations to evaluate** (ablation-ready):
  - A: Handcrafted features only (~430-dim)
  - B: ESM-2 embeddings only (320-dim)
  - C: ESM-2 + handcrafted (750-dim)
  - D: ESM-2 + physicochemical only (326-dim)
  
  Run each through 5-fold CV with XGBoost. Record all 4 metrics. This doubles as the ablation study for interpretability.

5.3. **Final model selection:** pick the config with best **Macro F1** (most relevant for imbalanced multi-class). Retrain on full training data if desired, but keep CV results as the reported numbers.

5.4. Save best model to `outputs/models/best_model.joblib` (or `.pt` for PyTorch).

### Verification
- Metrics table with all combinations × 4 metrics. ESM-2 features should dominate.
- Best Macro F1 should be significantly above baseline (target: >0.70 for ESM-2 models, >0.50 for handcrafted).
- Saved model file exists and loads correctly.

### Files Created
- `src/models/advanced.py`

---

## Phase 6: Interpretability (Extra Credit)

**Goal:** At least two interpretability analyses, with publication-quality figures.

*Depends on: Phase 5*

### Steps

6.1. **`src/interpretability.py`** — implement:
  - `feature_importance_analysis(model, X, y, feature_names, save_dir)`:
    - For tree-based models: extract Gini/gain importance.
    - Permutation importance on validation set as a model-agnostic alternative.
    - Bar chart of top-20 features saved to `outputs/figures/feature_importance.png`.
  - `ablation_study(model_fn, feature_groups: dict[str, np.ndarray], y, cv_splits)`:
    - Train with each feature group removed, record metric drop.
    - Table showing contribution of each feature set.
    - This is **already partially done in step 5.2** — formalise results here.
  - `class_confusion_analysis(y_true, y_pred, y_proba, class_names)`:
    - Per-class precision/recall.
    - Most confused class pairs.
    - Example misclassified sequences with their confidence.

6.2. **Optional (if time allows):** SHAP summary plot for the XGBoost model using `shap.TreeExplainer`. Warning: slow on 40k samples — use a subsample of ~2000.

### Verification
- At least 2 interpretability figures saved in `outputs/figures/`.
- Ablation results show ESM-2 is the dominant feature group.
- Feature importance ranking is plausible (e.g., AA composition features like C, H for enzymes).

### Files Created
- `src/interpretability.py`

---

## Phase 7: Confidence Scoring & Blind Prediction Pipeline

**Goal:** Calibrated confidence levels, ready-to-use blind prediction script.

*Depends on: Phase 5*

### Steps

7.1. **`src/confidence.py`** — implement:
  - `assign_confidence(proba: np.ndarray) -> list[str]`: max probability → High/Medium/Low per thresholds in instructions.
  - `confidence_calibration_report(y_true, y_proba, save_path)`:
    - On CV held-out predictions: for each confidence level, compute accuracy.
    - E.g., "High confidence predictions are correct X% of the time".
    - Reliability diagram (calibration curve) saved to `outputs/figures/`.

7.2. **`src/predict_blind.py`** — implement:
  - `predict_blind(fasta_path: Path, model_path: Path, output_path: Path)`:
    - Load model from disk.
    - Parse FASTA from given path.
    - Extract same features used during training.
    - Predict class + probabilities.
    - Map to confidence levels.
    - Write output in exact format: `SEQ01 1 Confidence High`.
  - CLI interface: `python -m src.predict_blind --fasta <path> --model <path> --output <path>`.
  - **Must work on any FASTA file**, not just training data.

7.3. **Dry run:** run `predict_blind.py` on a held-out fold (pretending it's the blind set) to verify output format.

### Verification
- Output file matches exact format: `<seq_id> <class> Confidence <level>` with no header.
- Confidence distribution makes sense (most high-confidence predictions are correct).
- Script runs end-to-end on an arbitrary FASTA file.

### Files Created
- `src/confidence.py`, `src/predict_blind.py`

---

## Phase 8: Final Evaluation, Figures & Report Support

**Goal:** All figures and tables needed for the report, plus a summary script.

*Depends on: Phases 5, 6, 7*

### Steps

8.1. **Generate all report figures** (save to `outputs/figures/`):
  - `confusion_matrix_best.png` — normalised confusion matrix of best model (from CV)
  - `metrics_comparison.png` — bar chart comparing baseline vs advanced across 4 metrics
  - `class_distribution.png` — bar chart of class sizes
  - `feature_importance.png` — top-20 features (from Phase 6)
  - `ablation_results.png` — metric drop per feature group
  - `confidence_calibration.png` — reliability diagram
  - `sequence_length_distribution.png` — per-class histogram

8.2. **Generate metrics summary table** as markdown/LaTeX for direct inclusion in report — all models × all metrics × mean ± std.

8.3. Write `README.md` with:
  - Setup instructions (`pip install -r requirements.txt`)
  - How to reproduce results (`python -m src.training`)
  - How to generate blind predictions (`python -m src.predict_blind --fasta <path>`)

### Verification
- All 7+ figures exist in `outputs/figures/`.
- Metrics table is consistent with what was logged during training.
- `README.md` instructions work from a clean environment.

### Files Created
- `README.md`, all figure files in `outputs/figures/`

---

## Dependency Graph

```
Phase 1 (scaffolding)
  ├── Phase 2 (handcrafted features)
  │     └── Phase 3 (baseline models)
  │           └─────────┐
  ├── Phase 4 (ESM-2)  │ (parallel with Phase 3)
  │     └─────────┐     │
  │               v     v
  │           Phase 5 (advanced models)
  │             ├── Phase 6 (interpretability)
  │             └── Phase 7 (confidence + blind pipeline)
  │                   │
  │                   v
  └─────────── Phase 8 (figures + report support)
```

Phases 3 and 4 can run in parallel. Phases 6 and 7 can run in parallel. Everything else is sequential.

---

## Key Decisions

1. **Primary model: XGBoost on ESM-2 embeddings** — best accuracy/speed tradeoff for tabular features from PLMs. Easy to explain, fast to train, handles imbalance well.
2. **ESM-2 model size: start with `esm2_t6_8M_UR50D` (8M, 320-dim), upgrade to 650M if GPU budget allows.** The small model is sufficient for strong results and fits in <1GB VRAM.
3. **Imbalance strategy: `class_weight='balanced'` for sklearn models, `scale_pos_weight` for XGBoost, weighted CE loss for PyTorch.** Prefer weighting over SMOTE — simpler, no synthetic samples to worry about.
4. **Report metric for model selection: Macro F1** — penalises poor performance on minority classes, aligns with marking criteria.
5. **Interpretability targets: ablation study (feature groups) + permutation importance + confusion error analysis** — three techniques for maximum extra credit.

## Scope Boundaries

**Included:**
- All 8 phases above
- Handcrafted + PLM features
- Baseline + advanced comparison
- Full interpretability section
- Blind prediction pipeline (ready for when test set is released)

**Excluded:**
- Hierarchical classification (enzyme vs non-enzyme, then EC class) — more complex for marginal gain
- Sequence-level attention visualisation — requires custom model architecture
- Ensemble of multiple PLM models — diminishing returns
- Data augmentation (no biological basis for augmenting protein sequences)
