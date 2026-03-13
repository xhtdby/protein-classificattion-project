# Developer Guide — Protein Enzyme Classification Project

> **Audience:** New developers joining or extending this project.
> **Last updated:** March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Description](#3-dataset-description)
4. [Architecture Overview](#4-architecture-overview)
5. [Module Reference](#5-module-reference)
6. [Data Pipeline](#6-data-pipeline)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Training](#8-model-training)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Interpretability](#10-interpretability)
11. [Confidence Scoring](#11-confidence-scoring)
12. [Blind Prediction Pipeline](#12-blind-prediction-pipeline)
13. [Results Summary](#13-results-summary)
14. [Data-Leakage Prevention](#14-data-leakage-prevention)
15. [Reproducibility](#15-reproducibility)
16. [Extension Guide](#16-extension-guide)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Project Overview

This project implements a **7-class supervised classification** system that predicts whether a
protein is an enzyme and, if so, its top-level Enzyme Commission (EC) class. The classes are:

| Class | Label | Count | Proportion |
|-------|-------|-------|------------|
| 0 | Not an enzyme | 32 410 | 81.5 % |
| 1 | Oxidoreductase | 1 184 | 3.0 % |
| 2 | Transferase | 2 769 | 7.0 % |
| 3 | Hydrolase | 2 108 | 5.3 % |
| 4 | Lyase | 600 | 1.5 % |
| 5 | Isomerase | 411 | 1.0 % |
| 6 | Ligase | 282 | 0.7 % |
| **Total** | | **39 764** | **100 %** |

The dataset is **heavily class-imbalanced** (class 0 ≈ 82 %). Every design decision —
loss weighting, oversampling, metric selection — accounts for this.

### Technology Stack

- **Language:** Python 3.10+
- **ML frameworks:** scikit-learn, XGBoost, LightGBM, PyTorch
- **Protein language model:** ESM-2 (Meta AI) via the `fair-esm` library
- **Bioinformatics:** BioPython (`Bio.SeqIO`, `Bio.SeqUtils.ProtParam`)
- **Visualisation:** matplotlib, seaborn
- **Interpretability:** SHAP

---

## 2. Environment Setup

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (recommended for ESM-2 extraction and fine-tuning)
- ~12 GB disk space for embeddings and model checkpoints

### Installation

```bash
# 1. Clone the repository
git clone <repo-url> && cd protein-classification

# 2. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install PyTorch with CUDA support (adjust URL for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### requirements.txt

```
biopython>=1.83
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.2
torch>=2.1
fair-esm>=2.0
matplotlib>=3.8
seaborn>=0.13
shap>=0.44
joblib>=1.3
scipy>=1.12
imbalanced-learn>=0.12
tqdm>=4.66
```

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| Handcrafted features + baselines | CPU only | Any modern CPU |
| ESM-2 8M embedding extraction | 4 GB VRAM | 8 GB VRAM |
| ESM-2 650M embedding extraction | 8 GB VRAM | 16 GB VRAM |
| Fine-tuning ESM-2 8M | 6 GB VRAM | 12 GB VRAM |

---

## 3. Dataset Description

Seven FASTA files in the workspace root:

| File | Class |
|------|-------|
| `class0_rep_seq.fasta.txt` | 0 (Not an enzyme) |
| `ec1_rep_seq.fasta.txt` | 1 (Oxidoreductase) |
| `ec2_rep_seq.fasta.txt` | 2 (Transferase) |
| `ec3_rep_seq.fasta.txt` | 3 (Hydrolase) |
| `ec4_rep_seq.fasta.txt` | 4 (Lyase) |
| `ec5_rep_seq.fasta.txt` | 5 (Isomerase) |
| `ec6_rep_seq.fasta.txt` | 6 (Ligase) |

Each file contains **representative sequences** — non-homologous after clustering, so no
additional deduplication is necessary.

### Sequence Statistics

- **Total sequences:** 39 764
- **Median length:** ~300 amino acids
- **Max length:** > 1 000 amino acids (a few sequences exceed ESM-2's 1 022-token limit)
- **Alphabet:** Standard 20 canonical amino acids plus occasional non-standard residues (X, U, B, Z, O) which are handled gracefully by all feature extractors.

---

## 4. Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                  │
│  data_loading.py: FASTA parsing → DataFrame → stratified CV splits │
└───────────┬────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Feature Layer                                  │
│  composition.py      → 421-d (AA freq + dipeptide + length)        │
│  physicochemical.py  → 8-d  (MW, pI, GRAVY, charge, SS fracs)     │
│  embeddings.py       → 320-d / 1280-d (ESM-2 mean-pooled)         │
│  Combined            → 1288-d (ESM-2 650M + physicochemical)       │
└───────────┬────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Training Layer                                  │
│  training.py: leak-safe CV loop (scaler + SMOTE on train only)     │
│  baseline.py:  Logistic Regression, Random Forest                  │
│  advanced.py:  XGBoost, LightGBM, ablation, threshold optimisation │
│  finetune.py:  ESM-2 8M end-to-end fine-tuning (PyTorch)          │
│  ensemble.py:  Soft-vote fusion of fine-tuned 8M + XGBoost 650M   │
└───────────┬────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────────┐
│                   Evaluation & Output Layer                         │
│  evaluation.py:       metrics, confusion matrices, comparison plots │
│  interpretability.py: feature importance, SHAP, error analysis      │
│  confidence.py:       probability → High / Medium / Low calibration │
│  predict_blind.py:    FASTA → model → formatted predictions         │
│  generate_plots.py:   combined model comparison figures              │
└────────────────────────────────────────────────────────────────────┘
```

### Execution Order

```
Phase 1: python -m src.data_loading           # verify data
Phase 2: python -m src.models.baseline        # handcrafted baselines
Phase 3: python -m src.features.embeddings    # ESM-2 extraction
Phase 4: python -m src.models.advanced        # XGBoost/LightGBM + ablation
Phase 5: python -m src.models.finetune        # end-to-end fine-tuning
Phase 6: python -m src.models.ensemble        # ensemble evaluation
Phase 7: python -m src.interpretability       # feature importance & SHAP
Phase 8: python -m src.confidence             # confidence calibration
```

Phases 2 and 3 are independent and can run in parallel. All other phases are sequential.

---

## 5. Module Reference

### `src/data_loading.py`

**Purpose:** Parse all 7 FASTA files and generate stratified cross-validation splits.

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_all_sequences` | `(root: Path) → pd.DataFrame` | Returns DataFrame with columns `seq_id`, `sequence`, `label`, `length` |
| `get_cv_splits` | `(labels, n_splits=5, seed=42) → list[tuple]` | Stratified K-fold train/val index pairs |
| `print_class_distribution` | `(df) → None` | Formatted class count table |

**Key constants:**
- `SEED = 42`
- `FASTA_LABEL_MAP` — maps each filename to its integer label
- `CLASS_NAMES` — `["Not enzyme", "Oxidoreductase", "Transferase", "Hydrolase", "Lyase", "Isomerase", "Ligase"]`

---

### `src/features/composition.py`

**Purpose:** Amino acid composition and dipeptide frequency features.

| Function | Output Shape | Description |
|----------|-------------|-------------|
| `amino_acid_composition(seq)` | `(20,)` | Normalised frequency of each canonical AA |
| `dipeptide_frequencies(seq)` | `(400,)` | Normalised frequency of each AA pair |
| `extract_composition_features(sequences)` | `(N, 421)` | AA (20) + dipeptide (400) + length (1) |
| `get_feature_names()` | `list[str]` | Ordered feature names for the 421-d vector |

---

### `src/features/physicochemical.py`

**Purpose:** Physicochemical property features using `Bio.SeqUtils.ProtParam`.

| Feature | Method |
|---------|--------|
| Molecular weight | `ProtParam.molecular_weight()` |
| Isoelectric point (pI) | `ProtParam.isoelectric_point()` |
| Aromaticity | Fraction of F + W + Y |
| GRAVY | Grand average of hydropathicity (Kyte–Doolittle) |
| Charge at pH 7 | `ProtParam.charge_at_pH(7.0)` |
| Helix / Turn / Sheet fractions | Chou–Fasman secondary structure propensity |

Output: `(N, 8)` array. Feature names available via `get_feature_names()`.

---

### `src/features/embeddings.py`

**Purpose:** Extract mean-pooled embeddings from ESM-2 protein language models.

| Model | Parameter Count | Embedding Dim | Recommended Batch Size |
|-------|----------------|---------------|----------------------|
| `esm2_t6_8M_UR50D` | 8 M | 320 | 64 (GPU) / 8 (CPU) |
| `esm2_t33_650M_UR50D` | 650 M | 1 280 | 8 (GPU) / 1 (CPU) |

**Key functions:**
- `extract_esm2_embeddings(sequences, model_name, batch_size, device)` — raw extraction
- `load_or_compute_embeddings(sequences, cache_path, model_name)` — with disk caching

**Behaviour:**
1. Sequences longer than 1 022 tokens are truncated with a warning.
2. BOS/EOS tokens are excluded from mean-pooling.
3. Device fallback: CUDA → MPS → CPU.
4. Embeddings are cached as `.npy` files in `outputs/features/`.

---

### `src/training.py`

**Purpose:** Cross-validation loop with strict data-leakage prevention.

```python
cross_validate_model(
    model_fn,          # callable returning a fresh model instance
    X, y,              # features and labels
    cv_splits=None,    # pre-computed fold indices (or generated internally)
    n_splits=5,
    use_scaler=True,   # fit StandardScaler on train fold only
    use_smote=False,   # apply SMOTE on train fold only
    use_class_weight=False,
    model_name="model"
) -> dict
```

**Returns:**
```python
{
    "model_name": str,
    "fold_metrics": [{"accuracy": ..., "macro_f1": ..., ...}, ...],
    "summary": {"accuracy_mean": ..., "accuracy_std": ..., ...},
    "oof_preds": np.ndarray,   # out-of-fold predictions
    "oof_proba": np.ndarray,   # out-of-fold probability vectors
    "oof_true": np.ndarray     # ground truth labels
}
```

**Also provides:**
- `optimize_thresholds(y_true, y_proba)` — per-class threshold tuning via `scipy.optimize.minimize`

---

### `src/models/baseline.py`

**Purpose:** Establish performance floor with simple classifiers on handcrafted features.

| Model | Configuration |
|-------|---------------|
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000`, `solver='lbfgs'` |
| Random Forest | `n_estimators=300`, `max_depth=20`, `class_weight='balanced'` |
| Random Forest + PCA(50) | Pipeline: `PCA(50) → RandomForest` |

All models are evaluated via the leak-safe `cross_validate_model` loop.

---

### `src/models/advanced.py`

**Purpose:** Train gradient-boosted models on ESM-2 embeddings with ablation study.

**Models trained:**
1. XGBoost (ESM-2 650M + Physicochemical)
2. XGBoost with SMOTE
3. LightGBM
4. LightGBM with SMOTE
5. Tuned XGBoost (optional random search)

**Ablation study** evaluates four feature combinations:
- Handcrafted only (429-d)
- ESM-2 only (1 280-d)
- ESM-2 + all Handcrafted (1 709-d)
- ESM-2 + Physicochemical (1 288-d) ← winner

**Output artefact:** `outputs/models/best_model.joblib` containing:
```python
{
    "model": trained_xgboost,
    "scaler": fitted_StandardScaler,
    "feature_source": "ESM-2 + Physicochemical",
    "esm_model": "esm2_t33_650M_UR50D",
    "thresholds": np.ndarray,
    "oof_preds": np.ndarray,
    "oof_proba": np.ndarray,
    "oof_true": np.ndarray
}
```

---

### `src/models/finetune.py`

**Purpose:** End-to-end fine-tuning of ESM-2 8M for 7-class classification.

**Architecture:**
```
ESM-2 8M (6 transformer layers, fully unfrozen)
    → Mean-pool over sequence tokens
    → Linear(320, 128)
    → GELU
    → Dropout(0.3)
    → Linear(128, 7)
```

**Training details:**
- AdamW optimiser with differential learning rates: backbone = 2 × 10⁻⁵, head = 1 × 10⁻⁴
- OneCycleLR scheduler
- bfloat16 mixed precision (if supported)
- Per-fold class weights in the cross-entropy loss
- Early stopping on validation Macro F1 (patience = 5)
- Gradient accumulation (effective batch size = `batch_size × grad_accum`)
- Max sequence length = 512 (trades accuracy for VRAM)

**Output artefact:** `outputs/models/finetune_artifact.joblib` containing a `FinetunePredictor`
instance that wraps the PyTorch model for sklearn-style `predict_proba`.

---

### `src/models/ensemble.py`

**Purpose:** Soft-vote ensemble combining fine-tuned ESM-2 8M and XGBoost on frozen 650M
embeddings.

**How it works:**
1. Collect out-of-fold (OOF) probability vectors from both models.
2. Compute ensemble weights proportional to each model's OOF Macro F1.
3. Blend: `p_ensemble = w_ft × p_finetune + w_xgb × p_xgboost`.
4. Apply per-class threshold optimisation to the blended probabilities.

**Final weights:** w_finetune ≈ 0.46, w_xgboost ≈ 0.54 (data-driven, not hand-tuned).

**Class `EnsemblePredictor`** provides a unified `predict_proba(sequences)` interface
for the blind prediction pipeline.

---

### `src/evaluation.py`

**Purpose:** Compute metrics, plot confusion matrices, and persist results.

| Function | Description |
|----------|-------------|
| `compute_metrics(y_true, y_pred)` | Returns `{accuracy, macro_f1, balanced_accuracy, mcc}` |
| `plot_confusion_matrix(...)` | Row-normalised heatmap, saved to PNG |
| `print_metrics_table(results_list)` | Formatted comparison table |
| `plot_metrics_comparison(...)` | Grouped bar chart of models × metrics |
| `plot_class_distribution(...)` | Class frequency bar chart (log scale) |
| `plot_sequence_length_distribution(...)` | Per-class histograms |
| `save_results_json(data, path)` | JSON with numpy type conversion |

---

### `src/interpretability.py`

**Purpose:** Feature importance, SHAP analysis, and per-class error breakdown.

**Analyses implemented (4 techniques for extra credit):**

1. **Feature importance** — Gini importance for tree-based models; permutation importance
   as a model-agnostic alternative. Top-20 bar chart.

2. **SHAP values** — `TreeExplainer` for XGBoost on a random subsample of 500 sequences.
   Summary plot of mean |SHAP| per feature.

3. **Per-class precision / recall / F1** — bar chart with per-class breakdown.

4. **High-confidence error analysis** — breakdown of misclassifications where p ≥ 0.80,
   grouped by true and predicted class.

---

### `src/confidence.py`

**Purpose:** Map predicted probabilities to confidence levels.

| Level | Condition | Challenge Score |
|-------|-----------|-----------------|
| High | max(p) ≥ 0.80 | ±1 |
| Medium | 0.50 ≤ max(p) < 0.80 | ±0.5 |
| Low | max(p) < 0.50 | 0 |

The `confidence_calibration_report` function generates:
- Per-level accuracy statistics
- A **reliability diagram** (calibration curve) saved to `outputs/figures/reliability_diagram.png`

---

### `src/predict_blind.py`

**Purpose:** End-to-end prediction pipeline for the blind challenge test set.

```bash
# Ensemble (recommended)
python -m src.predict_blind \
    --fasta <test.fasta> \
    --model outputs/models/best_model.joblib \
    --model-finetune outputs/models/finetune_artifact.joblib \
    --output outputs/predictions/blind_predictions.txt
```

**Output format** (one line per sequence, no header):
```
SEQ01 1 Confidence High
SEQ02 0 Confidence Medium
SEQ03 3 Confidence Low
```

**Pipeline steps:**
1. Parse input FASTA with `Bio.SeqIO`.
2. Load model artefact(s) from disk.
3. Extract features (ESM-2 embeddings + physicochemical, or raw sequences for fine-tuned model).
4. Scale features using the saved scaler.
5. Predict class probabilities.
6. Apply per-class thresholds (if available).
7. Map max probability to confidence level.
8. Write formatted output.

---

### `src/generate_plots.py`

**Purpose:** Generate combined model-comparison and ablation-study figures from all JSON result files.

Reads `baseline_results.json`, `advanced_results.json`, `finetune_results.json`, and
`ensemble_results.json` to produce:
- `outputs/figures/model_comparison.png`
- `outputs/figures/ablation_results.png`

---

## 6. Data Pipeline

```python
from pathlib import Path
from src.data_loading import load_all_sequences, get_cv_splits

# Load all 39 764 sequences
df = load_all_sequences(Path("."))
sequences = df["sequence"].tolist()
labels = df["label"].values

# Generate stratified 5-fold CV splits (seed = 42)
cv_splits = get_cv_splits(labels, n_splits=5, seed=42)

# Each fold preserves the original class proportions
for train_idx, val_idx in cv_splits:
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
```

---

## 7. Feature Engineering

### 7.1 Handcrafted Features (429-d)

**Amino acid composition (20-d):** Normalised frequency of each of the 20 canonical amino acids
(A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y). Each row sums to ~1.0.

**Dipeptide frequencies (400-d):** Normalised frequency of each ordered pair of canonical amino
acids (AA, AC, …, YY). Captures local sequential patterns.

**Sequence length (1-d):** Raw integer count of amino acids.

**Physicochemical properties (8-d):**
- Molecular weight (Daltons)
- Isoelectric point (pI)
- Aromaticity (fraction of Phe + Trp + Tyr)
- GRAVY (Kyte–Doolittle grand average of hydropathicity)
- Charge at pH 7.0
- Helix fraction (Chou–Fasman)
- Turn fraction (Chou–Fasman)
- Sheet fraction (Chou–Fasman)

### 7.2 ESM-2 Embeddings

**ESM-2** (Evolutionary Scale Modelling) is a protein language model pre-trained on millions of
protein sequences. We use it as a **feature extractor** — no label information is leaked:

1. Tokenise the amino acid sequence.
2. Forward pass through the transformer.
3. Mean-pool over the sequence dimension (excluding BOS/EOS special tokens).
4. Output: a dense vector per sequence (320-d for 8M model, 1 280-d for 650M model).

Embeddings are **stateless** (no label info) and therefore safe to pre-compute once before
cross-validation. However, **normalisation** (StandardScaler) is still fit on the training fold
only.

### 7.3 Feature Combination Results

| Feature Set | Dimensions | Macro F1 |
|-------------|-----------|----------|
| Handcrafted only | 429 | 0.206 |
| ESM-2 650M only | 1 280 | 0.562 |
| ESM-2 + all Handcrafted | 1 709 | 0.552 |
| **ESM-2 + Physicochemical** | **1 288** | **0.575** |

Adding the full 429-d handcrafted feature vector to ESM-2 **hurts** performance — the
400 dipeptide dimensions introduce noise that XGBoost cannot ignore. The 8-d physicochemical
features, however, provide complementary signal (especially molecular weight).

---

## 8. Model Training

### 8.1 Cross-Validation Protocol

All experiments use **stratified 5-fold cross-validation** with `seed = 42`. The stratification
ensures each fold preserves the original class proportions.

### 8.2 Class Imbalance Strategies

| Strategy | Where Used |
|----------|-----------|
| `class_weight='balanced'` | Logistic Regression, Random Forest |
| `scale_pos_weight` / sample weights | XGBoost |
| SMOTE (on training fold only) | XGBoost + SMOTE, LightGBM + SMOTE |
| Weighted cross-entropy loss | Fine-tuned ESM-2 8M |

### 8.3 Metrics

Every experiment reports **four metrics**:

| Metric | Why |
|--------|-----|
| **Accuracy** | Overall correctness (biased by class 0) |
| **Macro F1** | Unweighted average F1 across classes — **primary selection metric** |
| **Balanced Accuracy** | Average recall per class |
| **MCC** | Matthews Correlation Coefficient — robust to imbalance |

### 8.4 Model Progression

```
Logistic Regression (handcrafted)     → Macro F1: 0.191  (floor)
Random Forest (handcrafted)           → Macro F1: 0.141
XGBoost (ESM-2 650M + physico)        → Macro F1: 0.575  (+200 %)
XGBoost + SMOTE                       → Macro F1: 0.595  (+3.5 %)
Fine-tuned ESM-2 8M                   → Macro F1: 0.509
Ensemble (8M FT + XGBoost 650M)       → Macro F1: 0.625  (+5 %)
```

---

## 9. Evaluation Framework

### 9.1 Metrics Computation

```python
from src.evaluation import compute_metrics

metrics = compute_metrics(y_true, y_pred)
# {'accuracy': 0.886, 'macro_f1': 0.595, 'balanced_accuracy': 0.552, 'mcc': 0.631}
```

### 9.2 Confusion Matrices

All confusion matrices are **row-normalised** (each row sums to 1.0), showing the proportion
of true-class samples predicted as each class. Saved as PNG heatmaps.

### 9.3 Result Persistence

All experiment results are saved as JSON files in `outputs/`:
- `baseline_results.json`
- `advanced_results.json`
- `finetune_results.json`
- `ensemble_results.json`
- `confidence_results.json`
- `interpretability_results.json`

---

## 10. Interpretability

### 10.1 Feature Importance

XGBoost Gini (gain-based) importance shows that **ESM-2 embedding dimensions dominate**
the top-20 features. The most important single handcrafted feature is `molecular_weight`
(rank 5). This confirms that ESM-2 captures rich protein representations that subsume most
handcrafted biochemical features.

### 10.2 SHAP Analysis

SHAP values (TreeExplainer, 500-sample subsample) provide per-instance feature attributions.
The summary plot confirms:
- ESM-2 dimensions 112, 308, 216 are the most globally important.
- Physicochemical features contribute modestly but consistently.

### 10.3 Per-Class Error Analysis

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Not enzyme | 0.934 | 0.938 | 0.936 |
| Oxidoreductase | 0.586 | 0.629 | 0.607 |
| Transferase | 0.579 | 0.653 | 0.614 |
| Hydrolase | 0.554 | 0.565 | 0.559 |
| Lyase | 0.329 | 0.192 | 0.242 |
| Isomerase | 0.554 | 0.200 | 0.293 |
| Ligase | 0.603 | 0.426 | 0.499 |

Lyase and Isomerase have the lowest recall because they are the smallest classes
(600 and 411 samples respectively). The model favours predicting the majority class in
ambiguous cases.

### 10.4 High-Confidence Error Analysis

Of 5 312 total errors, 1 499 (28.2 %) are high-confidence errors (p ≥ 0.80). These are
disproportionately:
- **True class:** Not enzyme (551), Hydrolase (320), Transferase (311)
- **Predicted class:** Not enzyme (785) — the model's strongest attractor

---

## 11. Confidence Scoring

```python
from src.confidence import assign_confidence

confidences = assign_confidence(proba)  # proba shape: (N, 7)
# Returns: ["High", "Medium", "Low", ...]
```

### Calibration Results

| Level | Count | Accuracy | Mean Max Prob |
|-------|-------|----------|---------------|
| High (p ≥ 0.80) | 30 448 (76.6 %) | 95.1 % | 0.959 |
| Medium (0.50 ≤ p < 0.80) | 6 658 (16.7 %) | 66.2 % | 0.663 |
| Low (p < 0.50) | 2 658 (6.7 %) | 41.2 % | 0.411 |

The model is **well-calibrated**: high-confidence predictions are genuinely reliable.

---

## 12. Blind Prediction Pipeline

`src/predict_blind.py` provides a complete pipeline that:

1. Accepts any FASTA file (not just the training data).
2. Detects whether the model artefact is a traditional ML model or a fine-tuned neural network.
3. Supports ensemble mode when both `--model` and `--model-finetune` are provided.
4. Applies the same preprocessing (feature extraction + scaling) used during training.
5. Applies per-class thresholds if they were optimised during training.
6. Writes output in the exact required format.

### Usage Examples

```bash
# Single model
python -m src.predict_blind \
    --fasta blind_test.fasta \
    --model outputs/models/best_model.joblib

# Ensemble (recommended)
python -m src.predict_blind \
    --fasta blind_test.fasta \
    --model outputs/models/best_model.joblib \
    --model-finetune outputs/models/finetune_artifact.joblib

# Fine-tuned model only
python -m src.predict_blind \
    --fasta blind_test.fasta \
    --model outputs/models/finetune_artifact.joblib
```

---

## 13. Results Summary

### Final Model Comparison

| Model | Accuracy | Macro F1 | Bal. Acc. | MCC |
|-------|----------|----------|-----------|-----|
| Logistic Regression | 0.457 ± 0.003 | 0.191 ± 0.002 | 0.300 ± 0.010 | 0.171 ± 0.003 |
| Random Forest | 0.816 ± 0.001 | 0.141 ± 0.002 | 0.149 ± 0.001 | 0.093 ± 0.011 |
| Random Forest + PCA(50) | 0.816 ± 0.002 | 0.197 ± 0.002 | 0.185 ± 0.001 | 0.216 ± 0.006 |
| XGBoost (ESM-2 650M + Physico) | 0.886 ± 0.003 | 0.575 ± 0.013 | 0.516 ± 0.011 | 0.621 ± 0.010 |
| LightGBM (ESM-2 650M + Physico) | 0.879 ± 0.003 | 0.515 ± 0.013 | 0.467 ± 0.008 | 0.605 ± 0.010 |
| LightGBM + SMOTE | 0.880 ± 0.002 | 0.576 ± 0.014 | 0.535 ± 0.010 | 0.620 ± 0.006 |
| XGBoost + SMOTE | 0.886 ± 0.002 | 0.595 ± 0.014 | 0.552 ± 0.011 | 0.631 ± 0.006 |
| ESM-2 8M Fine-tune | 0.823 ± 0.008 | 0.509 ± 0.013 | 0.570 ± 0.015 | 0.553 ± 0.008 |
| **Ensemble** | **0.890** | **0.625** | **0.604** | **0.659** |

### Generated Figures (19 total)

| Figure | File |
|--------|------|
| Class distribution | `class_distribution.png` |
| Sequence length distribution | `sequence_length_distribution.png` |
| Confusion matrix (per model) | `cm_*.png` (10 files) |
| Model comparison | `model_comparison.png` |
| Ablation results | `ablation_results.png` |
| Feature importance | `feature_importance.png` |
| SHAP importance | `shap_importance.png` |
| Per-class metrics | `per_class_metrics.png` |
| Reliability diagram | `reliability_diagram.png` |
| High-confidence errors | `high_confidence_errors.png` |

---

## 14. Data-Leakage Prevention

These rules are enforced throughout the codebase:

1. **Stratified K-Fold first.** Fold indices are created before any feature computation or
   normalisation.

2. **Scalers fit on training fold only.** `StandardScaler.fit()` is called only on `X[train_idx]`.
   The validation fold is transformed with `.transform()` only.

3. **SMOTE on training fold only.** Synthetic minority samples are generated only from the
   training fold. The validation fold remains untouched.

4. **No target leakage.** All features derive solely from the amino acid sequence — never from
   the label.

5. **ESM-2 embeddings are stateless.** The forward pass through a pre-trained model uses no
   label information, so embeddings are safe to pre-compute once. But normalisation still
   follows rule 2.

---

## 15. Reproducibility

### Random Seeds

All random number generators are seeded with `42`:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

### Deterministic Behaviour

- `StratifiedKFold(shuffle=True, random_state=42)` ensures identical folds.
- XGBoost and LightGBM use `random_state=42`.
- PyTorch fine-tuning seeds are set per fold.

### Result Persistence

All experiment results are saved as JSON files with full metric dictionaries,
enabling exact reproduction of figures and tables without retraining.

---

## 16. Extension Guide

### Adding a New Feature Extractor

1. Create `src/features/new_feature.py` following the pattern of `composition.py`:
   - Implement `extract_new_features(sequences: list[str]) -> np.ndarray`.
   - Implement `get_feature_names() -> list[str]`.
2. Add the new features to the ablation study in `src/models/advanced.py`.
3. Update `src/predict_blind.py` to handle the new feature source.

### Adding a New Model

1. Create a factory function in `src/models/` (e.g., `make_catboost()`).
2. Call it through `cross_validate_model` in `src/training.py`.
3. Add the results to the comparison table in `src/evaluation.py`.

### Upgrading the ESM-2 Model

To use `esm2_t33_650M_UR50D` for fine-tuning (instead of the 8M model):

1. Update `ESM_MODEL_NAME` and `EMB_DIM` in `src/models/finetune.py`.
2. Reduce `MAX_LEN` and `batch_size` to fit in VRAM (24 GB+ recommended).
3. Consider freezing early transformer layers to reduce memory.

### Potential Improvements

| Improvement | Expected Impact | Difficulty |
|-------------|----------------|------------|
| Fine-tune 650M model | +5–10 % Macro F1 | High (24 GB+ VRAM) |
| Focal loss in fine-tuning | +2–5 % on minority classes | Medium |
| Hierarchical classification | +1–3 % Macro F1 | Medium |
| Per-class threshold tuning on ensemble | +0.5–1 % | Low |
| Sequence-level attention visualisation | Interpretability only | High |

---

## 17. Troubleshooting

### CUDA Out of Memory

- Reduce `--batch-size` in embedding extraction or fine-tuning.
- Use `--max-len 256` in fine-tuning to reduce VRAM.
- Use `--grad-accum 8` to maintain effective batch size with smaller GPU batches.

### ESM-2 Model Download Fails

The `fair-esm` library downloads model weights on first use. If behind a firewall:
1. Download the checkpoint manually from https://github.com/facebookresearch/esm.
2. Place it in the `torch.hub` cache directory.

### SMOTE Fails on Small Classes

SMOTE requires at least `k_neighbors + 1` samples per class. With default `k=5`, classes
with ≤ 5 samples will fail. Our smallest class (Ligase, n = 282) is well above this threshold,
but if you subsample the data, reduce `k_neighbors` accordingly.

### Mismatched Feature Dimensions in predict_blind

The saved model expects the same feature dimensionality as during training. If you change
the ESM-2 model (8M ↔ 650M) or feature combination, retrain and save a new artefact.

---

## Appendix: Output Files Reference

### JSON Result Files

| File | Content |
|------|---------|
| `baseline_results.json` | 3 baseline models: per-fold and summary metrics |
| `advanced_results.json` | 5 advanced models + 4 ablation configs + best model params |
| `finetune_results.json` | 5-fold fine-tuning metrics + configuration |
| `ensemble_results.json` | Ensemble weights, thresholds, blended metrics |
| `confidence_results.json` | Per-level accuracy and mean probability |
| `interpretability_results.json` | Top features, per-class metrics, high-confidence errors |

### Model Artefacts

| File | Content | Size |
|------|---------|------|
| `best_model.joblib` | XGBoost + scaler + thresholds + OOF data | ~50 MB |
| `finetune_artifact.joblib` | FinetunePredictor (lazy-loads PyTorch model) | ~1 MB |
| `finetune_final.pt` | PyTorch state dict (full-dataset retrained) | ~35 MB |
| `finetune_fold{1-5}.pt` | Per-fold best checkpoints | ~35 MB each |
