# COMP0082 — Protein Enzyme Classification

Seven-class supervised classification of protein sequences into Enzyme Commission (EC)
classes using machine learning and protein language model embeddings.

## Quick Start

```bash
# 1. Install PyTorch (CUDA 12.4 — adjust URL for your hardware)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (steps 1–8 below)
```

## Project Structure

```
protein-classification/
├── src/
│   ├── data_loading.py            # FASTA parsing, label assignment, stratified CV splits
│   ├── features/
│   │   ├── composition.py         # AA composition (20-d) + dipeptide frequencies (400-d)
│   │   ├── physicochemical.py     # MW, pI, GRAVY, charge, aromaticity, SS fractions (8-d)
│   │   └── embeddings.py          # ESM-2 mean-pooled embedding extraction (320/1280-d)
│   ├── models/
│   │   ├── baseline.py            # Logistic Regression, Random Forest
│   │   ├── advanced.py            # XGBoost, LightGBM + SMOTE, ablation study
│   │   ├── finetune.py            # End-to-end ESM-2 8M fine-tuning (PyTorch)
│   │   └── ensemble.py            # Soft-vote ensemble (fine-tuned 8M + XGBoost 650M)
│   ├── training.py                # Leak-safe CV loop with scaler/SMOTE on train fold only
│   ├── evaluation.py              # Metrics, confusion matrices, comparison plots
│   ├── interpretability.py        # Feature importance, SHAP, per-class error analysis
│   ├── confidence.py              # Probability → High / Medium / Low mapping
│   ├── predict_blind.py           # Blind challenge prediction pipeline
│   └── generate_plots.py          # Model comparison and ablation figures
├── outputs/
│   ├── models/                    # Saved model artefacts (.joblib, .pt)
│   ├── figures/                   # 19 publication-quality PNG plots
│   └── predictions/               # Blind challenge output files
├── DEVELOPER_GUIDE.md             # Exhaustive developer documentation
├── REPORT.md                      # 2 500-word academic submission report
├── requirements.txt               # Pinned Python dependencies
└── README.md                      # This file
```

## Pipeline

### Step 1 — Verify data loading

```bash
python -m src.data_loading
```

### Step 2 — Baseline models (handcrafted features only)

```bash
python -m src.models.baseline
```

### Step 3 — Extract ESM-2 embeddings (GPU recommended)

```bash
# 650M model (best quality, ~2.5 h on RTX 3060)
python -m src.features.embeddings --model 650M

# 8M model (fast, ~15 min)
python -m src.features.embeddings --model 8M
```

### Step 4 — Advanced models + ablation study

```bash
python -m src.models.advanced
```

### Step 5 — Fine-tune ESM-2 8M end-to-end (GPU required)

```bash
python -m src.models.finetune
```

### Step 6 — Ensemble evaluation

```bash
python -m src.models.ensemble
```

### Step 7 — Interpretability analysis

```bash
python -m src.interpretability
```

### Step 8 — Confidence calibration

```bash
python -m src.confidence
```

### Blind challenge predictions

```bash
# Recommended: ensemble (fine-tuned 8M + XGBoost 650M)
python -m src.predict_blind --fasta <test.fasta> \
    --model outputs/models/best_model.joblib \
    --model-finetune outputs/models/finetune_artifact.joblib \
    --output outputs/predictions/blind_predictions.txt

# Alternative: best single model only
python -m src.predict_blind --fasta <test.fasta> \
    --model outputs/models/best_model.joblib \
    --output outputs/predictions/blind_predictions.txt
```

Output format (one line per sequence, no header):
```
SEQ01 1 Confidence High
SEQ02 0 Confidence Medium
```

## Results

All results from **stratified 5-fold cross-validation** (seed = 42). Best model chosen by Macro F1.

### Model Comparison

| Model | Features | Accuracy | Macro F1 | Bal. Acc. | MCC |
|-------|----------|----------|----------|-----------|-----|
| Logistic Regression | Handcrafted (429-d) | 0.457 | 0.191 | 0.300 | 0.171 |
| Random Forest | Handcrafted (429-d) | 0.816 | 0.141 | 0.149 | 0.093 |
| XGBoost | ESM-2 650M + Physico (1 288-d) | 0.886 | 0.575 | 0.516 | 0.621 |
| LightGBM + SMOTE | ESM-2 650M + Physico (1 288-d) | 0.880 | 0.576 | 0.535 | 0.620 |
| ESM-2 8M Fine-tune | Raw sequences | 0.823 | 0.509 | 0.570 | 0.553 |
| XGBoost + SMOTE | ESM-2 650M + Physico (1 288-d) | 0.886 | **0.595** | 0.552 | 0.631 |
| **Ensemble** | **Soft vote (8M FT + XGB 650M)** | **0.890** | **0.625** | **0.604** | **0.659** |

### Feature Ablation (XGBoost, balanced weights)

| Feature Set | Accuracy | Macro F1 | Bal. Acc. | MCC |
|-------------|----------|----------|-----------|-----|
| Handcrafted only (429-d) | 0.817 | 0.206 | 0.191 | 0.236 |
| ESM-2 only (1 280-d) | 0.884 | 0.562 | 0.506 | 0.615 |
| ESM-2 + Handcrafted (1 709-d) | 0.884 | 0.552 | 0.496 | 0.609 |
| **ESM-2 + Physicochemical (1 288-d)** | **0.886** | **0.575** | **0.516** | **0.621** |

### Confidence Calibration

| Level | Count | Accuracy | Mean Max Prob |
|-------|-------|----------|---------------|
| High (p ≥ 0.80) | 30 448 (76.6 %) | 95.1 % | 0.959 |
| Medium (0.50 ≤ p < 0.80) | 6 658 (16.7 %) | 66.2 % | 0.663 |
| Low (p < 0.50) | 2 658 (6.7 %) | 41.2 % | 0.411 |

### Class Distribution

| Class | Label | Count |
|-------|-------|-------|
| 0 | Not an enzyme | 32 410 |
| 1 | Oxidoreductase | 1 184 |
| 2 | Transferase | 2 769 |
| 3 | Hydrolase | 2 108 |
| 4 | Lyase | 600 |
| 5 | Isomerase | 411 |
| 6 | Ligase | 282 |

## Key Findings

1. **ESM-2 650M embeddings dominate** handcrafted features (Macro F1: 0.56 vs 0.21).
2. **SMOTE on training folds** boosts minority-class recall (+2 % Macro F1) at no accuracy cost.
3. **ESM-2 + Physicochemical (8-d)** is the strongest feature pairing; adding full handcrafted features introduces noise.
4. **Ensemble** of fine-tuned ESM-2 8M and XGBoost on frozen 650M embeddings achieves the best overall Macro F1 of **0.625**.
5. **Minority classes remain challenging**: Lyase (F1 = 0.24) and Isomerase (F1 = 0.29) suffer from extreme class imbalance.
6. **High-confidence predictions are reliable**: 95.1 % accuracy when p ≥ 0.80.

## Documentation

| Document | Purpose |
|----------|---------|
| [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) | Exhaustive developer onboarding, architecture guide, and extension reference |
| [`REPORT.md`](REPORT.md) | 2 500-word academic submission report for COMP0082 |
