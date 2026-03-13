# COMP0082 — Protein Enzyme Classification

7-class classification of protein sequences into EC (Enzyme Commission) classes using machine learning.

## Setup

```bash
# Install PyTorch with CUDA support (for NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

## Project Structure

```
src/
├── data_loading.py          # FASTA parsing, label assignment, CV splits
├── features/
│   ├── composition.py       # AA composition, dipeptide frequencies
│   ├── physicochemical.py   # MW, pI, GRAVY, charge, aromaticity
│   └── embeddings.py        # ESM-2 protein language model embeddings
├── models/
│   ├── baseline.py          # Logistic Regression, Random Forest
│   ├── advanced.py          # XGBoost, LightGBM, feature ablation
│   └── finetune.py          # End-to-end ESM-2 8M fine-tuning
├── training.py              # Cross-validation loop (leak-safe)
├── evaluation.py            # Metrics, confusion matrix, comparison plots
├── interpretability.py      # Feature importance, ablation, error analysis
├── confidence.py            # Probability → High/Medium/Low
└── predict_blind.py         # Blind challenge prediction pipeline
```

## Usage

### 1. Verify data loading

```bash
python -m src.data_loading
```

### 2. Run baseline models (handcrafted features)

```bash
python -m src.models.baseline
```

### 3. Extract ESM-2 embeddings (GPU recommended)

```bash
python -m src.features.embeddings
```

### 4. Run advanced models + ablation study

```bash
python -m src.models.advanced
```

### 5. Fine-tune ESM-2 end-to-end (GPU required, ~5 h on RTX 5090)

```bash
python -m src.models.finetune
```

### 6. Run interpretability analysis

```bash
python -m src.interpretability
```

### 7. Run confidence calibration

```bash
python -m src.confidence
```

### 8. Generate blind challenge predictions

```bash
# Using best model (XGBoost + SMOTE on ESM-2 650M + Physicochemical)
python -m src.predict_blind --fasta <path_to_blind_test.fasta> \
    --model outputs/models/best_model.joblib \
    --output outputs/predictions/blind_predictions.txt

# Using fine-tuned ESM-2 8M model
python -m src.predict_blind --fasta <path_to_blind_test.fasta> \
    --model outputs/models/finetune_artifact.joblib \
    --output outputs/predictions/blind_predictions_finetune.txt
```

## Outputs

- `outputs/models/` — Saved model artefacts
- `outputs/figures/` — Confusion matrices, metrics comparison, feature importance
- `outputs/predictions/` — Blind challenge predictions
- `outputs/features/` — Cached feature matrices

## Results Summary

All experiments use **stratified 5-fold cross-validation**. Best model selected by Macro F1.

### Model Comparison

Features for XGBoost / LightGBM: **ESM-2 650M** (`esm2_t33_650M_UR50D`, 1280-d) + physicochemical (8-d) = 1288-d total.

| Model | Features | Accuracy | Macro F1 | Balanced Acc. | MCC |
|-------|----------|----------|----------|---------------|-----|
| Logistic Regression | Handcrafted (429-d) | 0.457 | 0.191 | 0.300 | 0.171 |
| Random Forest | Handcrafted (429-d) | 0.816 | 0.141 | 0.149 | 0.093 |
| XGBoost | ESM-2 650M + Physico (1288-d) | 0.886 | 0.575 | 0.516 | 0.621 |
| LightGBM | ESM-2 650M + Physico (1288-d) | 0.879 | 0.515 | 0.467 | 0.605 |
| LightGBM + SMOTE | ESM-2 650M + Physico (1288-d) | 0.880 | 0.576 | 0.535 | 0.620 |
| ESM-2 8M Fine-tune (end-to-end) | Raw sequences | 0.823 | 0.509 | 0.570 | 0.553 |
| **XGBoost + SMOTE** | **ESM-2 650M + Physico (1288-d)** | **0.886** | **0.595** | **0.552** | **0.631** |

### Feature Ablation Study (XGBoost with balanced sample weights, ESM-2 650M)

| Feature Set | Accuracy | Macro F1 | Balanced Acc. | MCC |
|-------------|----------|----------|---------------|-----|
| Handcrafted only (429-d) | 0.817 | 0.206 | 0.191 | 0.236 |
| ESM-2 + Handcrafted (1709-d) | 0.884 | 0.552 | 0.496 | 0.609 |
| ESM-2 only (1280-d) | 0.884 | 0.562 | 0.506 | 0.615 |
| **ESM-2 + Physicochemical (1288-d)** | **0.886** | **0.575** | **0.516** | **0.621** |

### Key Findings

- **ESM-2 650M embeddings dominate** handcrafted features (Macro F1: 0.56 vs 0.21)
- **SMOTE on training fold** improves XGBoost minority-class recall (+2% Macro F1) at no accuracy cost
- **ESM-2 + Physicochemical** is the strongest feature combination (Macro F1=0.595, MCC=0.631)
- Adding all handcrafted features to ESM-2 **hurts** performance (noise from 429 extra dimensions)
- **End-to-end fine-tuning** (ESM-2 8M) achieves F1=0.509 — outperformed by XGBoost on frozen 650M embeddings, likely due to model size (8M vs 650M parameters)
- **Best model**: XGBoost + SMOTE on ESM-2 650M + Physicochemical features
- Per-class weaknesses: Lyase (F1=0.24), Isomerase (F1=0.29) — smallest minority classes

### Confidence Calibration (best model: XGBoost + SMOTE)

| Level | Count | Accuracy | Mean Max Prob |
|-------|-------|----------|---------------|
| High (p ≥ 0.80) | 30,448 (76.6%) | 95.1% | 0.959 |
| Medium (0.50 ≤ p < 0.80) | 6,658 (16.7%) | 66.2% | 0.663 |
| Low (p < 0.50) | 2,658 (6.7%) | 41.2% | 0.411 |

## Class Distribution

| Class | Label          | Count  |
|-------|----------------|--------|
| 0     | Not an enzyme  | 32,410 |
| 1     | Oxidoreductase | 1,184  |
| 2     | Transferase    | 2,769  |
| 3     | Hydrolase      | 2,108  |
| 4     | Lyase          | 600    |
| 5     | Isomerase      | 411    |
| 6     | Ligase         | 282    |
