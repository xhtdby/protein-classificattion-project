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
│   └── advanced.py          # XGBoost, LightGBM, feature ablation
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

### 3. Extract ESM-2 embeddings (GPU recommended, ~15 min on RTX 3060)

```bash
python -m src.features.embeddings
```

### 4. Run advanced models + ablation study (XGBoost uses CUDA)

```bash
python -m src.models.advanced
```

### 5. Run interpretability analysis

```bash
python -m src.interpretability
```

### 6. Run confidence calibration

```bash
python -m src.confidence
```

### 7. Generate blind challenge predictions

```bash
python -m src.predict_blind --fasta <path_to_blind_test.fasta> --model outputs/models/best_model.joblib --output outputs/predictions/blind_predictions.txt
```

## Outputs

- `outputs/models/` — Saved model artefacts
- `outputs/figures/` — Confusion matrices, metrics comparison, feature importance
- `outputs/predictions/` — Blind challenge predictions
- `outputs/features/` — Cached feature matrices

## Results Summary

All experiments use **stratified 5-fold cross-validation**. Best model selected by Macro F1.

### Model Comparison

| Model | Features | Accuracy | Macro F1 | Balanced Acc. | MCC |
|-------|----------|----------|----------|---------------|-----|
| Logistic Regression | Handcrafted (429-d) | 0.457 | 0.191 | 0.300 | 0.171 |
| Random Forest | Handcrafted (429-d) | 0.815 | 0.128 | 0.143 | -0.003 |
| XGBoost | ESM-2 (320-d) | 0.867 | 0.466 | 0.394 | 0.522 |
| LightGBM | ESM-2 (320-d) | 0.867 | 0.516 | 0.481 | 0.576 |
| XGBoost + SMOTE | ESM-2 (320-d) | 0.849 | 0.527 | 0.541 | 0.569 |
| **LightGBM + SMOTE** | **ESM-2 (320-d)** | **0.854** | **0.528** | **0.526** | **0.569** |

### Feature Ablation Study (XGBoost baseline)

| Feature Set | Accuracy | Macro F1 | Balanced Acc. | MCC |
|-------------|----------|----------|---------------|-----|
| ESM-2 only (320-d) | 0.867 | 0.466 | 0.394 | 0.522 |
| Handcrafted only (429-d) | 0.818 | 0.168 | 0.164 | 0.163 |
| ESM-2 + Handcrafted (749-d) | 0.863 | 0.431 | 0.361 | 0.498 |
| ESM-2 + Physicochemical (328-d) | 0.868 | 0.472 | 0.399 | 0.528 |

### Key Findings

- **ESM-2 embeddings dominate** handcrafted features (Macro F1: 0.47 vs 0.17)
- **SMOTE** significantly improves balanced accuracy (+0.13) at minimal accuracy cost
- Adding handcrafted features to ESM-2 **hurts** performance (noise dilution)
- **Best model**: LightGBM + SMOTE with `class_weight='balanced'` on ESM-2 features
- Per-class weaknesses: Lyase (F1=0.27), Isomerase (F1=0.30) — smallest classes

### Confidence Calibration

| Level | Count | Accuracy | Mean Max Prob |
|-------|-------|----------|---------------|
| High (p ≥ 0.80) | 30,842 (77.6%) | 94.0% | 0.966 |
| Medium (0.50 ≤ p < 0.80) | 6,504 (16.4%) | 61.6% | 0.658 |
| Low (p < 0.50) | 2,418 (6.1%) | 39.3% | 0.415 |

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
