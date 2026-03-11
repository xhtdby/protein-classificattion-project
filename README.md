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
| **XGBoost** | **ESM-2 (320-d)** | **0.865** | **0.531** | **0.508** | **0.581** |
| LightGBM | ESM-2 (320-d) | 0.867 | 0.515 | 0.480 | 0.575 |
| XGBoost + SMOTE | ESM-2 (320-d) | 0.849 | 0.527 | 0.541 | 0.569 |
| LightGBM + SMOTE | ESM-2 (320-d) | 0.853 | 0.526 | 0.523 | 0.565 |

### Feature Ablation Study (XGBoost with balanced sample weights)

| Feature Set | Accuracy | Macro F1 | Balanced Acc. | MCC |
|-------------|----------|----------|---------------|-----|
| ESM-2 only (320-d) | 0.865 | 0.531 | 0.508 | 0.581 |
| Handcrafted only (429-d) | 0.781 | 0.247 | 0.246 | 0.304 |
| ESM-2 + Handcrafted (749-d) | 0.865 | 0.502 | 0.473 | 0.570 |
| **ESM-2 + Physicochemical (328-d)** | **0.866** | **0.535** | **0.515** | **0.588** |

### Key Findings

- **ESM-2 embeddings dominate** handcrafted features (Macro F1: 0.53 vs 0.25)
- **Balanced sample weights** significantly improve XGBoost minority-class recall (+14% Macro F1 vs unweighted)
- **ESM-2 + Physicochemical** is the strongest feature combination (F1=0.535, MCC=0.588)
- Adding all handcrafted features to ESM-2 **hurts** performance (noise dilution from 429 extra dimensions)
- **Best model**: XGBoost with balanced sample weights on ESM-2 features
- Per-class weaknesses: Lyase (F1=0.24), Isomerase (F1=0.30) — smallest classes

### Confidence Calibration

| Level | Count | Accuracy | Mean Max Prob |
|-------|-------|----------|---------------|
| High (p ≥ 0.80) | 30,404 (76.5%) | 95.0% | 0.958 |
| Medium (0.50 ≤ p < 0.80) | 6,697 (16.8%) | 65.9% | 0.662 |
| Low (p < 0.50) | 2,663 (6.7%) | 40.5% | 0.409 |

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
