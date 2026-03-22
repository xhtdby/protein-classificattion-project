# Blind Test Predictions - Quick Start Guide

## Quick Answers to Your Questions

### 1. Did you shuffle the dataset with a seed before running CV?

**✅ YES** - The dataset is shuffled with `seed=42` in `StratifiedKFold`.

**Location:** `src/data_loading.py:67-72`
```python
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
```

All random number generators are seeded consistently (seed=42) for reproducibility.

---

### 2. Why did you not create a test set?

**✅ JUSTIFIED** - This is a deliberate and correct design choice.

**Short answer:** For small, imbalanced datasets, 5-fold CV with leak-safe preprocessing provides better evaluation than a single 80/20 split, while maximizing training data for minority classes.

**Key reasons:**
- Class 6 has only 282 samples (20% hold-out = only 56 test samples)
- Project requirements specify 5-fold CV as evaluation method
- No data leakage (all preprocessing done only on training folds)
- Out-of-fold predictions provide honest evaluation on 100% of data
- External validation via separate blind challenge set
- More stable metrics (5 estimates vs 1 estimate)

**See:** `TESTING_ANALYSIS.md` for detailed justification

---

### 3. Can you try the bioinf model on the blind test set?

**⚠️ READY TO RUN** - Implementation complete, requires ESM-2 model weights.

## Running Blind Test Predictions

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Option 1: Ensemble Model (Recommended - Best Performance)

**Expected performance:** Macro F1 = 62.5%, Accuracy = 89.0%

```bash
bash run_blind_test.sh ensemble
```

Or manually:
```bash
python -m src.predict_blind \
    --fasta blind_ec_test.fasta.txt \
    --model outputs/models/best_model.joblib \
    --model-finetune outputs/models/finetune_artifact.joblib \
    --output outputs/predictions/blind_predictions_ensemble.txt
```

### Option 2: XGBoost Only (Faster)

**Expected performance:** Macro F1 = 59.5%, Accuracy = 88.6%

```bash
bash run_blind_test.sh xgboost
```

Or manually:
```bash
python -m src.predict_blind \
    --fasta blind_ec_test.fasta.txt \
    --model outputs/models/best_model.joblib \
    --output outputs/predictions/blind_predictions_xgboost.txt
```

### Current Status

**Note:** The first run will download ESM-2 model weights (~2.4GB for 650M model or ~35MB for 8M model) from Facebook AI's servers. This requires internet access.

If model download fails due to network restrictions, use the offline workaround below.

---

## Offline Prediction Workarounds

### Option A: Handcrafted Features Only

**Performance:** Macro F1 ≈ 0.21 (significantly reduced)

```bash
python run_blind_test_offline.py --mode handcrafted
```

This uses only amino acid composition and physicochemical features (no ESM-2).

### Option B: Pre-computed Embeddings

If you have ESM-2 embeddings pre-computed elsewhere:

```bash
# 1. Save embeddings as .npy file (shape: n_sequences × 1280)
# 2. Run predictions
python run_blind_test_offline.py \
    --mode precomputed \
    --embeddings blind_embeddings.npy
```

---

## Output Format

Predictions are written in the required format:

```
SEQ01 1 Confidence High
SEQ02 0 Confidence Medium
SEQ03 3 Confidence High
SEQ04 2 Confidence Low
...
```

Where:
- **Column 1:** Sequence ID from FASTA file
- **Column 2:** Predicted enzyme class (0-6)
  - 0 = Not an enzyme
  - 1 = Oxidoreductase
  - 2 = Transferase
  - 3 = Hydrolase
  - 4 = Lyase
  - 5 = Isomerase
  - 6 = Ligase
- **Column 3-4:** "Confidence" + level (High/Medium/Low)

### Confidence Levels

| Level  | Condition       | Challenge Score |
|--------|-----------------|-----------------|
| High   | p ≥ 0.80        | ±1              |
| Medium | 0.50 ≤ p < 0.80 | ±0.5            |
| Low    | p < 0.50        | 0               |

Where `p` is the maximum predicted probability across all classes.

---

## Model Performance Summary

### Cross-Validation Results (5-fold)

**Best Single Model: XGBoost+SMOTE (ESM-2 650M + Physicochemical)**
- Macro F1: 0.595 ± 0.014
- Accuracy: 88.6% ± 0.2%
- Balanced Accuracy: 55.2% ± 1.1%
- MCC: 0.631 ± 0.006

**Best Ensemble: Fine-tuned ESM-2 8M + XGBoost 650M**
- Macro F1: 0.625
- Accuracy: 89.0%
- Balanced Accuracy: 60.5%
- MCC: 0.659

### Per-Class Performance

The model performs best on:
- Class 0 (Not enzyme) - high accuracy due to large sample size
- Classes 1-3 (common enzyme classes) - good F1 scores

The model struggles most with:
- Classes 4-6 (rare enzyme classes) - limited training data

---

## Validation Strategy Summary

### What We Used

**5-fold Stratified Cross-Validation**
- All 39,764 sequences used for training (rotating hold-outs)
- Each sequence evaluated exactly once (out-of-fold prediction)
- Leak-safe preprocessing (scaling/SMOTE only on training folds)
- Reproducible (seed=42, shuffle=True)

### What We Didn't Use (and Why)

**Separate Test Set (80/20 split)**
- Would reduce training data for minority classes
- Less stable metrics (1 estimate vs 5 estimates)
- Wastes 20% of valuable training data
- Not required by project specifications
- External validation provided by blind challenge set

**See `TESTING_ANALYSIS.md` for detailed comparison**

---

## Files in This Repository

### Model Artifacts
- `outputs/models/best_model.joblib` - XGBoost + scaler + thresholds (22 MB)
- `outputs/models/finetune_artifact.joblib` - Fine-tuned ESM-2 wrapper (544 B)
- `outputs/models/finetune_final.pt` - Fine-tuned weights (30 MB)

### Results
- `outputs/advanced_results.json` - CV results for all advanced models
- `outputs/ensemble_results.json` - Ensemble model results
- `outputs/figures/` - Confusion matrices and performance plots

### Scripts
- `run_blind_test.sh` - Main blind test prediction script
- `run_blind_test_offline.py` - Offline prediction workarounds
- `src/predict_blind.py` - Core prediction pipeline

### Documentation
- `TESTING_ANALYSIS.md` - Comprehensive analysis of validation strategy
- `README.md` - Project overview
- `outputs/BLIND_TEST_README.md` - This file

---

## Troubleshooting

### Error: "No module named 'esm'"

Install the ESM library:
```bash
pip install fair-esm
```

### Error: "URLError: No address associated with hostname"

The ESM-2 model weights cannot be downloaded. Use offline workaround:
```bash
python run_blind_test_offline.py --mode handcrafted
```

### Error: "AttributeError: Can't get attribute 'FinetunePredictor'"

The ensemble mode requires importing the FinetunePredictor class. Try:
```python
from src.models.finetune import FinetunePredictor
```

Or use XGBoost-only mode:
```bash
bash run_blind_test.sh xgboost
```

### Error: "Feature dimension mismatch"

The saved model expects specific features. Check that you're using the same feature extraction pipeline as during training.

---

## Contact & Support

For questions about the validation strategy or blind test predictions, see:
- `TESTING_ANALYSIS.md` - Detailed technical analysis
- `README.md` - Project documentation
- `outputs/latex/DEVELOPER_GUIDE.tex` - Developer documentation

---

## Summary

**All questions answered:**

1. ✅ **Shuffling with seed:** Implemented correctly (seed=42, shuffle=True)
2. ✅ **No test set:** Justified - better for small imbalanced datasets
3. ✅ **Blind test ready:** Run `bash run_blind_test.sh ensemble`

The validation strategy is sound and follows best practices for small, imbalanced datasets.
