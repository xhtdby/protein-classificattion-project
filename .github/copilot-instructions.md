# COMP0082 – Protein Enzyme Classification Project

## Task

7-class supervised classification of protein sequences into EC (Enzyme Commission) classes.
**This coursework is worth 40% of the module marks.**

| Class | Label          | Count  |
|-------|----------------|--------|
| 0     | Not an enzyme  | 32 410 |
| 1     | Oxidoreductase | 1 184  |
| 2     | Transferase    | 2 769  |
| 3     | Hydrolase      | 2 108  |
| 4     | Lyase          | 600    |
| 5     | Isomerase      | 411    |
| 6     | Ligase         | 282    |

The dataset is **heavily class-imbalanced** (class 0 ≈ 82%). All strategies must account for this.

## Data Rules

- FASTA files live in the workspace root: `class0_rep_seq.fasta.txt`, `ec1_rep_seq.fasta.txt` … `ec6_rep_seq.fasta.txt`.
- All sequences are non-homologous — no deduplication needed.
- **Use ONLY the provided datasets.** Never download, scrape, or reference external sequence databases.
- The blind challenge test set is **NOT available yet**. Do not fabricate or simulate it. Build a `predict_blind.py` script that can accept any FASTA file at runtime.

## Allowed / Prohibited

| Allowed | Prohibited |
|---------|-----------|
| `scikit-learn`, `PyTorch`, `TensorFlow/Keras`, `XGBoost`, `LightGBM` | BLAST, HMMer, Pfam, InterPro, UniProt API |
| `BioPython` (`Bio.SeqIO`, `Bio.SeqUtils`) | Any specialised bioinformatics tool or database |
| Protein language models as feature extractors (ESM-2, ProtTrans) | Downloading pre-computed annotations or alignments |
| `matplotlib`, `seaborn`, `pandas`, `numpy`, `scipy` | External sequence data beyond the 7 provided files |

## Project Structure

```
protein-classification/
├── .github/
│   └── copilot-instructions.md
├── src/
│   ├── __init__.py
│   ├── data_loading.py            # FASTA parsing, label assignment, train/val splits
│   ├── features/
│   │   ├── __init__.py
│   │   ├── composition.py         # AA composition, dipeptide, k-mer frequencies
│   │   ├── physicochemical.py     # Charge, hydrophobicity, MW, pI
│   │   └── embeddings.py          # ESM-2 / ProtTrans embedding extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py            # Logistic Regression, Random Forest
│   │   └── advanced.py            # XGBoost / neural-net / ensemble
│   ├── training.py                # CV loop, class-imbalance handling, metric logging
│   ├── evaluation.py              # Metrics, confusion matrix, plots
│   ├── interpretability.py        # Feature importance, ablation, SHAP / attribution
│   ├── confidence.py              # Probability → High / Medium / Low mapping
│   └── predict_blind.py           # Load model + FASTA → formatted predictions
├── notebooks/                     # Exploratory analysis (optional)
├── outputs/
│   ├── models/                    # Saved model artefacts
│   ├── figures/                   # Plots for the report
│   └── predictions/               # Blind challenge output files
├── requirements.txt
└── README.md
```

Do not create files outside `src/`, `notebooks/`, or `outputs/`.

## Data-Leakage Prevention (CRITICAL)

These rules are **non-negotiable**:

1. **Stratified K-Fold first.** Create fold indices before any feature computation or normalisation.
2. **Fit scalers/encoders on training fold only.** Never `.fit()` or `.fit_transform()` on the full dataset.
3. **SMOTE / oversampling on training fold only.** Never oversample before splitting.
4. **No target leakage.** Features derive solely from amino acid sequence — never from the label.
5. **PLM embeddings are stateless.** ESM-2 forward pass has no label info — safe to pre-compute once. But normalisation of embeddings still follows rule 2.

## Feature Engineering

Implement in this order (1 = first, 5 = highest impact):

1. **Amino acid composition** — 20-dim normalised frequency vector
2. **Sequence length** — single scalar
3. **Dipeptide frequencies** — 400-dim vector (optionally reduce via PCA)
4. **Physicochemical properties** — charge, hydrophobicity (Kyte-Doolittle), molecular weight, isoelectric point, aromaticity
5. **Protein language model embeddings** — ESM-2 (`esm2_t6_8M_UR50D` for speed or `esm2_t33_650M_UR50D` for accuracy), mean-pool → 320- or 1280-dim vector

## Modelling Conventions

- **Stratified 5-fold cross-validation**, imbalanced distribution preserved in every fold.
- Report **all four metrics** per experiment: Accuracy, Macro F1, Balanced Accuracy, MCC.
- Address class imbalance via `class_weight='balanced'`, `scale_pos_weight`, weighted loss, or SMOTE (train fold only).
- Final model outputs **probability vector** over 7 classes (softmax / `predict_proba`).
- Compare at least a **baseline** and an **advanced** model.

## Interpretability (Extra Credit)

Implement **at least two**:

- Feature importance ranking (permutation or Gini importance)
- Ablation study (drop feature groups, re-evaluate)
- SHAP values or gradient-based attribution
- Confusion matrix error analysis (which classes are most confused and why)

## Confidence Scoring

Map `p = max(predict_proba)` to confidence:

| Level  | Condition       | Challenge score |
|--------|-----------------|-----------------|
| High   | p ≥ 0.80        | ±1              |
| Medium | 0.50 ≤ p < 0.80 | ±0.5            |
| Low    | p < 0.50        | 0               |

Blind challenge output format (one line per sequence, no header):
```
SEQ01 1 Confidence High
SEQ02 0 Confidence Medium
```

## Code Style

- Python 3.10+; PEP 8.
- `pathlib.Path` for all file paths — never raw string concatenation.
- Parse FASTA with `Bio.SeqIO`.
- Seed **all** RNGs: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- Save sklearn models with `joblib`; PyTorch with `torch.save`.
- Every script: `if __name__ == "__main__":` block **and** importable as module.
- Pin all dependencies in `requirements.txt`.

## Report Structure (≈2 500 words)

1. **Introduction** — problem statement, biological context
2. **Methods** — feature engineering, model selection justification, experimental design
3. **Results** — CV metrics table, confusion matrix, approach comparison
4. **Interpretability** — feature importance / ablation / attribution findings
5. **Discussion** — limitations, error analysis, what worked vs didn't
6. **Conclusion** — key findings, potential improvements
7. **Appendix** — code listings, blind challenge predictions (plain text, no images)
