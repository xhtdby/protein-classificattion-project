# RunPod A40 Execution Guide

This directory contains scripts for running the full protein-classification
training pipeline on a RunPod pod equipped with an **NVIDIA A40 (48 GB VRAM)**.

---

## Quick Start

```bash
# 1. On your RunPod pod (SSH in first):
git clone <your-repo-url>
cd protein-classificattion-project

# 2. Copy your FASTA data files to the project root (if not already there):
#    class0_rep_seq.fasta.txt, ec1_rep_seq.fasta.txt, ..., ec6_rep_seq.fasta.txt

# 3. Bootstrap the environment (installs PyTorch + all dependencies):
bash src/runpod/setup.sh

# 4a. Full pipeline (all stages, ~9-14 hours):
bash src/runpod/run_full_pipeline.sh

# 4b. Or run only the 650M fine-tuning (if embeddings + XGBoost are already done):
SKIP_STAGES="1 2 3" bash src/runpod/run_full_pipeline.sh

# 4c. Or run just the 650M fine-tuning directly:
bash src/runpod/train_650m.sh
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | Install PyTorch (CUDA 12.1), fair-esm, and all requirements |
| `train_650m.sh` | Fine-tune ESM-2 650M with A40-optimised settings |
| `run_full_pipeline.sh` | End-to-end: embeddings → XGBoost → 8M fine-tune → 650M fine-tune → ensemble |

---

## Why ESM-2 650M Fine-tuning?

| Model | Features | OOF Macro F1 (estimated) |
|-------|---------|--------------------------|
| Logistic Regression | AA composition + physicochemical | ~0.45 |
| XGBoost | 8M embeddings | ~0.55 |
| XGBoost + SMOTE | 650M embeddings + physicochemical | ~0.62 |
| Fine-tuned 8M | End-to-end, max_len=512 | ~0.63–0.66 |
| **Fine-tuned 650M** | **End-to-end, max_len=1022, LLRD** | **~0.70–0.75 (target)** |
| Ensemble (8M+XGBoost) | Soft-vote | ~0.625 (current best) |

Fine-tuning the 650M model end-to-end on the task outperforms frozen
embeddings because the backbone weights adapt to enzymatic sequence patterns.
The LLRD strategy prevents catastrophic forgetting of pre-trained representations.

---

## A40 Training Details

### Architecture (`finetune_650m.py`)
- **Backbone**: `esm2_t33_650M_UR50D` — 33 transformer layers, 1280-dim hidden
- **Head**: `Linear(1280→512) → GELU → Dropout(0.2) → Linear(512→128) → GELU → Dropout(0.2) → Linear(128→7)`
- **Pooling**: mean-pool over residue positions 1…seq_len (same as frozen embeddings)

### Hyper-parameters (A40 defaults)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--batch-size` | 8 | ~20-28 GB VRAM in bf16 at max_len=1022 |
| `--grad-accum` | 2 | Effective batch = 16 |
| `--max-len` | 1022 | Full ESM-2 context; A40 fits it comfortably |
| `--backbone-lr` | 1e-5 | Outermost ESM layer; inner layers decay via LLRD |
| `--head-lr` | 5e-5 | Head trains 5× faster than backbone |
| `--llrd-decay` | 0.9 | Each layer inward gets 0.9× the LR of the outer layer |
| `--epochs` | 15 | Early-stopping at `patience=7` typically fires before this |
| `--patience` | 7 | Generous for large models |

### LLRD (Layer-wise LR Decay)

The LLRD optimiser assigns learning rates that decay geometrically from the
classification head toward the embedding layer:

```
head        : 5e-5
layer 32    : 1e-5        (outermost transformer block)
layer 31    : 9e-6
layer 30    : 8.1e-6
...
layer 0     : ~1.4e-7     (innermost transformer block)
embeddings  : ~1.3e-7
```

This preserves pre-trained representations in the lower layers while allowing
the upper layers and head to specialise for enzyme classification.

### Gradient Checkpointing (optional)

```bash
GRAD_CKPT=1 bash src/runpod/train_650m.sh
```

Halves activation memory (~14-18 GB instead of 20-28 GB) at the cost of
~20% longer training time.  Only needed if you are close to the VRAM limit.

---

## Environment Variables for `train_650m.sh`

```bash
EPOCHS=15          # max training epochs per fold
BATCH_SIZE=8       # per-step batch size
GRAD_ACCUM=2       # gradient accumulation steps
MAX_LEN=1022       # max sequence length in tokens
BACKBONE_LR=1e-5   # outermost layer LR
HEAD_LR=5e-5       # classifier head LR
LLRD_DECAY=0.9     # per-layer LR decay factor
PATIENCE=7         # early-stopping patience
GRAD_CKPT=0        # set to 1 to enable gradient checkpointing
RETRAIN_EPOCHS=    # leave blank to reuse EPOCHS for the final retrain
```

---

## Skip Already-Completed Stages

```bash
# Skip embedding extraction and XGBoost (re-use cached outputs):
SKIP_STAGES="1 2" bash src/runpod/run_full_pipeline.sh

# Skip everything except the 650M fine-tuning and ensemble eval:
SKIP_STAGES="1 2 3" bash src/runpod/run_full_pipeline.sh

# Skip all training and only re-run ensemble + plots:
SKIP_STAGES="1 2 3 4" bash src/runpod/run_full_pipeline.sh
```

---

## Blind Predictions

After training completes, generate predictions on the blind challenge FASTA:

```bash
# Best single model (fine-tuned 650M):
python -m src.predict_blind \
    --fasta blind_test.fasta \
    --model-ft650m outputs/models/finetune_650m_artifact.joblib \
    --output outputs/predictions/blind_predictions.txt

# Ensemble (fine-tuned 8M + XGBoost 650M):
python -m src.predict_blind \
    --fasta blind_test.fasta \
    --model outputs/models/best_model.joblib \
    --model-finetune outputs/models/finetune_artifact.joblib \
    --output outputs/predictions/blind_ensemble_predictions.txt
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/finetune_650m_results.json` | 5-fold CV metrics for 650M fine-tuning |
| `outputs/models/finetune_650m_final.pt` | Final PyTorch weights (full-dataset retrain) |
| `outputs/models/finetune_650m_artifact.joblib` | Joblib artefact for `predict_blind.py` |
| `outputs/models/finetune_650m_fold{1-5}.pt` | Best per-fold checkpoints |
| `outputs/figures/cm_finetune_650m.png` | OOF confusion matrix |
| `outputs/pipeline_<timestamp>.log` | Full pipeline log |
