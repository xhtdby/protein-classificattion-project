#!/usr/bin/env bash
# =============================================================================
# runpod/train_650m.sh
#
# Launch ESM-2 650M fine-tuning on a RunPod A40 (48 GB VRAM).
#
# Usage
# -----
#   bash runpod/train_650m.sh                # default A40 settings
#   bash runpod/train_650m.sh --epochs 20    # override epoch count
#   GRAD_CKPT=1 bash runpod/train_650m.sh   # enable gradient checkpointing
#
# Default hyper-parameters (tuned for A40 48 GB)
# ------------------------------------------------
#   --epochs       15          (early stopping at patience=7)
#   --batch-size   8           (per-step; effective_batch = 8×2 = 16)
#   --grad-accum   2
#   --max-len      1022        (full ESM-2 context window)
#   --backbone-lr  1e-5        (outer ESM-2 layer, decays inward via LLRD)
#   --head-lr      5e-5
#   --llrd-decay   0.9
#   --patience     7
#
# Expected VRAM usage: ~20-28 GB in bfloat16 (fits A40 comfortably)
# Add GRAD_CKPT=1 to reduce to ~14-18 GB at ~20% compute overhead.
#
# Expected wall-clock time on A40
# --------------------------------
#   ~39764 sequences total → ~31811 train / ~7953 val per fold
#   ~1200 steps per epoch at batch=8, 5 folds × 15 epochs ≈ 4-6 hours
#   Final retrain adds ~1-1.5 hours
#   Total estimate: 5-8 hours
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[train_650m]${NC} $*"; }
warn()  { echo -e "${YELLOW}[train_650m]${NC} $*"; }
error() { echo -e "${RED}[train_650m]${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON=$(command -v python3 || command -v python)

# ---------------------------------------------------------------------------
# GPU sanity check
# ---------------------------------------------------------------------------
"${PYTHON}" -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available.  Run setup.sh first.', file=sys.stderr)
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(f'GPU: {props.name}  |  VRAM: {total/1e9:.1f} GB  ({free/1e9:.1f} GB free)')
"

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_LEN="${MAX_LEN:-1022}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
HEAD_LR="${HEAD_LR:-5e-5}"
LLRD_DECAY="${LLRD_DECAY:-0.9}"
PATIENCE="${PATIENCE:-7}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-}"  # blank = same as EPOCHS

EXTRA_ARGS=("$@")   # forward any extra CLI arguments

GRAD_CKPT_FLAG=""
if [[ "${GRAD_CKPT:-0}" == "1" ]]; then
    GRAD_CKPT_FLAG="--use-grad-ckpt"
    warn "Gradient checkpointing ENABLED (lower VRAM, ~20% slower)"
fi

RETRAIN_FLAG=""
if [[ -n "${RETRAIN_EPOCHS}" ]]; then
    RETRAIN_FLAG="--retrain-epochs ${RETRAIN_EPOCHS}"
fi

info "Starting ESM-2 650M fine-tuning..."
info "  epochs=${EPOCHS}  batch=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}"
info "  max_len=${MAX_LEN}  backbone_lr=${BACKBONE_LR}  head_lr=${HEAD_LR}"
info "  llrd_decay=${LLRD_DECAY}  patience=${PATIENCE}"
info "  Project root: ${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Run training (redirect stdout + stderr to log file as well as terminal)
# ---------------------------------------------------------------------------
LOG_DIR="${PROJECT_ROOT}/outputs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_650m_$(date +%Y%m%d_%H%M%S).log"

info "Log file: ${LOG_FILE}"
info "---"

"${PYTHON}" -m src.models.finetune_650m \
    --epochs        "${EPOCHS}" \
    --batch-size    "${BATCH_SIZE}" \
    --grad-accum    "${GRAD_ACCUM}" \
    --max-len       "${MAX_LEN}" \
    --backbone-lr   "${BACKBONE_LR}" \
    --head-lr       "${HEAD_LR}" \
    --llrd-decay    "${LLRD_DECAY}" \
    --patience      "${PATIENCE}" \
    ${GRAD_CKPT_FLAG} \
    ${RETRAIN_FLAG} \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

info "---"
info "Training complete!"
info "Artifacts:"
info "  CV results  : ${PROJECT_ROOT}/outputs/finetune_650m_results.json"
info "  Final model : ${PROJECT_ROOT}/outputs/models/finetune_650m_final.pt"
info "  Artifact    : ${PROJECT_ROOT}/outputs/models/finetune_650m_artifact.joblib"
info "  Conf. matrix: ${PROJECT_ROOT}/outputs/figures/cm_finetune_650m.png"
info "  Log         : ${LOG_FILE}"
info ""
info "Next step: run blind predictions with"
info "  python -m src.predict_blind --fasta <FASTA> \\"
info "      --model-ft650m outputs/models/finetune_650m_artifact.joblib \\"
info "      --output outputs/predictions/blind_predictions.txt"
