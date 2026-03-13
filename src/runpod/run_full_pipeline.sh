#!/usr/bin/env bash
# =============================================================================
# src/runpod/run_full_pipeline.sh
#
# Complete end-to-end pipeline on a RunPod A40 (48 GB VRAM).
# Runs every model stage in order and produces all artefacts needed for:
#   - Blind challenge predictions (650M fine-tuned model)
#   - Ensemble predictions (fine-tuned 8M + XGBoost 650M)
#
# Execution order
# ---------------
#   Stage 1  Extract ESM-2 650M embeddings (cached to disk)
#   Stage 2  Train XGBoost on 650M embeddings + physicochemical features
#   Stage 3  Fine-tune ESM-2 8M  (fast baseline, needed for ensemble)
#   Stage 4  Fine-tune ESM-2 650M (primary high-accuracy model)
#   Stage 5  Build ensemble of fine-tuned 8M + XGBoost 650M
#   Stage 6  Generate evaluation plots
#
# Usage
# -----
#   bash src/runpod/run_full_pipeline.sh           # all stages
#   SKIP_STAGES="1 2" bash src/runpod/run_full_pipeline.sh  # skip already-done stages
#
# Estimated total wall-clock time on A40
# ---------------------------------------
#   Stage 1 (650M embeddings):  ~1.5-2.5 h
#   Stage 2 (XGBoost):          ~20-40 min
#   Stage 3 (8M fine-tune):     ~1-2 h
#   Stage 4 (650M fine-tune):   ~5-8 h
#   Stage 5 (ensemble eval):    ~5-15 min
#   Stage 6 (plots):            ~2-5 min
#   Total:                      ~9-14 h
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[pipeline]${NC} $*"; }
stage()   { echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"; \
            echo -e "${CYAN}  STAGE $1: $2${NC}"; \
            echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════${NC}"; }
warn()    { echo -e "${YELLOW}[pipeline]${NC} $*"; }
error()   { echo -e "${RED}[pipeline]${NC} $*" >&2; exit 1; }
skip_msg(){ echo -e "${YELLOW}[pipeline]${NC} Skipping stage $1 (in SKIP_STAGES)"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON=$(command -v python3 || command -v python)

# Comma- or space-separated list of stage numbers to skip, e.g. "1 2"
SKIP_STAGES="${SKIP_STAGES:-}"

should_skip() {
    local n="$1"
    for s in ${SKIP_STAGES}; do
        [[ "$s" == "$n" ]] && return 0
    done
    return 1
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "Project root : ${PROJECT_ROOT}"
info "Python       : ${PYTHON}"

"${PYTHON}" -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available.  Run setup.sh first.', file=sys.stderr)
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print(f'GPU: {props.name}  |  VRAM: {total/1e9:.1f} GB  ({free/1e9:.1f} GB free)')
"

LOG_DIR="${PROJECT_ROOT}/outputs"
mkdir -p \
    "${LOG_DIR}/models" \
    "${LOG_DIR}/figures" \
    "${LOG_DIR}/features" \
    "${LOG_DIR}/predictions"

PIPELINE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
info "Pipeline log : ${PIPELINE_LOG}"

# Tee all output to the pipeline log
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

T_PIPELINE_START=$(date +%s)

# ===========================================================================
# Stage 1 — Extract ESM-2 650M embeddings
# ===========================================================================
stage 1 "Extract ESM-2 650M embeddings"
if should_skip 1; then
    skip_msg 1
else
    EMB_CACHE="${LOG_DIR}/features/esm2_embeddings_esm2_t33_650M_UR50D.npy"
    if [[ -f "${EMB_CACHE}" ]]; then
        warn "Embedding cache already exists at ${EMB_CACHE} -- skipping extraction."
        warn "Pass SKIP_STAGES='1' or delete the cache to force re-extraction."
    else
        T1=$(date +%s)
        "${PYTHON}" -m src.features.embeddings --model 650M --batch-size 8
        info "Stage 1 done in $(( ($(date +%s) - T1) / 60 )) min"
    fi
fi

# ===========================================================================
# Stage 2 — XGBoost on 650M embeddings + physicochemical features
# ===========================================================================
stage 2 "Train XGBoost (650M embeddings + physicochemical)"
if should_skip 2; then
    skip_msg 2
else
    T2=$(date +%s)
    "${PYTHON}" -m src.models.advanced
    info "Stage 2 done in $(( ($(date +%s) - T2) / 60 )) min"
fi

# ===========================================================================
# Stage 3 — Fine-tune ESM-2 8M  (needed for the ensemble)
# ===========================================================================
stage 3 "Fine-tune ESM-2 8M (ensemble component)"
if should_skip 3; then
    skip_msg 3
else
    T3=$(date +%s)
    # A40 can handle larger batches for the 8M model
    "${PYTHON}" -m src.models.finetune \
        --epochs 10 \
        --batch-size 32 \
        --grad-accum 1 \
        --max-len 512
    info "Stage 3 done in $(( ($(date +%s) - T3) / 60 )) min"
fi

# ===========================================================================
# Stage 4 — Fine-tune ESM-2 650M  (primary high-accuracy model)
# ===========================================================================
stage 4 "Fine-tune ESM-2 650M (primary model)"
if should_skip 4; then
    skip_msg 4
else
    T4=$(date +%s)
    bash "${SCRIPT_DIR}/train_650m.sh"
    info "Stage 4 done in $(( ($(date +%s) - T4) / 60 )) min"
fi

# ===========================================================================
# Stage 5 — Build ensemble (fine-tuned 8M + XGBoost 650M)
# ===========================================================================
stage 5 "Build ensemble (fine-tuned 8M + XGBoost 650M)"
if should_skip 5; then
    skip_msg 5
else
    T5=$(date +%s)
    "${PYTHON}" -m src.models.ensemble
    info "Stage 5 done in $(( ($(date +%s) - T5) / 60 )) min"
fi

# ===========================================================================
# Stage 6 — Generate evaluation plots
# ===========================================================================
stage 6 "Generate evaluation plots"
if should_skip 6; then
    skip_msg 6
else
    T6=$(date +%s)
    "${PYTHON}" -m src.generate_plots 2>/dev/null || warn "generate_plots failed (non-fatal)"
    info "Stage 6 done in $(( ($(date +%s) - T6) / 60 )) min"
fi

# ===========================================================================
# Summary
# ===========================================================================
T_TOTAL=$(( ($(date +%s) - T_PIPELINE_START) / 60 ))

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  PIPELINE COMPLETE  (total time: ${T_TOTAL} min)$(printf '%*s' $((37 - ${#T_TOTAL})) '')║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Key artefacts:"
info "  650M fine-tuned weights : outputs/models/finetune_650m_final.pt"
info "  650M joblib artifact    : outputs/models/finetune_650m_artifact.joblib"
info "  650M CV results JSON    : outputs/finetune_650m_results.json"
info "  650M confusion matrix   : outputs/figures/cm_finetune_650m.png"
info "  Ensemble results        : outputs/ensemble_results.json"
info "  Pipeline log            : ${PIPELINE_LOG}"
echo ""
info "Blind predictions (650M fine-tuned, highest accuracy):"
info "  python -m src.predict_blind \\"
info "      --fasta <blind_test.fasta> \\"
info "      --model-ft650m outputs/models/finetune_650m_artifact.joblib \\"
info "      --output outputs/predictions/blind_predictions.txt"
echo ""
info "Ensemble predictions (8M fine-tuned + XGBoost 650M):"
info "  python -m src.predict_blind \\"
info "      --fasta <blind_test.fasta> \\"
info "      --model outputs/models/best_model.joblib \\"
info "      --model-finetune outputs/models/finetune_artifact.joblib \\"
info "      --output outputs/predictions/blind_ensemble_predictions.txt"
