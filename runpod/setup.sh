#!/usr/bin/env bash
# =============================================================================
# runpod/setup.sh
#
# One-shot environment bootstrap for the protein-classification project on a
# RunPod pod (or any Ubuntu/Debian machine) equipped with an A40 GPU (48 GB).
#
# Usage
# -----
#   bash runpod/setup.sh
#
# What it does
# ------------
#   1. Checks that a CUDA-capable GPU is present.
#   2. Installs / upgrades pip.
#   3. Installs PyTorch 2.x with CUDA 12.1 (matches the A40 driver stack).
#   4. Installs fair-esm (the Meta ESM-2 library).
#   5. Installs all remaining project requirements.
#   6. Creates the required output directories.
#   7. Prints a sanity-check summary.
#
# Notes
# -----
#   - Run as the user that will execute training (not as root).
#   - If you need a specific CUDA version, edit the --index-url below.
#   - The script is idempotent: re-running it is safe.
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
error() { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Locate the project root (directory containing this script's parent)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
info "Project root: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# 1. Verify CUDA GPU
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Is the NVIDIA driver installed?"
fi
info "GPU(s) detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader | sed 's/^/  /'

# ---------------------------------------------------------------------------
# 2. Python version check (3.10+)
# ---------------------------------------------------------------------------
PYTHON=$(command -v python3 || command -v python)
PY_VER=$("${PYTHON}" -c "import sys; print(sys.version_info[:2])")
info "Python: ${PYTHON}  (${PY_VER})"
"${PYTHON}" -c "
import sys
if sys.version_info < (3, 10):
    print('ERROR: Python 3.10+ required', file=sys.stderr)
    sys.exit(1)
"

# ---------------------------------------------------------------------------
# 3. Upgrade pip / setuptools
# ---------------------------------------------------------------------------
info "Upgrading pip and setuptools..."
"${PYTHON}" -m pip install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 4. PyTorch 2.x with CUDA 12.1
#    (A40 supports CUDA 11.x and 12.x; 12.1 wheels are stable on RunPod)
# ---------------------------------------------------------------------------
info "Installing PyTorch 2.x (CUDA 12.1)..."
"${PYTHON}" -m pip install --quiet \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is accessible from PyTorch
"${PYTHON}" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after PyTorch install!'
print(f'  torch={torch.__version__}  CUDA={torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# 5. fair-esm (Meta ESM-2)
# ---------------------------------------------------------------------------
info "Installing fair-esm..."
"${PYTHON}" -m pip install --quiet "fair-esm>=2.0"

# Pre-download the 650M model weights to avoid a timeout during training.
# The weights (~1.3 GB) are cached in ~/.cache/torch/hub/checkpoints/.
info "Pre-downloading ESM-2 650M model weights (~1.3 GB, please wait)..."
"${PYTHON}" -c "
import esm
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
print('  ESM-2 650M downloaded successfully')
del model
" || warn "ESM-2 650M pre-download failed (will retry at training time)"

# ---------------------------------------------------------------------------
# 6. Remaining project requirements
# ---------------------------------------------------------------------------
info "Installing project requirements from requirements.txt..."
"${PYTHON}" -m pip install --quiet -r requirements.txt

# ---------------------------------------------------------------------------
# 7. Create output directories
# ---------------------------------------------------------------------------
info "Creating output directories..."
mkdir -p \
    "${PROJECT_ROOT}/outputs/models" \
    "${PROJECT_ROOT}/outputs/figures" \
    "${PROJECT_ROOT}/outputs/features" \
    "${PROJECT_ROOT}/outputs/predictions"

# ---------------------------------------------------------------------------
# 8. Sanity-check imports
# ---------------------------------------------------------------------------
info "Sanity-checking key imports..."
"${PYTHON}" -c "
import torch, esm, numpy, pandas, sklearn, xgboost, Bio, joblib, tqdm
print('  All key packages imported successfully')
"

info "Setup complete!  Run  bash runpod/train_650m.sh  to start fine-tuning."
