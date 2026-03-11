#!/bin/bash
# Pod setup script for COMP0082 protein classification project
# Run this once after SSH-ing into the pod:
#   bash pod_setup.sh
#
# Before running, upload the .npy data files (see pod_upload.ps1 on your local machine)

set -e

REPO="https://github.com/xhtdby/protein-classificattion-project.git"
WORKDIR="/workspace/protein-classification"

echo "========================================"
echo "  POD SETUP — protein classification"
echo "========================================"

# ── 1. Clone repo ─────────────────────────────────────────────────────────────
if [ ! -d "$WORKDIR" ]; then
    echo "[1/5] Cloning repo..."
    git clone "$REPO" "$WORKDIR"
else
    echo "[1/5] Repo already cloned — pulling latest..."
    cd "$WORKDIR" && git pull
fi
cd "$WORKDIR"

# ── 2. Python environment ──────────────────────────────────────────────────────
echo "[2/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -r requirements.txt

# ── 3. Make output dirs ────────────────────────────────────────────────────────
echo "[3/5] Creating output directories..."
mkdir -p outputs/features outputs/models outputs/figures outputs/predictions

# ── 4. Check .npy files ────────────────────────────────────────────────────────
echo "[4/5] Checking feature files..."
MISSING=0

check_npy() {
    if [ -f "outputs/features/$1" ]; then
        SIZE=$(du -m "outputs/features/$1" | cut -f1)
        echo "  ✓  $1  (${SIZE}MB)"
    else
        echo "  ✗  $1  NOT FOUND"
        MISSING=1
    fi
}

check_npy "esm2_embeddings_esm2_t33_650M_UR50D.npy"
check_npy "handcrafted_features.npy"
check_npy "feature_names.npy"

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "  Some .npy files are missing."
    echo "  Option A: Upload them from your local machine (see pod_upload.ps1)"
    echo "  Option B: Re-extract now (adds ~30-60 min on an A100):"
    echo ""
    echo "    python -m src.models.baseline   # generates handcrafted_features.npy"
    echo "    python -m src.features.embeddings --model 650M --batch-size 8"
    echo ""
    read -p "  Re-extract now? [y/N] " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "  Extracting handcrafted features..."
        python -m src.models.baseline

        echo "  Extracting 650M ESM-2 embeddings (this takes ~30-60 min on A100)..."
        python -m src.features.embeddings --model 650M --batch-size 8
    else
        echo "  Skipping — upload the files then re-run this script or go straight to training."
        exit 1
    fi
fi

# ── 5. Launch training ─────────────────────────────────────────────────────────
echo "[5/5] Starting advanced training with 650M ESM-2..."
echo "  Command: python -m src.models.advanced --esm-model 650M --tune --tune-iter 30"
echo ""

python -u -m src.models.advanced --esm-model 650M --tune --tune-iter 30 \
    2>&1 | tee outputs/advanced_650M_run.txt

echo ""
echo "========================================"
echo "  TRAINING COMPLETE"
echo "========================================"
echo "Results: outputs/advanced_results.json"
echo "Model:   outputs/models/best_model.joblib"
