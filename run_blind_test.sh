#!/bin/bash
# Script to run blind test predictions on blind_ec_test.fasta.txt
#
# This script requires:
# 1. All Python dependencies installed (pip install -r requirements.txt)
# 2. ESM-2 model weights downloaded (happens automatically on first run)
# 3. Saved model artifacts in outputs/models/
#
# Usage:
#   bash run_blind_test.sh [ensemble|xgboost]
#
# Options:
#   ensemble (default) - Use ensemble of fine-tuned ESM-2 8M + XGBoost 650M (best performance)
#   xgboost           - Use XGBoost 650M only (faster, slightly lower performance)

set -e  # Exit on error

MODE="${1:-ensemble}"
FASTA="blind_ec_test.fasta.txt"
OUTPUT_DIR="outputs/predictions"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  Blind Test Prediction Pipeline"
echo "=============================================="
echo "Input FASTA: $FASTA"
echo "Mode: $MODE"
echo ""

# Check if FASTA file exists
if [ ! -f "$FASTA" ]; then
    echo "ERROR: Blind test FASTA file not found: $FASTA"
    echo "Expected location: $(pwd)/$FASTA"
    exit 1
fi

# Check if model files exist
if [ ! -f "outputs/models/best_model.joblib" ]; then
    echo "ERROR: Best model not found: outputs/models/best_model.joblib"
    echo "Please run model training first:"
    echo "  python -m src.models.advanced"
    exit 1
fi

# Run predictions based on mode
if [ "$MODE" == "ensemble" ]; then
    echo "Running ENSEMBLE prediction (Fine-tuned ESM-2 8M + XGBoost 650M)..."
    echo "Expected performance: Macro F1 = 62.5%, Accuracy = 89.0%"
    echo ""

    if [ ! -f "outputs/models/finetune_artifact.joblib" ]; then
        echo "ERROR: Fine-tune artifact not found: outputs/models/finetune_artifact.joblib"
        echo "Please run fine-tuning first:"
        echo "  python -m src.models.finetune"
        exit 1
    fi

    python -m src.predict_blind \
        --fasta "$FASTA" \
        --model outputs/models/best_model.joblib \
        --model-finetune outputs/models/finetune_artifact.joblib \
        --output "$OUTPUT_DIR/blind_predictions_ensemble.txt"

    OUTPUT_FILE="$OUTPUT_DIR/blind_predictions_ensemble.txt"

elif [ "$MODE" == "xgboost" ]; then
    echo "Running XGBoost-only prediction (ESM-2 650M + Physicochemical)..."
    echo "Expected performance: Macro F1 = 59.5%, Accuracy = 88.6%"
    echo ""

    python -m src.predict_blind \
        --fasta "$FASTA" \
        --model outputs/models/best_model.joblib \
        --output "$OUTPUT_DIR/blind_predictions_xgboost.txt"

    OUTPUT_FILE="$OUTPUT_DIR/blind_predictions_xgboost.txt"

else
    echo "ERROR: Unknown mode: $MODE"
    echo "Valid modes: ensemble, xgboost"
    exit 1
fi

echo ""
echo "=============================================="
echo "  Predictions Complete!"
echo "=============================================="
echo "Output file: $OUTPUT_FILE"
echo ""
echo "Sample predictions:"
head -5 "$OUTPUT_FILE"
echo "..."
echo ""
echo "File contains $(wc -l < "$OUTPUT_FILE") predictions"
echo ""
