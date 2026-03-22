"""
Helper script to create predictions when ESM-2 model weights are unavailable.

This script demonstrates how to make predictions using pre-computed embeddings
or handcrafted features only.

Usage:
    # Option 1: Use handcrafted features only (no ESM-2)
    python run_blind_test_offline.py --mode handcrafted

    # Option 2: Use pre-computed embeddings (if available)
    python run_blind_test_offline.py --mode precomputed --embeddings blind_embeddings.npy
"""

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
from Bio import SeqIO

from src.features.composition import extract_composition_features
from src.features.physicochemical import extract_physicochemical_features
from src.confidence import assign_confidence

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_sequences(fasta_path: Path) -> tuple[list[str], list[str]]:
    """Parse FASTA file and return (seq_ids, sequences)."""
    seq_ids = []
    sequences = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))
    return seq_ids, sequences


def predict_with_handcrafted_only(
    fasta_path: Path,
    output_path: Path,
) -> None:
    """Make predictions using only handcrafted features (no ESM-2).

    Note: This will have reduced performance compared to ESM-2 models.
    Expected Macro F1: ~0.21 vs 0.60 with ESM-2
    """
    logger.info("Mode: Handcrafted features only (no ESM-2)")
    logger.warning("Performance will be reduced without ESM-2 embeddings")
    logger.warning("Expected Macro F1: ~0.21 vs 0.60 with ESM-2")

    # Load sequences
    seq_ids, sequences = load_sequences(fasta_path)
    logger.info("Loaded %d sequences from %s", len(sequences), fasta_path)

    # Extract handcrafted features
    logger.info("Extracting handcrafted features...")
    comp = extract_composition_features(sequences)
    phys = extract_physicochemical_features(sequences)
    X = np.hstack([comp, phys])
    logger.info("Feature shape: %s", X.shape)

    # Load a baseline model trained on handcrafted features
    # Note: The saved best_model.joblib expects ESM-2 features, so we'd need
    # a separate model trained only on handcrafted features
    baseline_path = Path("outputs/models/baseline_rf.joblib")
    if not baseline_path.exists():
        logger.error("Baseline model not found: %s", baseline_path)
        logger.error("Please train a baseline model first:")
        logger.error("  python -m src.models.baseline")
        return

    artefact = joblib.load(baseline_path)
    model = artefact["model"]
    scaler = artefact.get("scaler")

    # Scale features
    if scaler is not None:
        X = scaler.transform(X)

    # Predict
    logger.info("Making predictions...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        y_pred = model.predict(X)

    # Assign confidence
    if y_proba is not None:
        confidence_levels = assign_confidence(y_proba)
    else:
        confidence_levels = ["Low"] * len(y_pred)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seq_id, pred, conf in zip(seq_ids, y_pred, confidence_levels):
            f.write(f"{seq_id} {pred} Confidence {conf}\n")

    logger.info("Predictions written to %s", output_path)

    # Summary
    from collections import Counter
    pred_counts = Counter(int(p) for p in y_pred)
    conf_counts = Counter(confidence_levels)
    print(f"\nClass distribution: {dict(sorted(pred_counts.items()))}")
    print(f"Confidence distribution: {dict(sorted(conf_counts.items()))}")


def predict_with_precomputed_embeddings(
    fasta_path: Path,
    embeddings_path: Path,
    output_path: Path,
) -> None:
    """Make predictions using pre-computed ESM-2 embeddings.

    The embeddings file should be a .npy file containing an array of shape
    (n_sequences, embedding_dim) where embedding_dim is 1280 for ESM-2 650M.
    """
    logger.info("Mode: Pre-computed embeddings")

    # Load sequences
    seq_ids, sequences = load_sequences(fasta_path)
    logger.info("Loaded %d sequences from %s", len(sequences), fasta_path)

    # Load pre-computed embeddings
    logger.info("Loading pre-computed embeddings from %s", embeddings_path)
    embeddings = np.load(embeddings_path)
    logger.info("Embeddings shape: %s", embeddings.shape)

    if len(embeddings) != len(sequences):
        raise ValueError(
            f"Mismatch: {len(sequences)} sequences but {len(embeddings)} embeddings"
        )

    # Extract physicochemical features
    logger.info("Extracting physicochemical features...")
    phys = extract_physicochemical_features(sequences)
    X = np.hstack([embeddings, phys])
    logger.info("Combined feature shape: %s", X.shape)

    # Load model
    model_path = Path("outputs/models/best_model.joblib")
    logger.info("Loading model from %s", model_path)
    artefact = joblib.load(model_path)
    model = artefact["model"]
    scaler = artefact["scaler"]
    thresholds = artefact.get("thresholds")

    # Scale features
    X = scaler.transform(X)

    # Predict
    logger.info("Making predictions...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        y_proba = model.predict_proba(X)

    # Apply thresholds if available
    if thresholds is not None:
        adjusted = y_proba / np.array(thresholds)[np.newaxis, :]
        y_pred = adjusted.argmax(axis=1)
        logger.info("Applied per-class thresholds: %s", np.round(thresholds, 3))
    else:
        y_pred = y_proba.argmax(axis=1)

    # Assign confidence
    confidence_levels = assign_confidence(y_proba)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seq_id, pred, conf in zip(seq_ids, y_pred, confidence_levels):
            f.write(f"{seq_id} {pred} Confidence {conf}\n")

    logger.info("Predictions written to %s", output_path)

    # Summary
    from collections import Counter
    pred_counts = Counter(int(p) for p in y_pred)
    conf_counts = Counter(confidence_levels)
    print(f"\nClass distribution: {dict(sorted(pred_counts.items()))}")
    print(f"Confidence distribution: {dict(sorted(conf_counts.items()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline blind test predictions (workarounds for missing ESM-2 weights)"
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("blind_ec_test.fasta.txt"),
        help="Input FASTA file",
    )
    parser.add_argument(
        "--mode",
        choices=["handcrafted", "precomputed"],
        required=True,
        help="Prediction mode: handcrafted (no ESM-2) or precomputed (use saved embeddings)",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="Path to pre-computed embeddings .npy file (required for precomputed mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions/blind_predictions_offline.txt"),
        help="Output predictions file",
    )

    args = parser.parse_args()

    if args.mode == "precomputed":
        if args.embeddings is None:
            parser.error("--embeddings is required for precomputed mode")
        if not args.embeddings.exists():
            parser.error(f"Embeddings file not found: {args.embeddings}")
        predict_with_precomputed_embeddings(args.fasta, args.embeddings, args.output)
    else:
        predict_with_handcrafted_only(args.fasta, args.output)


if __name__ == "__main__":
    main()
