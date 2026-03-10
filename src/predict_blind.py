"""
Blind challenge prediction pipeline.

Loads a saved model, parses an input FASTA file, extracts features,
predicts enzyme class with confidence, and writes output in the exact
required format:

    SEQ01 1 Confidence High
    SEQ02 0 Confidence Medium

Usage:
    python -m src.predict_blind --fasta <path> --model <path> --output <path>
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import joblib
from Bio import SeqIO

from src.features.composition import extract_composition_features
from src.features.physicochemical import extract_physicochemical_features
from src.confidence import assign_confidence

logger = logging.getLogger(__name__)


def load_fasta_sequences(fasta_path: Path) -> tuple[list[str], list[str]]:
    """Parse FASTA file and return (seq_ids, sequences)."""
    seq_ids = []
    sequences = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq_ids.append(record.id)
        sequences.append(str(record.seq))
    return seq_ids, sequences


def extract_features(
    sequences: list[str],
    feature_source: str,
    model_name: str = "esm2_t6_8M_UR50D",
) -> np.ndarray:
    """Extract features matching the training feature source.

    Args:
        sequences: List of amino acid strings.
        feature_source: One of "Handcrafted", "ESM-2", "Handcrafted + ESM-2".

    Returns:
        Feature matrix (N, D).
    """
    parts = []

    if "Handcrafted" in feature_source or "handcrafted" in feature_source.lower():
        comp = extract_composition_features(sequences)
        phys = extract_physicochemical_features(sequences)
        parts.append(np.hstack([comp, phys]))

    if "ESM" in feature_source or "esm" in feature_source.lower():
        from src.features.embeddings import extract_esm2_embeddings
        emb = extract_esm2_embeddings(sequences, model_name=model_name)
        parts.append(emb)

    if not parts:
        raise ValueError(f"Unknown feature source: {feature_source}")

    return np.hstack(parts) if len(parts) > 1 else parts[0]


def predict_blind(
    fasta_path: Path,
    model_path: Path,
    output_path: Path,
) -> None:
    """Full blind prediction pipeline.

    Args:
        fasta_path: Path to input FASTA file.
        model_path: Path to saved model (.joblib).
        output_path: Path for output predictions file.
    """
    # Load model artefact
    artefact = joblib.load(model_path)
    model = artefact["model"]
    scaler = artefact["scaler"]
    feature_source = artefact.get("feature_source", "Handcrafted")

    logger.info("Loaded model from %s (feature source: %s)", model_path, feature_source)

    # Parse FASTA
    seq_ids, sequences = load_fasta_sequences(fasta_path)
    logger.info("Parsed %d sequences from %s", len(sequences), fasta_path)

    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {fasta_path}")

    # Extract features
    X = extract_features(sequences, feature_source)
    X = scaler.transform(X)

    # Predict -- suppress sklearn's feature-names warning for LightGBM
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        y_pred = model.predict(X)
        y_proba = (
            model.predict_proba(X)
            if hasattr(model, "predict_proba")
            else np.zeros((len(X), 7))
        )

    # Confidence
    confidence_levels = assign_confidence(y_proba)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for seq_id, pred, conf in zip(seq_ids, y_pred, confidence_levels):
            f.write(f"{seq_id} {pred} Confidence {conf}\n")

    logger.info("Predictions written to %s", output_path)
    print(f"Wrote {len(seq_ids)} predictions to {output_path}")

    # Summary
    from collections import Counter
    pred_counts = Counter(int(p) for p in y_pred)
    conf_counts = Counter(confidence_levels)
    print(f"Class distribution: {dict(sorted(pred_counts.items()))}")
    print(f"Confidence distribution: {dict(sorted(conf_counts.items()))}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Blind challenge predictions")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file")
    parser.add_argument(
        "--model", type=Path, default=Path("outputs/models/best_model.joblib"),
        help="Path to saved model",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/predictions/blind_predictions.txt"),
        help="Output predictions file",
    )
    args = parser.parse_args()

    predict_blind(args.fasta, args.model, args.output)
