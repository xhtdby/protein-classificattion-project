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
        feature_source: One of "Handcrafted", "ESM-2", "Handcrafted + ESM-2",
                        "ESM-2 + Physicochemical".

    Returns:
        Feature matrix (N, D).
    """
    parts = []
    source_lower = feature_source.lower()

    needs_handcrafted = "handcrafted" in source_lower
    needs_esm = "esm" in source_lower
    needs_physico = "physicochemical" in source_lower

    if needs_handcrafted:
        comp = extract_composition_features(sequences)
        phys = extract_physicochemical_features(sequences)
        parts.append(np.hstack([comp, phys]))

    if needs_esm:
        from src.features.embeddings import extract_esm2_embeddings
        emb = extract_esm2_embeddings(sequences, model_name=model_name)
        parts.append(emb)

    if needs_physico and not needs_handcrafted:
        # Physicochemical only (without full handcrafted composition)
        phys = extract_physicochemical_features(sequences)
        parts.append(phys)

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
    esm_model_name = artefact.get("esm_model_name", "esm2_t6_8M_UR50D")
    expected_dim = artefact.get("esm_embedding_dim")
    thresholds = artefact.get("thresholds")  # per-class decision thresholds

    logger.info("Loaded model from %s (feature source: %s, ESM: %s, expected_dim: %s)",
                model_path, feature_source, esm_model_name, expected_dim)

    # Parse FASTA
    seq_ids, sequences = load_fasta_sequences(fasta_path)
    logger.info("Parsed %d sequences from %s", len(sequences), fasta_path)

    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {fasta_path}")

    # Extract features
    X = extract_features(sequences, feature_source, model_name=esm_model_name)
    if X.shape[1] != scaler.n_features_in_:
        raise ValueError(
            f"Feature dimension mismatch: extracted {X.shape[1]}, "
            f"model expects {scaler.n_features_in_}"
        )
    X = scaler.transform(X)

    # Predict -- suppress sklearn's feature-names warning for LightGBM
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
        y_proba = (
            model.predict_proba(X)
            if hasattr(model, "predict_proba")
            else np.zeros((len(X), 7))
        )
        # Apply per-class thresholds if available
        if thresholds is not None:
            adjusted = y_proba / thresholds[np.newaxis, :]
            y_pred = adjusted.argmax(axis=1)
            logger.info("Applied per-class thresholds: %s", np.round(thresholds, 3))
        else:
            y_pred = y_proba.argmax(axis=1)

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
