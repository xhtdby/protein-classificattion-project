"""
Data loading and cross-validation split utilities.

Parses FASTA files from the workspace root via Bio.SeqIO,
assigns integer labels (0-6), and provides stratified K-fold splits.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Mapping from filename stem to integer label
FASTA_LABEL_MAP: dict[str, int] = {
    "class0_rep_seq.fasta.txt": 0,
    "ec1_rep_seq.fasta.txt": 1,
    "ec2_rep_seq.fasta.txt": 2,
    "ec3_rep_seq.fasta.txt": 3,
    "ec4_rep_seq.fasta.txt": 4,
    "ec5_rep_seq.fasta.txt": 5,
    "ec6_rep_seq.fasta.txt": 6,
}

CLASS_NAMES: list[str] = [
    "Not enzyme",
    "Oxidoreductase",
    "Transferase",
    "Hydrolase",
    "Lyase",
    "Isomerase",
    "Ligase",
]


def load_all_sequences(root: Path) -> pd.DataFrame:
    """Load all 7 FASTA files and return a DataFrame.

    Columns: seq_id, sequence, label, length
    """
    records: list[dict] = []
    for fname, label in FASTA_LABEL_MAP.items():
        fpath = root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Expected FASTA file not found: {fpath}")
        for record in SeqIO.parse(str(fpath), "fasta"):
            seq_str = str(record.seq)
            records.append(
                {
                    "seq_id": record.id,
                    "sequence": seq_str,
                    "label": label,
                    "length": len(seq_str),
                }
            )
    df = pd.DataFrame(records)
    return df


def get_cv_splits(
    labels: np.ndarray, n_splits: int = 5, seed: int = SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified K-fold train/val index pairs."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def print_class_distribution(df: pd.DataFrame) -> None:
    """Print class distribution summary."""
    print("\n=== Class Distribution ===")
    total = len(df)
    for label in sorted(df["label"].unique()):
        count = (df["label"] == label).sum()
        pct = 100.0 * count / total
        name = CLASS_NAMES[label]
        print(f"  Class {label} ({name:>16s}): {count:>6d}  ({pct:5.1f}%)")
    print(f"  {'Total':>28s}: {total:>6d}")


if __name__ == "__main__":
    # Resolve project root (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    print(f"Loading sequences from: {project_root}")

    df = load_all_sequences(project_root)
    print_class_distribution(df)

    # Verify CV splits preserve class proportions
    splits = get_cv_splits(df["label"].values)
    print(f"\n=== Stratified {len(splits)}-Fold Split Verification ===")
    for i, (train_idx, val_idx) in enumerate(splits):
        train_dist = np.bincount(df["label"].values[train_idx], minlength=7)
        val_dist = np.bincount(df["label"].values[val_idx], minlength=7)
        print(f"  Fold {i}: train={train_dist.tolist()}, val={val_dist.tolist()}")
