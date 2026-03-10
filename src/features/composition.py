"""
Amino acid composition and dipeptide frequency features.

- amino_acid_composition: 20-dim normalised frequency vector
- dipeptide_frequencies: 400-dim vector (20×20 canonical pairs)
- extract_composition_features: batch wrapper -> (N, 421) array
"""

import numpy as np
from tqdm import tqdm

# Canonical amino acids in alphabetical order
AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
N_AA = len(AMINO_ACIDS)  # 20

# Dipeptide pairs
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
DI_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}
N_DI = len(DIPEPTIDES)  # 400


def amino_acid_composition(seq: str) -> np.ndarray:
    """Compute normalised 20-dim amino acid frequency vector."""
    counts = np.zeros(N_AA, dtype=np.float64)
    for aa in seq.upper():
        idx = AA_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def dipeptide_frequencies(seq: str) -> np.ndarray:
    """Compute normalised 400-dim dipeptide frequency vector."""
    counts = np.zeros(N_DI, dtype=np.float64)
    seq_upper = seq.upper()
    for i in range(len(seq_upper) - 1):
        dp = seq_upper[i : i + 2]
        idx = DI_INDEX.get(dp)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def extract_composition_features(sequences: list[str]) -> np.ndarray:
    """Batch extraction: AA composition (20) + dipeptide (400) + length (1) = 421-dim.

    Returns array of shape (N, 421).
    """
    n = len(sequences)
    features = np.zeros((n, N_AA + N_DI + 1), dtype=np.float64)
    for i, seq in enumerate(tqdm(sequences, desc="Composition features")):
        features[i, :N_AA] = amino_acid_composition(seq)
        features[i, N_AA : N_AA + N_DI] = dipeptide_frequencies(seq)
        features[i, -1] = len(seq)
    return features


def get_feature_names() -> list[str]:
    """Return ordered list of feature names matching extract_composition_features columns."""
    return (
        [f"AA_{aa}" for aa in AMINO_ACIDS]
        + [f"DI_{dp}" for dp in DIPEPTIDES]
        + ["seq_length"]
    )


if __name__ == "__main__":
    # Quick sanity check
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHIKK"
    aac = amino_acid_composition(test_seq)
    dpf = dipeptide_frequencies(test_seq)
    print(f"Test sequence length: {len(test_seq)}")
    print(f"AA composition shape: {aac.shape}, sum: {aac.sum():.4f}")
    print(f"Dipeptide freq shape: {dpf.shape}, sum: {dpf.sum():.4f}")
    print(f"Top 5 AAs: {', '.join(AMINO_ACIDS[i] for i in np.argsort(aac)[-5:][::-1])}")
