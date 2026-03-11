"""
Physicochemical property features derived from amino acid sequences.

Features per sequence:
- Molecular weight
- Isoelectric point (pI)
- Aromaticity (fraction of F + W + Y)
- GRAVY (Grand Average of Hydropathicity, Kyte-Doolittle)
- Net charge at pH 7.0
- Secondary structure propensity fractions (helix, turn, sheet)
"""

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm import tqdm

# Kyte-Doolittle hydropathicity scale
KD_SCALE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

FEATURE_NAMES = [
    "molecular_weight",
    "isoelectric_point",
    "aromaticity",
    "gravy",
    "charge_at_ph7",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
]


def _clean_sequence(seq: str) -> str:
    """Remove non-standard amino acids (X, B, Z, U, O, J) for ProtParam compatibility."""
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in seq.upper() if aa in standard)


def physicochemical_features(seq: str) -> np.ndarray:
    """Compute physicochemical feature vector for a single sequence.

    Returns 8-dim vector.
    """
    cleaned = _clean_sequence(seq)
    if len(cleaned) < 2:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    pa = ProteinAnalysis(cleaned)

    # Molecular weight
    mw = pa.molecular_weight()

    # Isoelectric point
    pi = pa.isoelectric_point()

    # Aromaticity
    arom = pa.aromaticity()

    # GRAVY
    gravy = pa.gravy()

    # Charge at pH 7
    charge = pa.charge_at_pH(7.0)

    # Secondary structure fractions (helix, turn, sheet) from Chou-Fasman
    ss = pa.secondary_structure_fraction()
    helix_frac = ss[0]
    turn_frac = ss[1]
    sheet_frac = ss[2]

    return np.array(
        [mw, pi, arom, gravy, charge, helix_frac, turn_frac, sheet_frac],
        dtype=np.float64,
    )


def extract_physicochemical_features(sequences: list[str]) -> np.ndarray:
    """Batch extraction of physicochemical features.

    Returns array of shape (N, 8).
    """
    n = len(sequences)
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    for i, seq in enumerate(tqdm(sequences, desc="Physicochemical features")):
        features[i] = physicochemical_features(seq)
    return features


def get_feature_names() -> list[str]:
    """Return ordered list of feature names."""
    return list(FEATURE_NAMES)


if __name__ == "__main__":
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHIKK"
    feats = physicochemical_features(test_seq)
    print(f"Test sequence length: {len(test_seq)}")
    for name, val in zip(FEATURE_NAMES, feats):
        print(f"  {name}: {val:.4f}")
