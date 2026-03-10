"""
Protein language model (ESM-2) embedding extraction.

Pre-computes mean-pooled embeddings for all sequences and caches to disk.
Supports GPU -> MPS -> CPU fallback.
"""

import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

MODEL_REGISTRY = {
    "esm2_t6_8M_UR50D": {"dim": 320, "max_len": 1022},
    "esm2_t33_650M_UR50D": {"dim": 1280, "max_len": 1022},
}


def _get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_device_info(device: torch.device) -> None:
    print(f"\n  Device        : {device}")
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        print(f"  GPU           : {props.name}")
        print(f"  VRAM total    : {total / 1e9:.1f} GB")
        print(f"  VRAM free     : {free / 1e9:.1f} GB")
    elif device.type == "mps":
        print("  GPU           : Apple MPS")
    else:
        print("  (No GPU -- running on CPU, this will be slow)")


def extract_esm2_embeddings(
    sequences: list[str],
    seq_ids: list[str] | None = None,
    model_name: str = "esm2_t6_8M_UR50D",
    batch_size: int = 32,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract mean-pooled ESM-2 embeddings for a list of protein sequences.

    Returns ndarray of shape (N, embedding_dim).
    """
    import esm  # deferred -- only needed here

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")

    info = MODEL_REGISTRY[model_name]
    emb_dim = info["dim"]
    max_len = info["max_len"]

    if device is None:
        device = _get_device()

    print(f"\n  Loading ESM-2 model: {model_name}  (dim={emb_dim})")
    _print_device_info(device)

    t_load = time.time()
    model_loader = getattr(esm.pretrained, model_name)
    model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time() - t_load:.1f}s\n")

    n = len(sequences)
    embeddings = np.zeros((n, emb_dim), dtype=np.float32)
    n_truncated = 0
    n_batches = (n + batch_size - 1) // batch_size
    t_start = time.time()

    with tqdm(total=n, desc=f"ESM-2 embeddings", unit="seq",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for b_idx, start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            batch_data = []
            for i in range(start, end):
                sid = seq_ids[i] if seq_ids else f"seq_{i}"
                seq = sequences[i][:max_len]
                if len(sequences[i]) > max_len:
                    n_truncated += 1
                batch_data.append((sid, seq))

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[model.num_layers])
            token_reps = results["representations"][model.num_layers]

            for j in range(token_reps.size(0)):
                seq_len = len(batch_data[j][1])
                if seq_len == 0:
                    # Empty sequence -> zero vector (avoids NaN from mean of empty slice)
                    embeddings[start + j] = 0.0
                else:
                    rep = token_reps[j, 1: seq_len + 1, :]
                    embeddings[start + j] = rep.mean(dim=0).cpu().numpy()

            pbar.update(end - start)

            # Log GPU memory every 10 batches
            if device.type == "cuda" and (b_idx + 1) % 10 == 0:
                free, total = torch.cuda.mem_get_info()
                used_gb = (total - free) / 1e9
                pbar.set_postfix({"VRAM_used": f"{used_gb:.1f}GB"}, refresh=False)

    elapsed = time.time() - t_start
    seqs_per_sec = n / elapsed
    print(f"\n  Embedded {n:,} sequences in {elapsed:.1f}s  ({seqs_per_sec:.0f} seq/s)")
    if n_truncated:
        print(f"  WARNING: {n_truncated} sequences truncated to {max_len} tokens")

    return embeddings


def load_or_compute_embeddings(
    sequences: list[str],
    seq_ids: list[str] | None = None,
    cache_path: Path | None = None,
    model_name: str = "esm2_t6_8M_UR50D",
    batch_size: int = 32,
) -> np.ndarray:
    """Load cached embeddings or compute and cache them."""
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        if data.shape[0] == len(sequences):
            print(f"  Loaded cached embeddings from {cache_path}  shape={data.shape}")
            return data
        logger.warning("Cache shape mismatch (%d vs %d) -- recomputing", data.shape[0], len(sequences))

    embeddings = extract_esm2_embeddings(
        sequences, seq_ids=seq_ids, model_name=model_name, batch_size=batch_size
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"  Saved embeddings -> {cache_path}  shape={embeddings.shape}")

    return embeddings


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    from src.data_loading import load_all_sequences
    from src.training import _fmt_time

    device = _get_device()
    print("=" * 72)
    print("  ESM-2 EMBEDDING EXTRACTION")
    print("=" * 72)
    _print_device_info(device)

    project_root = Path(__file__).resolve().parent.parent.parent
    features_dir = project_root / "outputs" / "features"
    cache_path = features_dir / "esm2_embeddings.npy"

    t0 = time.time()
    print("\nLoading sequences...")
    df = load_all_sequences(project_root)
    sequences = df["sequence"].tolist()
    seq_ids = df["seq_id"].tolist()
    print(f"  Loaded {len(sequences):,} sequences  [{_fmt_time(time.time() - t0)}]")

    # Auto batch size: smaller on CPU to avoid memory issues; conservative on GPU
    batch_size = 16 if device.type == "cuda" else 8

    try:
        embeddings = load_or_compute_embeddings(
            sequences=sequences,
            seq_ids=seq_ids,
            cache_path=cache_path,
            model_name="esm2_t6_8M_UR50D",
            batch_size=batch_size,
        )
        print(f"\nEmbedding matrix: {embeddings.shape}")
        print(f"Sample norms (first 5): {np.linalg.norm(embeddings[:5], axis=1).round(3)}")
        print(f"\nTotal wall time: {_fmt_time(time.time() - t0)}")
    except ImportError:
        print("\nERROR: fair-esm not installed. Run:  pip install fair-esm", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\nERROR during ESM-2 extraction:")
        traceback.print_exc()
        sys.exit(1)

