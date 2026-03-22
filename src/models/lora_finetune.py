"""
LoRA (Low-Rank Adaptation) fine-tuning of ESM-2 650M for enzyme classification.

Instead of full fine-tuning of 650M parameters (which requires >24 GB VRAM),
LoRA injects small rank-decomposition matrices into the attention blocks.
The base transformer weights are frozen; only the LoRA matrices and the
classification head are trained — reducing trainable parameters by ~99%.

Architecture:
    ESM-2 650M (esm2_t33_650M_UR50D, frozen base weights)
    + LoRA adapters on Q and V projections (rank=8, alpha=16)
    -> mean-pool over sequence positions
    -> Linear(1280, 256) -> GELU -> Dropout(0.3) -> Linear(256, 7)

Training:
    - Stratified 5-fold CV (same indices as rest of pipeline)
    - AdamW: LoRA lr=2e-4, classifier head lr=1e-3
    - OneCycleLR (10% warmup + cosine decay)
    - Focal Loss (gamma=2.0, per-class alpha from train fold)
    - bfloat16 mixed precision
    - Gradient clipping max_norm=1.0
    - Early stopping on val Macro F1, patience=5

Usage:
    python -m src.models.lora_finetune
    python -m src.models.lora_finetune --epochs 15 --batch-size 2
    python -m src.models.lora_finetune --lora-rank 16 --lora-alpha 32
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)

from src.data_loading import load_all_sequences, get_cv_splits, SEED, CLASS_NAMES
from src.losses import FocalLoss, compute_focal_alpha
from src.training import fmt_time
from src.evaluation import plot_confusion_matrix, save_results_json

logger = logging.getLogger(__name__)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MAX_LEN = 512
N_CLASSES = 7
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
EMB_DIM = 1280


# ---------------------------------------------------------------------------
# LoRA layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around a frozen nn.Linear.

    Adds a trainable low-rank decomposition:  ΔW = B @ A  where
    A ∈ R^{r×in}, B ∈ R^{out×r}.  The original linear weights are frozen.

    Parameters
    ----------
    original : nn.Linear — the frozen linear layer to adapt.
    rank : int — rank of the decomposition (r).
    alpha : float — scaling factor (alpha/rank).
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Low-rank decomposition: W' = W + scaling * B @ A
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming initialisation for A, zero for B (so ΔW starts at 0)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


def inject_lora(
    esm_model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> int:
    """Inject LoRA adapters into ESM-2 attention layers.

    Replaces the specified projection layers with LoRALinear wrappers.
    Returns the number of LoRA parameters added.
    """
    n_lora_params = 0

    for name, module in esm_model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                    setattr(module, target, lora_layer)
                    n_lora_params += (
                        lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
                    )

    return n_lora_params


def freeze_base_model(esm_model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters."""
    for name, param in esm_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int], max_len: int = MAX_LEN):
        self.sequences = [s[:max_len] for s in sequences]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.sequences[idx], self.labels[idx]


def make_collate_fn(batch_converter):
    def collate(batch):
        seqs, labels = zip(*batch)
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return tokens, torch.tensor(labels, dtype=torch.long), seq_lengths
    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LoRAESM2Classifier(nn.Module):
    """ESM-2 650M with LoRA adapters + classification head."""

    def __init__(
        self,
        esm_model,
        emb_dim: int = EMB_DIM,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.esm = esm_model
        self.n_layers = esm_model.num_layers
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, tokens: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        out = self.esm(tokens, repr_layers=[self.n_layers])
        reps = out["representations"][self.n_layers]  # (B, L+2, D)
        B, L, D = reps.shape
        # Mean-pool positions 1..seq_len
        pos = torch.arange(L, device=reps.device).unsqueeze(0)
        mask = (pos >= 1) & (pos <= seq_lengths.unsqueeze(1))
        mask_f = mask.unsqueeze(-1).float()
        pooled = (reps * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Sklearn-style predictor for predict_blind.py
# ---------------------------------------------------------------------------

class LoRAPredictor:
    """Wraps a saved LoRA checkpoint for sklearn-style predict_proba."""

    def __init__(
        self,
        model_path: Path | str,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        self.model_path = Path(model_path)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def predict_proba(self, sequences: list[str], batch_size: int = 2) -> np.ndarray:
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if (hasattr(torch.backends, "mps")
                                               and torch.backends.mps.is_available())
                  else torch.device("cpu"))
        import esm as esm_lib

        esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
        inject_lora(esm_model, rank=self.lora_rank, alpha=self.lora_alpha)
        freeze_base_model(esm_model)
        model = LoRAESM2Classifier(esm_model).to(device)
        state = torch.load(self.model_path, map_location=device, weights_only=True)
        # Load only LoRA + classifier weights (base model is frozen)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)
        model.eval()

        batch_converter = alphabet.get_batch_converter()
        all_proba: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch_seqs = sequences[start:start + batch_size]
                data = [(f"s{i}", s[:MAX_LEN]) for i, s in enumerate(batch_seqs)]
                _, _, tokens = batch_converter(data)
                seq_lengths = torch.tensor(
                    [len(s[:MAX_LEN]) for s in batch_seqs], dtype=torch.long,
                )
                tokens = tokens.to(device)
                seq_lengths = seq_lengths.to(device)
                logits = model(tokens, seq_lengths)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                all_proba.append(proba)

        del model, esm_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return np.vstack(all_proba)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }


def _run_epoch(
    model: LoRAESM2Classifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    amp_scaler: torch.amp.GradScaler,
    device: torch.device,
    train: bool = True,
    scheduler=None,
    grad_accum: int = 1,
) -> dict:
    model.train(train)
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    amp_enabled = device.type == "cuda"
    n_batches = len(loader)

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        for step, (tokens, labels, seq_lengths) in enumerate(loader):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            is_last = (step == n_batches - 1)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                    enabled=amp_enabled):
                logits = model(tokens, seq_lengths)
                loss = criterion(logits, labels)
                if train:
                    loss = loss / grad_accum

            if train and optimizer is not None:
                amp_scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or is_last:
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            logged_loss = loss.item() * (grad_accum if train else 1)
            total_loss += logged_loss * len(labels)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return {"loss": total_loss / len(y_true), **_compute_metrics(y_true, y_pred)}


def _collect_oof(model, loader, device):
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for tokens, _, seq_lengths in loader:
            tokens = tokens.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            logits = model(tokens, seq_lengths)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(proba.argmax(axis=1))
            all_proba.append(proba)
    return np.concatenate(all_preds), np.vstack(all_proba)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_lora(
    sequences: list[str],
    y: np.ndarray,
    cv_splits: list,
    *,
    epochs: int = 10,
    batch_size: int = 2,
    grad_accum: int = 8,
    lr_lora: float = 2e-4,
    lr_head: float = 1e-3,
    patience: int = 5,
    max_len: int = MAX_LEN,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    focal_gamma: float = 2.0,
    device: torch.device,
    figures_dir: Path,
    checkpoints_dir: Path,
) -> dict:
    """5-fold CV with LoRA fine-tuning of ESM-2 650M."""
    import esm as esm_lib

    fold_results: list[dict] = []
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    print(f"\n{'='*72}")
    print(f"  LoRA FINE-TUNING ESM-2 650M -- {len(cv_splits)}-fold CV")
    print(f"  epochs={epochs}  batch={batch_size}  grad_accum={grad_accum}  "
          f"(effective_batch={batch_size*grad_accum})")
    print(f"  lr_lora={lr_lora:.0e}  lr_head={lr_head:.0e}  "
          f"rank={lora_rank}  alpha={lora_alpha}  focal_gamma={focal_gamma}")
    print(f"{'='*72}")

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n{'-'*72}")
        print(f"  FOLD {fold_i+1}/{len(cv_splits)}  "
              f"train={len(train_idx):,}  val={len(val_idx):,}")
        print(f"{'-'*72}")
        t_fold = time.time()

        # Fresh model per fold
        esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()

        # Inject LoRA and freeze base
        n_lora = inject_lora(esm_model, rank=lora_rank, alpha=lora_alpha)
        freeze_base_model(esm_model)

        model = LoRAESM2Classifier(esm_model).to(device)
        batch_converter = alphabet.get_batch_converter()
        collate = make_collate_fn(batch_converter)

        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  LoRA params: {n_lora:,}  |  Classifier params: {trainable - n_lora:,}  |  "
              f"Total trainable: {trainable:,} / {total_params:,} "
              f"({100*trainable/total_params:.2f}%)")

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        train_loader = DataLoader(
            ProteinDataset(train_seqs, y_train.tolist(), max_len=max_len),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            ProteinDataset(val_seqs, y_val.tolist(), max_len=max_len),
            batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )

        # Focal loss with per-class alpha from train fold only
        alpha = compute_focal_alpha(y_train, n_classes=N_CLASSES).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)

        # Separate parameter groups: LoRA params + classifier head
        lora_params = [p for n, p in model.esm.named_parameters() if p.requires_grad]
        head_params = list(model.classifier.parameters())

        optimizer = AdamW(
            [
                {"params": lora_params, "lr": lr_lora},
                {"params": head_params, "lr": lr_head},
            ],
            weight_decay=0.01,
        )

        accum_steps = (len(train_loader) + grad_accum - 1) // grad_accum
        total_steps = epochs * accum_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr_lora, lr_head],
            total_steps=total_steps,
            pct_start=0.1,
        )
        grad_scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_val_f1 = -1.0
        patience_ctr = 0
        ckpt_path = checkpoints_dir / f"lora_fold{fold_i+1}.pt"
        best_oof_preds = None
        best_oof_proba = None

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            tr = _run_epoch(model, train_loader, criterion, optimizer,
                            grad_scaler, device, train=True, scheduler=scheduler,
                            grad_accum=grad_accum)
            vl = _run_epoch(model, val_loader, criterion, None,
                            grad_scaler, device, train=False)

            improved = vl["macro_f1"] > best_val_f1
            marker = " * NEW BEST" if improved else ""
            print(
                f"  Epoch {epoch:>2}/{epochs}  "
                f"loss={tr['loss']:.4f}/{vl['loss']:.4f}  "
                f"F1={vl['macro_f1']:.4f}  BA={vl['balanced_accuracy']:.4f}  "
                f"[{fmt_time(time.time()-t_ep)}]{marker}",
                flush=True,
            )

            if improved:
                best_val_f1 = vl["macro_f1"]
                patience_ctr = 0
                # Save only trainable weights (LoRA + classifier)
                state = {k: v for k, v in model.state_dict().items()
                         if any(t in k for t in ("lora_", "classifier"))}
                torch.save(state, ckpt_path)
                best_oof_preds, best_oof_proba = _collect_oof(model, val_loader, device)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stop (patience={patience})")
                    break

        oof_preds[val_idx] = best_oof_preds
        oof_proba[val_idx] = best_oof_proba

        fold_m = _compute_metrics(y_val, best_oof_preds)
        fold_results.append(fold_m)
        print(
            f"\n  Fold {fold_i+1} done  F1={fold_m['macro_f1']:.4f}  "
            f"BA={fold_m['balanced_accuracy']:.4f}  MCC={fold_m['mcc']:.4f}  "
            f"[{fmt_time(time.time()-t_fold)}]"
        )

        del model, esm_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    keys = list(fold_results[0].keys())
    summary = {k + "_mean": float(np.mean([f[k] for f in fold_results])) for k in keys}
    summary.update({k + "_std": float(np.std([f[k] for f in fold_results])) for k in keys})

    return {
        "model_name": "LoRA-ESM2-650M",
        "fold_results": fold_results,
        "summary": summary,
        "oof_preds": oof_preds,
        "oof_true": y,
        "oof_proba": oof_proba,
    }


# ---------------------------------------------------------------------------
# Full retrain
# ---------------------------------------------------------------------------

def retrain_full_lora(
    sequences: list[str],
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    grad_accum: int = 8,
    lr_lora: float = 2e-4,
    lr_head: float = 1e-3,
    max_len: int = MAX_LEN,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    focal_gamma: float = 2.0,
    device: torch.device,
    save_path: Path,
) -> None:
    """Retrain LoRA model on full dataset and save checkpoint."""
    import esm as esm_lib

    print(f"\n{'='*72}")
    print(f"  FINAL RETRAIN (LoRA) on full dataset  "
          f"({len(sequences):,} sequences, {epochs} epochs)")
    print(f"{'='*72}")

    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    inject_lora(esm_model, rank=lora_rank, alpha=lora_alpha)
    freeze_base_model(esm_model)
    model = LoRAESM2Classifier(esm_model).to(device)
    collate = make_collate_fn(alphabet.get_batch_converter())

    loader = DataLoader(
        ProteinDataset(sequences, y.tolist(), max_len=max_len),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )

    alpha = compute_focal_alpha(y, n_classes=N_CLASSES).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)

    lora_params = [p for n, p in model.esm.named_parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": lr_lora},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=0.01,
    )

    accum_steps = (len(loader) + grad_accum - 1) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr_lora, lr_head],
        total_steps=epochs * accum_steps, pct_start=0.1,
    )
    grad_scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        t = time.time()
        m = _run_epoch(model, loader, criterion, optimizer,
                       grad_scaler, device, train=True, scheduler=scheduler,
                       grad_accum=grad_accum)
        print(f"  Epoch {epoch:>2}/{epochs}  loss={m['loss']:.4f}  "
              f"F1={m['macro_f1']:.4f}  [{fmt_time(time.time()-t)}]", flush=True)

    # Save full state dict (includes LoRA weights)
    torch.save(model.state_dict(), save_path)
    print(f"\n  Saved -> {save_path}")
    del model, esm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of ESM-2 650M for enzyme classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-GPU batch size (650M needs ~10 GB VRAM per sample)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr-lora", type=float, default=2e-4,
                        help="Learning rate for LoRA adapter weights")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Learning rate for classification head")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA decomposition rank (r)")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA scaling factor (alpha)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard CE)")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--retrain-epochs", type=int, default=None,
                        help="Epochs for final full retrain (default: same as --epochs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if (hasattr(torch.backends, "mps")
                                           and torch.backends.mps.is_available())
              else torch.device("cpu"))

    print(f"\n{'='*72}")
    print(f"  HARDWARE")
    print(f"{'='*72}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        print(f"  GPU    : {props.name}")
        print(f"  VRAM   : {total/1e9:.1f} GB  ({free/1e9:.1f} GB free)")
    print(f"{'='*72}")

    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = project_root / "outputs" / "figures"
    models_dir = project_root / "outputs" / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load sequences
    t0 = time.time()
    print("\nLoading sequences...")
    df = load_all_sequences(project_root)
    sequences = df["sequence"].tolist()
    y = df["label"].values
    cv_splits = get_cv_splits(y)
    print(f"  Loaded {len(df):,} sequences  [{fmt_time(time.time()-t0)}]")
    print(f"  Class distribution: {np.bincount(y).tolist()}")

    # Cross-validation
    t_cv = time.time()
    cv_result = cross_validate_lora(
        sequences, y, cv_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        patience=args.patience,
        max_len=args.max_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        focal_gamma=args.focal_gamma,
        device=device,
        figures_dir=figures_dir,
        checkpoints_dir=models_dir,
    )

    summary = cv_result["summary"]
    print(f"\n{'='*72}")
    print(f"  LoRA ESM-2 650M -- 5-Fold CV Summary")
    print(f"{'='*72}")
    print(f"  Accuracy:          {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Macro F1:          {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"  Balanced Accuracy: {summary['balanced_accuracy_mean']:.4f} +/- {summary['balanced_accuracy_std']:.4f}")
    print(f"  MCC:               {summary['mcc_mean']:.4f} +/- {summary['mcc_std']:.4f}")
    print(f"{'='*72}")

    print("\n--- OOF Classification Report ---")
    print(classification_report(
        cv_result["oof_true"], cv_result["oof_preds"],
        target_names=CLASS_NAMES, zero_division=0,
    ))

    plot_confusion_matrix(
        cv_result["oof_true"], cv_result["oof_preds"],
        save_path=figures_dir / "cm_lora_finetune.png",
        title="LoRA ESM-2 650M -- OOF Confusion Matrix",
    )

    # Save results JSON
    results_data = dict(cv_result)
    results_data["config"] = {
        "esm_model": ESM_MODEL_NAME,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_lora": args.lr_lora,
        "lr_head": args.lr_head,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "focal_gamma": args.focal_gamma,
    }
    save_results_json(results_data, project_root / "outputs" / "lora_finetune_results.json")

    # Final retrain
    retrain_ep = args.retrain_epochs or args.epochs
    final_path = models_dir / "lora_final.pt"
    retrain_full_lora(
        sequences, y,
        epochs=retrain_ep,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        max_len=args.max_len,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        focal_gamma=args.focal_gamma,
        device=device,
        save_path=final_path,
    )

    # Save joblib artifact for predict_blind.py
    predictor = LoRAPredictor(
        final_path, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    )
    artifact = {
        "model": predictor,
        "scaler": None,
        "feature_source": "lora_finetune",
        "esm_model_name": ESM_MODEL_NAME,
        "esm_embedding_dim": EMB_DIM,
        "cv_scores": summary,
        "model_name": "LoRA-ESM2-650M",
        "lora_config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
        },
    }
    artifact_path = models_dir / "lora_artifact.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Joblib artifact saved -> {artifact_path}")

    print(f"\nTotal wall time: {fmt_time(time.time()-t0)}")
