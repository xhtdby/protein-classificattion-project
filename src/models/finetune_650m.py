"""
ESM-2 650M end-to-end fine-tuning for 7-class enzyme classification.

Architecture
------------
    ESM-2 650M  (esm2_t33_650M_UR50D, 33 transformer layers, fully unfrozen)
    -> mean-pool over sequence positions (identical to embeddings.py)
    -> Linear(1280, 512) -> GELU -> Dropout(0.2)
    -> Linear(512, 128)  -> GELU -> Dropout(0.2)
    -> Linear(128, 7)

Training improvements over the 8M script
-----------------------------------------
- Layer-wise LR Decay (LLRD): outer transformer layers train at ``lr_backbone``;
  each layer inward is multiplied by ``llrd_decay`` (default 0.9).
- Larger classifier head captures more task-specific patterns.
- Full-length sequences (max_len=1022, same as ESM-2 pre-training) fit in 48 GB.
- Optional gradient checkpointing halves activation memory at ~20% compute cost.
- bfloat16 mixed precision throughout.
- Gradient clipping max_norm=1.0.
- AdamW weight_decay=0.01.
- OneCycleLR with 10% linear warmup + cosine decay.
- Weighted cross-entropy (balanced class weights fitted on train fold only).
- Early stopping on val Macro F1  patience=7.

Optimised for A40 (48 GB VRAM)
--------------------------------
Recommended launch:
    python -m src.models.finetune_650m \\
        --epochs 15 --batch-size 8 --grad-accum 2 --max-len 1022

That gives effective_batch=16 and should use ~20-28 GB VRAM in bf16.
Add --use-grad-ckpt to reduce to ~14-18 GB at some compute cost.

Usage
-----
    python -m src.models.finetune_650m
    python -m src.models.finetune_650m --epochs 15 --batch-size 8
    python -m src.models.finetune_650m --epochs 15 --batch-size 8 --use-grad-ckpt
    python -m src.models.finetune_650m --retrain-epochs 12
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
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
from sklearn.utils.class_weight import compute_class_weight

from src.data_loading import load_all_sequences, get_cv_splits, SEED, CLASS_NAMES
from src.training import fmt_time
from src.evaluation import plot_confusion_matrix, save_results_json

logger = logging.getLogger(__name__)

# -- Reproducibility -----------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -- Constants -----------------------------------------------------------------
MAX_LEN = 1022          # Full ESM-2 pre-training context length
N_CLASSES = 7
EMB_DIM = 1280          # esm2_t33_650M_UR50D hidden size
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"


# -- Dataset -------------------------------------------------------------------
class ProteinDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int],
                 max_len: int = MAX_LEN):
        self.sequences = [s[:max_len] for s in sequences]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.sequences[idx], self.labels[idx]


def make_collate_fn(batch_converter):
    """Return a collate function that tokenises sequences via ESM-2's alphabet."""
    def collate(batch: list[tuple[str, int]]):
        seqs, labels = zip(*batch)
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return tokens, torch.tensor(labels, dtype=torch.long), seq_lengths
    return collate


# -- Model ---------------------------------------------------------------------
class ESM2Classifier650M(nn.Module):
    """ESM-2 650M backbone + 3-layer classification head.

    The larger head (1280->512->128->7) gives more task-specific capacity than
    the 8M variant (320->128->7) while the 650M backbone's richer 1280-dim
    representations justify the extra depth.
    """

    def __init__(
        self,
        esm_model,
        emb_dim: int = EMB_DIM,
        n_classes: int = N_CLASSES,
        dropout: float = 0.2,
        use_grad_ckpt: bool = False,
    ):
        super().__init__()
        self.esm = esm_model
        self.n_layers = esm_model.num_layers
        self.use_grad_ckpt = use_grad_ckpt
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        if use_grad_ckpt:
            # Enable gradient checkpointing on every transformer block
            # to trade ~50% of activation memory for ~20% extra compute.
            for layer in self.esm.layers:
                layer.use_checkpoint = True  # handled in _forward_with_ckpt

    def _forward_esm(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Run ESM-2 forward pass, optionally with gradient checkpointing."""
        if self.use_grad_ckpt and self.training:
            x = self.esm.embed_tokens(tokens)
            # Apply positional embeddings if present (ESM-2 uses rotary, but the
            # embed_positions call is a no-op for rotary models).
            if hasattr(self.esm, "embed_positions"):
                x = x + self.esm.embed_positions(tokens)
            if hasattr(self.esm, "emb_layer_norm_before"):
                x = self.esm.emb_layer_norm_before(x)

            # Build padding mask (True where padded)
            padding_mask = tokens.eq(self.esm.padding_idx)
            if not padding_mask.any():
                padding_mask = None

            # Per-layer gradient checkpointing: trades ~50% activation memory
            # for ~20% extra compute.
            x_cur = x
            for layer in self.esm.layers:
                def make_ckpt(lyr):
                    def fn(x_in, pad_in):
                        out, _ = lyr(x_in, self_attn_padding_mask=pad_in,
                                     need_head_weights=False)
                        return out
                    return fn
                x_cur = torch.utils.checkpoint.checkpoint(
                    make_ckpt(layer), x_cur,
                    padding_mask if padding_mask is not None
                    else torch.zeros(tokens.shape[0], tokens.shape[1],
                                     dtype=torch.bool, device=tokens.device),
                    use_reentrant=False,
                )

            if hasattr(self.esm, "emb_layer_norm_after"):
                x_cur = self.esm.emb_layer_norm_after(x_cur)

            return x_cur  # (B, L, D)

        # Standard forward (no checkpointing)
        out = self.esm(tokens, repr_layers=[self.n_layers])
        return out["representations"][self.n_layers]  # (B, L, D)

    def forward(
        self, tokens: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        reps = self._forward_esm(tokens)           # (B, L, D)
        B, L, D = reps.shape

        # Mean-pool positions 1..seq_len (identical to embeddings.py)
        pos = torch.arange(L, device=reps.device).unsqueeze(0)   # (1, L)
        mask = (pos >= 1) & (pos <= seq_lengths.unsqueeze(1))     # (B, L)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (reps * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# -- LLRD optimizer ------------------------------------------------------------
def build_llrd_optimizer(
    model: ESM2Classifier650M,
    backbone_lr: float = 1e-5,
    head_lr: float = 5e-5,
    llrd_decay: float = 0.9,
    weight_decay: float = 0.01,
) -> AdamW:
    """Build AdamW with Layer-wise LR Decay.

    Learning rates:
        Classifier head     : ``head_lr``
        Outermost ESM layer : ``backbone_lr``
        Each deeper layer   : multiplied by ``llrd_decay``
        Embed tokens/pos    : ``backbone_lr * decay^n_layers``

    Decaying LR toward the embedding layer is well-established for fine-tuning
    large transformers (Sun et al., 2019; Howard & Ruder, 2018).
    """
    n = model.n_layers  # 33 for 650M
    param_groups: list[dict] = []

    # Classifier head — highest LR
    param_groups.append(
        {"params": list(model.classifier.parameters()), "lr": head_lr}
    )

    # Transformer layers (outermost = index n-1, innermost = index 0)
    for layer_idx in range(n - 1, -1, -1):
        depth = n - 1 - layer_idx                  # 0 for outermost layer
        lr = backbone_lr * (llrd_decay ** depth)
        param_groups.append(
            {
                "params": list(model.esm.layers[layer_idx].parameters()),
                "lr": lr,
            }
        )

    # Embedding parameters — lowest LR
    embed_params: list[nn.Parameter] = []
    for name, p in model.esm.named_parameters():
        if "layers." not in name:
            embed_params.append(p)
    if embed_params:
        embed_lr = backbone_lr * (llrd_decay ** n)
        param_groups.append({"params": embed_params, "lr": embed_lr})

    return AdamW(param_groups, weight_decay=weight_decay)


# -- Sklearn-compatible predictor (for predict_blind.py) ----------------------
class FinetunePredictor650M:
    """Wraps a saved ESM2Classifier650M state-dict for sklearn-style predict_proba.

    The backbone and head are loaded lazily on the first call to
    ``predict_proba`` and then **cached** on the instance so that repeated
    calls (e.g. across folds or in ensemble pipelines) avoid reloading the
    ~2.5 GB weights and re-allocating GPU memory.

    Parameters
    ----------
    model_path : Path or str
        Path to the ``.pt`` file saved by ``retrain_full_650m``.
    max_len : int
        Maximum sequence length used when *training* this model.  Persisted
        in the joblib artefact so inference always matches training truncation.
    """

    def __init__(self, model_path: Path | str, max_len: int = MAX_LEN):
        self.model_path = Path(model_path)
        self.max_len = max_len
        # Lazily-initialised cache — not serialised by joblib/pickle.
        self._model: ESM2Classifier650M | None = None
        self._batch_converter = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # Pickle/joblib compatibility: strip the in-process GPU cache so the
    # artefact file never contains serialised PyTorch tensors.
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = None
        state["_batch_converter"] = None
        state["_device"] = None
        return state

    def _ensure_loaded(self) -> None:
        """Load model and alphabet on first call; subsequent calls are no-ops."""
        if self._model is not None:
            return
        import esm as esm_lib

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
        model = ESM2Classifier650M(esm_model).to(device)
        state = torch.load(self.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        self._model = model
        self._batch_converter = alphabet.get_batch_converter()
        self._device = device

    def predict_proba(
        self,
        sequences: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Return softmax probability matrix, shape ``(N, 7)``."""
        self._ensure_loaded()
        model = self._model
        device = self._device
        max_len = self.max_len

        all_proba: list[np.ndarray] = []
        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch_seqs = sequences[start : start + batch_size]
                data = [(f"s{i}", s[:max_len]) for i, s in enumerate(batch_seqs)]
                _, _, tokens = self._batch_converter(data)
                seq_lengths = torch.tensor(
                    [len(s[:max_len]) for s in batch_seqs], dtype=torch.long
                )
                tokens = tokens.to(device)
                seq_lengths = seq_lengths.to(device)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    logits = model(tokens, seq_lengths)

                proba = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                all_proba.append(proba)

        return np.vstack(all_proba)


# -- Training helpers ----------------------------------------------------------
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }


def _run_epoch(
    model: ESM2Classifier650M,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
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

    if train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        for step, (tokens, labels, seq_lengths) in enumerate(loader):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            is_last = step == n_batches - 1

            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                logits = model(tokens, seq_lengths)
                loss = criterion(logits, labels)
                if train:
                    loss = loss / grad_accum

            if train:
                amp_scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or is_last:
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def _collect_oof(
    model: ESM2Classifier650M,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect OOF predictions and probabilities without gradients."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for tokens, _labels, seq_lengths in loader:
            tokens = tokens.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                logits = model(tokens, seq_lengths)
            proba = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            all_preds.append(proba.argmax(axis=1))
            all_proba.append(proba)
    return np.concatenate(all_preds), np.vstack(all_proba)


# -- Cross-validation ----------------------------------------------------------
def cross_validate_finetune_650m(
    sequences: list[str],
    y: np.ndarray,
    cv_splits: list,
    *,
    epochs: int = 15,
    batch_size: int = 8,
    grad_accum: int = 2,
    backbone_lr: float = 1e-5,
    head_lr: float = 5e-5,
    llrd_decay: float = 0.9,
    patience: int = 7,
    max_len: int = MAX_LEN,
    use_grad_ckpt: bool = False,
    device: torch.device,
    figures_dir: Path,
    checkpoints_dir: Path,
) -> dict:
    """5-fold cross-validation for ESM-2 650M fine-tuning.

    Parameters
    ----------
    sequences      : List of amino acid strings.
    y              : Integer label array of length N.
    cv_splits      : Stratified fold indices (from get_cv_splits).
    epochs         : Maximum training epochs per fold.
    batch_size     : Per-step batch size (before gradient accumulation).
    grad_accum     : Steps to accumulate before an optimiser update.
    backbone_lr    : LR for the outermost ESM-2 transformer layer.
    head_lr        : LR for the classification head.
    llrd_decay     : Per-layer LR decay factor (0.9 = 10% reduction per layer).
    patience       : Early-stopping patience (epochs without val F1 improvement).
    max_len        : Maximum sequence length in tokens.
    use_grad_ckpt  : Enable gradient checkpointing to reduce VRAM usage.
    device         : Torch device (cuda recommended).
    figures_dir    : Directory for confusion-matrix PNGs.
    checkpoints_dir: Directory for best-epoch model checkpoints.

    Returns
    -------
    dict with ``fold_results``, ``summary``, ``oof_preds``, ``oof_true``,
    ``oof_proba``, and ``model_name``.
    """
    import esm as esm_lib

    fold_results: list[dict] = []
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    eff_batch = batch_size * grad_accum
    print(f"\n{'='*72}")
    print(f"  ESM-2 650M FINE-TUNING -- {len(cv_splits)}-fold CV")
    print(f"  epochs={epochs}  batch={batch_size}  grad_accum={grad_accum}  "
          f"(eff_batch={eff_batch})  max_len={max_len}")
    print(f"  backbone_lr={backbone_lr:.0e}  head_lr={head_lr:.0e}  "
          f"llrd_decay={llrd_decay}  grad_ckpt={use_grad_ckpt}")
    print(f"{'='*72}")

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n{'-'*72}")
        print(f"  FOLD {fold_i+1}/{len(cv_splits)}  "
              f"train={len(train_idx):,}  val={len(val_idx):,}")
        print(f"{'-'*72}")
        t_fold = time.time()

        # Fresh model per fold — no cross-fold contamination
        esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
        model = ESM2Classifier650M(
            esm_model, use_grad_ckpt=use_grad_ckpt
        ).to(device)
        batch_converter = alphabet.get_batch_converter()
        collate = make_collate_fn(batch_converter)

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs   = [sequences[i] for i in val_idx]
        y_train    = y[train_idx]
        y_val      = y[val_idx]

        train_loader = DataLoader(
            ProteinDataset(train_seqs, y_train.tolist(), max_len=max_len),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            ProteinDataset(val_seqs, y_val.tolist(), max_len=max_len),
            batch_size=max(1, batch_size // 2), shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
        )

        # Class weights — fitted on train fold only
        cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_train)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float32).to(device)
        )

        # LLRD optimiser
        optimizer = build_llrd_optimizer(
            model,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            llrd_decay=llrd_decay,
        )

        # Scheduler steps once per effective-batch update
        accum_steps = (len(train_loader) + grad_accum - 1) // grad_accum
        total_steps = epochs * accum_steps

        # Collect per-group max_lrs for OneCycleLR
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.1,
        )
        grad_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

        best_val_f1   = -1.0
        patience_ctr  = 0
        ckpt_path     = checkpoints_dir / f"finetune_650m_fold{fold_i+1}.pt"
        best_oof_preds: np.ndarray | None = None
        best_oof_proba: np.ndarray | None = None

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            tr = _run_epoch(
                model, train_loader, criterion, optimizer,
                grad_scaler, device, train=True,
                scheduler=scheduler, grad_accum=grad_accum,
            )
            vl = _run_epoch(
                model, val_loader, criterion, optimizer,
                grad_scaler, device, train=False,
            )

            improved = vl["macro_f1"] > best_val_f1
            marker   = " * NEW BEST" if improved else ""

            # Log VRAM usage on CUDA
            vram_str = ""
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info()
                used_gb = (total - free) / 1e9
                vram_str = f"  VRAM={used_gb:.1f}GB"

            print(
                f"  Epoch {epoch:>2}/{epochs}  "
                f"loss={tr['loss']:.4f}/{vl['loss']:.4f}  "
                f"F1={vl['macro_f1']:.4f}  "
                f"BA={vl['balanced_accuracy']:.4f}  "
                f"[{fmt_time(time.time()-t_ep)}]{vram_str}{marker}",
                flush=True,
            )

            if improved:
                best_val_f1  = vl["macro_f1"]
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
                best_oof_preds, best_oof_proba = _collect_oof(
                    model, val_loader, device
                )
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(
                        f"  Early stop (patience={patience} epochs without improvement)"
                    )
                    break

        oof_preds[val_idx] = best_oof_preds
        oof_proba[val_idx] = best_oof_proba

        fold_m = _compute_metrics(y_val, best_oof_preds)
        fold_results.append(fold_m)
        print(
            f"\n  Fold {fold_i+1} done  "
            f"F1={fold_m['macro_f1']:.4f}  "
            f"BA={fold_m['balanced_accuracy']:.4f}  "
            f"MCC={fold_m['mcc']:.4f}  "
            f"[{fmt_time(time.time()-t_fold)}]"
        )

        del model, esm_model
        torch.cuda.empty_cache()

    # Aggregate CV summary
    keys = list(fold_results[0].keys())
    summary = {
        k + "_mean": float(np.mean([f[k] for f in fold_results])) for k in keys
    }
    summary.update(
        {k + "_std": float(np.std([f[k] for f in fold_results])) for k in keys}
    )

    return {
        "model_name":   "ESM2-650M-Finetune",
        "fold_results": fold_results,
        "summary":      summary,
        "oof_preds":    oof_preds,
        "oof_true":     y,
        "oof_proba":    oof_proba,
    }


# -- Full-dataset retrain ------------------------------------------------------
def retrain_full_650m(
    sequences: list[str],
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    grad_accum: int = 2,
    backbone_lr: float = 1e-5,
    head_lr: float = 5e-5,
    llrd_decay: float = 0.9,
    max_len: int = MAX_LEN,
    use_grad_ckpt: bool = False,
    device: torch.device,
    save_path: Path,
) -> None:
    """Retrain ESM-2 650M on the full dataset and save the final weights."""
    import esm as esm_lib

    print(f"\n{'='*72}")
    print(
        f"  FINAL RETRAIN on full dataset  "
        f"({len(sequences):,} sequences, {epochs} epochs)"
    )
    print(f"{'='*72}")

    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    model = ESM2Classifier650M(
        esm_model, use_grad_ckpt=use_grad_ckpt
    ).to(device)
    collate = make_collate_fn(alphabet.get_batch_converter())

    loader = DataLoader(
        ProteinDataset(sequences, y.tolist(), max_len=max_len),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=(device.type == "cuda"),
    )
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device)
    )
    optimizer = build_llrd_optimizer(
        model, backbone_lr=backbone_lr, head_lr=head_lr, llrd_decay=llrd_decay
    )
    max_lrs = [pg["lr"] for pg in optimizer.param_groups]
    accum_steps = (len(loader) + grad_accum - 1) // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=epochs * accum_steps,
        pct_start=0.1,
    )
    grad_scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        t = time.time()
        m = _run_epoch(
            model, loader, criterion, optimizer,
            grad_scaler, device, train=True,
            scheduler=scheduler, grad_accum=grad_accum,
        )
        vram_str = ""
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            used_gb = (total - free) / 1e9
            vram_str = f"  VRAM={used_gb:.1f}GB"
        print(
            f"  Epoch {epoch:>2}/{epochs}  "
            f"loss={m['loss']:.4f}  F1={m['macro_f1']:.4f}  "
            f"[{fmt_time(time.time()-t)}]{vram_str}",
            flush=True,
        )

    torch.save(model.state_dict(), save_path)
    print(f"\n  Saved -> {save_path}")
    del model, esm_model
    torch.cuda.empty_cache()


# -- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM-2 650M for enzyme classification (A40 optimised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "A40 (48 GB VRAM) recommended settings:\n"
            "  python -m src.models.finetune_650m \\\n"
            "      --epochs 15 --batch-size 8 --grad-accum 2 --max-len 1022\n\n"
            "Low-VRAM / debugging:\n"
            "  python -m src.models.finetune_650m \\\n"
            "      --epochs 5 --batch-size 2 --grad-accum 8 --use-grad-ckpt\n"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Maximum epochs per fold",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Per-step batch size (effective_batch = batch_size × grad_accum)",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-len", type=int, default=MAX_LEN,
        help="Maximum sequence length (tokens).  1022 = full ESM-2 context.",
    )
    parser.add_argument(
        "--backbone-lr", type=float, default=1e-5,
        help="LR for the outermost ESM-2 transformer layer",
    )
    parser.add_argument(
        "--head-lr", type=float, default=5e-5,
        help="LR for the classification head",
    )
    parser.add_argument(
        "--llrd-decay", type=float, default=0.9,
        help="Per-layer LR decay factor for LLRD (0.9 = 10%% reduction per layer)",
    )
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early-stopping patience (epochs without val Macro F1 improvement)",
    )
    parser.add_argument(
        "--use-grad-ckpt", action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage (~50%% less activations)",
    )
    parser.add_argument(
        "--retrain-epochs", type=int, default=None,
        help="Epochs for final full-dataset retrain (default: same as --epochs)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Device selection
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

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

    project_root    = Path(__file__).resolve().parent.parent.parent
    figures_dir     = project_root / "outputs" / "figures"
    models_dir      = project_root / "outputs" / "models"
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
    cv_result = cross_validate_finetune_650m(
        sequences, y, cv_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        llrd_decay=args.llrd_decay,
        patience=args.patience,
        max_len=args.max_len,
        use_grad_ckpt=args.use_grad_ckpt,
        device=device,
        figures_dir=figures_dir,
        checkpoints_dir=models_dir,
    )

    summary = cv_result["summary"]
    print(f"\n{'='*72}")
    print(f"  ESM-2 650M Fine-tune -- 5-Fold CV Summary")
    print(f"{'='*72}")
    print(f"  Accuracy:          {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  Macro F1:          {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
    print(f"  Balanced Accuracy: {summary['balanced_accuracy_mean']:.4f} ± {summary['balanced_accuracy_std']:.4f}")
    print(f"  MCC:               {summary['mcc_mean']:.4f} ± {summary['mcc_std']:.4f}")
    print(f"{'='*72}")
    print(f"  Total CV time: {fmt_time(time.time()-t_cv)}")

    # OOF classification report
    print("\n--- OOF Classification Report ---")
    print(
        classification_report(
            cv_result["oof_true"], cv_result["oof_preds"],
            target_names=CLASS_NAMES, zero_division=0,
        )
    )

    # Confusion matrix
    plot_confusion_matrix(
        cv_result["oof_true"], cv_result["oof_preds"],
        save_path=figures_dir / "cm_finetune_650m.png",
        title="ESM-2 650M Fine-tuned -- OOF Confusion Matrix",
    )

    # Save CV results JSON
    results_path = project_root / "outputs" / "finetune_650m_results.json"
    results_data = dict(cv_result)
    results_data["config"] = {
        "esm_model":    ESM_MODEL_NAME,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "grad_accum":   args.grad_accum,
        "max_len":      args.max_len,
        "backbone_lr":  args.backbone_lr,
        "head_lr":      args.head_lr,
        "llrd_decay":   args.llrd_decay,
        "use_grad_ckpt": args.use_grad_ckpt,
    }
    save_results_json(results_data, results_path)
    print(f"\nResults saved -> {results_path}")

    # Final retrain on full dataset
    retrain_ep = args.retrain_epochs if args.retrain_epochs else args.epochs
    final_model_path = models_dir / "finetune_650m_final.pt"
    retrain_full_650m(
        sequences, y,
        epochs=retrain_ep,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        llrd_decay=args.llrd_decay,
        max_len=args.max_len,
        use_grad_ckpt=args.use_grad_ckpt,
        device=device,
        save_path=final_model_path,
    )

    # Save joblib artifact for predict_blind.py
    predictor = FinetunePredictor650M(final_model_path, max_len=args.max_len)
    artifact = {
        "model":             predictor,
        "scaler":            None,
        "feature_source":    "finetune_650m",
        "esm_model_name":    ESM_MODEL_NAME,
        "esm_embedding_dim": EMB_DIM,
        "max_len":           args.max_len,
        "cv_scores":         summary,
        "model_name":        "ESM2-650M-Finetune",
    }
    artifact_path = models_dir / "finetune_650m_artifact.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Artifact saved -> {artifact_path}")
