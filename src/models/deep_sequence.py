"""
Native sequential deep learning: 1D-CNN and Bidirectional LSTM.

Instead of treating mean-pooled embeddings as tabular features for XGBoost,
these models operate on the *unpooled* per-token ESM-2 representations of shape
(sequence_length, embedding_dim).  They natively slide across the sequence to
capture local structural motifs and distant active-site interactions.

Architecture options:
    ProteinCNN   — Multi-scale 1D convolutions (kernel sizes 3, 5, 7)
                   → global max-pool → classifier head.
    ProteinLSTM  — 2-layer bidirectional LSTM → last hidden state → classifier.

Both are trained with Focal Loss for principled class imbalance correction.

Usage:
    python -m src.models.deep_sequence                        # 1D-CNN with 8M
    python -m src.models.deep_sequence --arch lstm            # BiLSTM with 8M
    python -m src.models.deep_sequence --esm-model 650M       # use 650M embeddings
    python -m src.models.deep_sequence --epochs 15 --batch-size 8
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

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

N_CLASSES = 7


# ---------------------------------------------------------------------------
# Dataset: extracts per-token representations on-the-fly from frozen ESM-2
# ---------------------------------------------------------------------------

class ProteinSeqDataset(Dataset):
    """Returns (sequence_string, label) pairs; tokenisation deferred to collate."""

    def __init__(self, sequences: list[str], labels: list[int], max_len: int = 512):
        self.sequences = [s[:max_len] for s in sequences]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.sequences[idx], self.labels[idx]


def make_collate_fn(batch_converter):
    """Collate that tokenises sequences using ESM-2's alphabet."""
    def collate(batch):
        seqs, labels = zip(*batch)
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return tokens, torch.tensor(labels, dtype=torch.long), seq_lengths
    return collate


# ---------------------------------------------------------------------------
# Feature extractor: frozen ESM-2 backbone
# ---------------------------------------------------------------------------

class FrozenESM2Backbone(nn.Module):
    """Frozen ESM-2 backbone that returns per-token representations."""

    def __init__(self, esm_model):
        super().__init__()
        self.esm = esm_model
        self.n_layers = esm_model.num_layers
        # Freeze all parameters
        for param in self.esm.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, seq_lengths: torch.Tensor):
        """Return per-token embeddings (B, L_max, D) and a valid-position mask."""
        out = self.esm(tokens, repr_layers=[self.n_layers])
        reps = out["representations"][self.n_layers]  # (B, L+2, D)
        B, L, D = reps.shape

        # Build mask for positions 1..seq_len (exclude BOS/EOS)
        pos = torch.arange(L, device=reps.device).unsqueeze(0)  # (1, L)
        mask = (pos >= 1) & (pos <= seq_lengths.unsqueeze(1))    # (B, L)

        # Extract only valid positions, zero-padded
        max_seq_len = seq_lengths.max().item()
        padded = torch.zeros(B, max_seq_len, D, device=reps.device, dtype=reps.dtype)
        for i in range(B):
            slen = seq_lengths[i].item()
            padded[i, :slen] = reps[i, 1:slen + 1]

        # Mask: True where valid
        pad_mask = torch.arange(max_seq_len, device=reps.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        return padded, pad_mask


# ---------------------------------------------------------------------------
# 1D-CNN model
# ---------------------------------------------------------------------------

class ProteinCNN(nn.Module):
    """Multi-scale 1D-CNN for protein sequence classification.

    Three parallel convolution branches (kernel sizes 3, 5, 7) each producing
    128 feature maps.  Global max-pooling over sequence length → concatenation
    → classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = N_CLASSES,
        n_filters: int = 128,
        kernel_sizes: tuple = (3, 5, 7),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, n_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(n_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        total_filters = n_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_filters, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D) — per-token embeddings
        mask : (B, L) — True for valid positions
        """
        # Conv1d expects (B, D, L)
        x = x.transpose(1, 2)  # (B, D, L)

        # Mask invalid positions
        mask_expanded = mask.unsqueeze(1).float()  # (B, 1, L)
        x = x * mask_expanded

        branch_outs = []
        for conv in self.conv_branches:
            h = conv(x)  # (B, n_filters, L)
            h = h * mask_expanded  # mask after conv
            # Global max-pool over sequence length
            h = h.masked_fill(~mask.unsqueeze(1), float("-inf"))
            h = h.max(dim=2).values  # (B, n_filters)
            branch_outs.append(h)

        combined = torch.cat(branch_outs, dim=1)  # (B, total_filters)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# BiLSTM model
# ---------------------------------------------------------------------------

class ProteinLSTM(nn.Module):
    """Bidirectional LSTM for protein sequence classification.

    2-layer BiLSTM → attention-weighted pooling → classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = N_CLASSES,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D) — per-token embeddings
        mask : (B, L) — True for valid positions
        """
        # Pack padded sequences for efficient LSTM processing
        lengths = mask.sum(dim=1).cpu()  # (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False,
        )
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # output: (B, L, 2*hidden_dim)

        # Attention-weighted pooling
        attn_scores = self.attention(output).squeeze(-1)  # (B, L)
        attn_scores = attn_scores.masked_fill(~mask[:, :output.size(1)], float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        context = (output * attn_weights).sum(dim=1)  # (B, 2*hidden_dim)

        return self.classifier(context)


# ---------------------------------------------------------------------------
# Wrapper combining frozen ESM-2 + sequence model
# ---------------------------------------------------------------------------

class ESM2SequenceClassifier(nn.Module):
    """Frozen ESM-2 backbone + trainable sequence classifier (CNN or LSTM)."""

    def __init__(self, backbone: FrozenESM2Backbone, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.seq_classifier = classifier

    def forward(self, tokens: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        embeddings, mask = self.backbone(tokens, seq_lengths)
        return self.seq_classifier(embeddings, mask)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }


def _run_epoch(
    model: ESM2SequenceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
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
    grad_scaler = torch.amp.GradScaler(enabled=amp_enabled)
    n_batches = len(loader)

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(train):
        for step, (tokens, labels, seq_lengths) in enumerate(loader):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            is_last = (step == n_batches - 1)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                logits = model(tokens, seq_lengths)
                loss = criterion(logits, labels)
                if train:
                    loss = loss / grad_accum

            if train and optimizer is not None:
                grad_scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or is_last:
                    grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
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

def cross_validate_deep_seq(
    sequences: list[str],
    y: np.ndarray,
    cv_splits: list,
    *,
    arch: str = "cnn",
    esm_model_name: str = "esm2_t6_8M_UR50D",
    epochs: int = 10,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-3,
    patience: int = 5,
    max_len: int = 512,
    device: torch.device,
    figures_dir: Path,
    checkpoints_dir: Path,
    focal_gamma: float = 2.0,
) -> dict:
    """5-fold CV for 1D-CNN or BiLSTM on ESM-2 per-token embeddings."""
    import esm as esm_lib

    from src.features.embeddings import MODEL_REGISTRY
    emb_dim = MODEL_REGISTRY[esm_model_name]["dim"]

    fold_results: list[dict] = []
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    arch_label = "1D-CNN" if arch == "cnn" else "BiLSTM"

    print(f"\n{'='*72}")
    print(f"  {arch_label} + Frozen ESM-2 ({esm_model_name}) -- {len(cv_splits)}-fold CV")
    print(f"  epochs={epochs}  batch={batch_size}  grad_accum={grad_accum}  "
          f"lr={lr:.0e}  focal_gamma={focal_gamma}")
    print(f"{'='*72}")

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n{'-'*72}")
        print(f"  FOLD {fold_i+1}/{len(cv_splits)}  "
              f"train={len(train_idx):,}  val={len(val_idx):,}")
        print(f"{'-'*72}")
        t_fold = time.time()

        # Fresh ESM-2 backbone per fold
        esm_loader = getattr(esm_lib.pretrained, esm_model_name)
        esm_model, alphabet = esm_loader()
        backbone = FrozenESM2Backbone(esm_model)
        batch_converter = alphabet.get_batch_converter()
        collate = make_collate_fn(batch_converter)

        # Build sequence classifier
        if arch == "cnn":
            seq_model = ProteinCNN(input_dim=emb_dim)
        else:
            seq_model = ProteinLSTM(input_dim=emb_dim)

        model = ESM2SequenceClassifier(backbone, seq_model).to(device)

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        train_loader = DataLoader(
            ProteinSeqDataset(train_seqs, y_train.tolist(), max_len=max_len),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            ProteinSeqDataset(val_seqs, y_val.tolist(), max_len=max_len),
            batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )

        # Focal loss with per-class alpha from train fold
        alpha = compute_focal_alpha(y_train, n_classes=N_CLASSES).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)

        # Only optimise the sequence classifier parameters (backbone is frozen)
        trainable_params = [p for p in model.seq_classifier.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)

        accum_steps = (len(train_loader) + grad_accum - 1) // grad_accum
        total_steps = epochs * accum_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1,
        )

        best_val_f1 = -1.0
        patience_ctr = 0
        ckpt_path = checkpoints_dir / f"deep_seq_{arch}_fold{fold_i+1}.pt"
        best_oof_preds = None
        best_oof_proba = None

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            tr = _run_epoch(model, train_loader, criterion, optimizer,
                            device, train=True, scheduler=scheduler,
                            grad_accum=grad_accum)
            vl = _run_epoch(model, val_loader, criterion, None,
                            device, train=False)

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
                torch.save(model.seq_classifier.state_dict(), ckpt_path)
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

        del model, esm_model, backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate
    keys = list(fold_results[0].keys())
    summary = {k + "_mean": float(np.mean([f[k] for f in fold_results])) for k in keys}
    summary.update({k + "_std": float(np.std([f[k] for f in fold_results])) for k in keys})

    return {
        "model_name": f"{arch_label}-ESM2-{esm_model_name.split('_')[2]}",
        "fold_results": fold_results,
        "summary": summary,
        "oof_preds": oof_preds,
        "oof_true": y,
        "oof_proba": oof_proba,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential deep learning (1D-CNN / BiLSTM) on ESM-2 embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", default="cnn", choices=["cnn", "lstm"],
                        help="Sequence model architecture")
    parser.add_argument("--esm-model", default="8M", choices=["8M", "650M"],
                        help="ESM-2 backbone size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard CE)")
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

    ESM_ALIAS = {"8M": "esm2_t6_8M_UR50D", "650M": "esm2_t33_650M_UR50D"}
    esm_model_name = ESM_ALIAS[args.esm_model]

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
    cv_result = cross_validate_deep_seq(
        sequences, y, cv_splits,
        arch=args.arch,
        esm_model_name=esm_model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        patience=args.patience,
        max_len=args.max_len,
        device=device,
        figures_dir=figures_dir,
        checkpoints_dir=models_dir,
        focal_gamma=args.focal_gamma,
    )

    summary = cv_result["summary"]
    arch_label = "1D-CNN" if args.arch == "cnn" else "BiLSTM"
    print(f"\n{'='*72}")
    print(f"  {arch_label} + ESM-2 ({esm_model_name}) -- 5-Fold CV Summary")
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
        save_path=figures_dir / f"cm_deep_seq_{args.arch}.png",
        title=f"{arch_label} + ESM-2 ({esm_model_name}) -- OOF Confusion Matrix",
    )

    # Save results
    results_data = dict(cv_result)
    results_data["config"] = {
        "arch": args.arch,
        "esm_model": esm_model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "focal_gamma": args.focal_gamma,
    }
    save_results_json(results_data, project_root / "outputs" / f"deep_seq_{args.arch}_results.json")

    print(f"\nTotal wall time: {fmt_time(time.time()-t0)}")
