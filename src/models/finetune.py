"""
ESM-2 8M end-to-end fine-tuning for 7-class enzyme classification.

Architecture:
    ESM-2 8M  (esm2_t6_8M_UR50D, 6 transformer layers, fully unfrozen)
    -> mean-pool over sequence positions (identical to embeddings.py)
    -> Linear(320, 128) -> GELU -> Dropout(0.3) -> Linear(128, 7)

Training:
    - Stratified 5-fold CV (same indices as rest of pipeline)
    - AdamW: backbone lr=2e-5, classifier head lr=1e-4
    - OneCycleLR (10% warmup + cosine decay)
    - Weighted cross-entropy (balanced class weights, train fold only)
    - bfloat16 mixed precision (fp16 fallback)
    - Gradient clipping  max_norm=1.0
    - Early stopping on val Macro F1  patience=5

Usage:
    python -m src.models.finetune
    python -m src.models.finetune --epochs 15 --batch-size 32
    python -m src.models.finetune --epochs 10 --batch-size 64 --lr-backbone 1e-5
"""

import argparse
import json
import logging
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
    accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

from src.data_loading import load_all_sequences, get_cv_splits, SEED, CLASS_NAMES
from src.training import fmt_time
from src.evaluation import plot_confusion_matrix, save_results_json

logger = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MAX_LEN = 1022
N_CLASSES = 7
EMB_DIM = 320       # ESM-2 8M output dimension
ESM_MODEL_NAME = "esm2_t6_8M_UR50D"


# ── Dataset ────────────────────────────────────────────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int], max_len: int = MAX_LEN):
        self.sequences = [s[:max_len] for s in sequences]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.sequences[idx], self.labels[idx]


def make_collate_fn(batch_converter):
    """Return a collate function that tokenizes sequences using ESM-2's alphabet."""
    def collate(batch: list[tuple[str, int]]):
        seqs, labels = zip(*batch)
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return tokens, torch.tensor(labels, dtype=torch.long), seq_lengths
    return collate


# ── Model ──────────────────────────────────────────────────────────────────────
class ESM2Classifier(nn.Module):
    """ESM-2 backbone + classification head."""

    def __init__(self, esm_model, emb_dim: int = EMB_DIM,
                 n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.esm = esm_model
        self.n_layers = esm_model.num_layers
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, tokens: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        out = self.esm(tokens, repr_layers=[self.n_layers])
        reps = out["representations"][self.n_layers]   # (B, L+2, D)
        B, L, D = reps.shape
        # Mean-pool positions 1..seq_len (matches embeddings.py exactly)
        pos = torch.arange(L, device=reps.device).unsqueeze(0)         # (1, L)
        mask = (pos >= 1) & (pos <= seq_lengths.unsqueeze(1))           # (B, L)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (reps * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


# ── Sklearn-compatible predictor (for predict_blind.py) ───────────────────────
class FinetunePredictor:
    """Wraps a saved ESM2Classifier state-dict for sklearn-style predict_proba.

    The model weights are loaded lazily on first call to predict_proba so the
    object can be pickled/joblibed without serialising the entire model.
    """

    def __init__(self, model_path: Path | str):
        self.model_path = Path(model_path)

    def predict_proba(self, sequences: list[str],
                      batch_size: int = 32) -> np.ndarray:
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))
        import esm as esm_lib
        esm_model, alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
        model = ESM2Classifier(esm_model).to(device)
        state = torch.load(self.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        batch_converter = alphabet.get_batch_converter()

        all_proba: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch_seqs = sequences[start:start + batch_size]
                data = [(f"s{i}", s[:MAX_LEN]) for i, s in enumerate(batch_seqs)]
                _, _, tokens = batch_converter(data)
                seq_lengths = torch.tensor([len(s[:MAX_LEN]) for s in batch_seqs],
                                           dtype=torch.long)
                tokens = tokens.to(device)
                seq_lengths = seq_lengths.to(device)
                logits = model(tokens, seq_lengths)
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
                all_proba.append(proba)

        del model, esm_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return np.vstack(all_proba)


# ── Training helpers ───────────────────────────────────────────────────────────
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }


def _run_epoch(
    model: ESM2Classifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train: bool = True,
    scheduler=None,
) -> dict:
    model.train(train)
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    ctx = torch.set_grad_enabled(train)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    amp_enabled = device.type == "cuda"

    with ctx:
        for tokens, labels, seq_lengths in loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                    enabled=amp_enabled):
                logits = model(tokens, seq_lengths)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return {"loss": total_loss / len(y_true), **_compute_metrics(y_true, y_pred)}


def _collect_oof(model: ESM2Classifier, loader: DataLoader,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Collect OOF predictions and probabilities (no gradient)."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, _labels, seq_lengths in loader:
            tokens = tokens.to(device, non_blocking=True)
            seq_lengths = seq_lengths.to(device, non_blocking=True)
            logits = model(tokens, seq_lengths)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            all_preds.append(proba.argmax(axis=1))
            all_proba.append(proba)
    return np.concatenate(all_preds), np.vstack(all_proba)


# ── Cross-validation ───────────────────────────────────────────────────────────
def cross_validate_finetune(
    sequences: list[str],
    y: np.ndarray,
    cv_splits: list,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr_backbone: float = 2e-5,
    lr_head: float = 1e-4,
    patience: int = 5,
    device: torch.device,
    figures_dir: Path,
    checkpoints_dir: Path,
) -> dict:
    import esm as esm_lib

    fold_results: list[dict] = []
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    print(f"\n{'='*72}")
    print(f"  ESM-2 8M FINE-TUNING — {len(cv_splits)}-fold CV")
    print(f"  epochs={epochs}  batch={batch_size}  "
          f"lr_backbone={lr_backbone:.0e}  lr_head={lr_head:.0e}")
    print(f"{'='*72}")

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n{'─'*72}")
        print(f"  FOLD {fold_i+1}/{len(cv_splits)}  "
              f"train={len(train_idx):,}  val={len(val_idx):,}")
        print(f"{'─'*72}")
        t_fold = time.time()

        # Fresh model per fold — no cross-fold contamination
        esm_model, alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
        model = ESM2Classifier(esm_model).to(device)
        batch_converter = alphabet.get_batch_converter()
        collate = make_collate_fn(batch_converter)

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs   = [sequences[i] for i in val_idx]
        y_train    = y[train_idx]
        y_val      = y[val_idx]

        train_loader = DataLoader(
            ProteinDataset(train_seqs, y_train.tolist()),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            ProteinDataset(val_seqs, y_val.tolist()),
            batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate, num_workers=0, pin_memory=True,
        )

        # Class weights fitted on train fold only
        cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_train)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(cw, dtype=torch.float32).to(device)
        )

        # Differential learning rates
        optimizer = AdamW(
            [
                {"params": model.esm.parameters(),         "lr": lr_backbone},
                {"params": model.classifier.parameters(),  "lr": lr_head},
            ],
            weight_decay=0.01,
        )
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr_backbone, lr_head],
            total_steps=total_steps,
            pct_start=0.1,
        )
        grad_scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_val_f1 = -1.0
        patience_ctr = 0
        ckpt_path = checkpoints_dir / f"finetune_fold{fold_i+1}.pt"
        best_oof_preds: np.ndarray | None = None
        best_oof_proba: np.ndarray | None = None

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            tr = _run_epoch(model, train_loader, criterion, optimizer,
                            grad_scaler, device, train=True, scheduler=scheduler)
            vl = _run_epoch(model, val_loader, criterion, optimizer,
                            grad_scaler, device, train=False)

            improved = vl["macro_f1"] > best_val_f1
            marker   = " ★ NEW BEST" if improved else ""
            print(
                f"  Epoch {epoch:>2}/{epochs}  "
                f"loss={tr['loss']:.4f}/{vl['loss']:.4f}  "
                f"F1={vl['macro_f1']:.4f}  "
                f"BA={vl['balanced_accuracy']:.4f}  "
                f"[{fmt_time(time.time()-t_ep)}]{marker}",
                flush=True,
            )

            if improved:
                best_val_f1 = vl["macro_f1"]
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
                best_oof_preds, best_oof_proba = _collect_oof(model, val_loader, device)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stop (patience={patience} epochs without improvement)")
                    break

        # Store OOF from best checkpoint epoch
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
    summary = {k + "_mean": float(np.mean([f[k] for f in fold_results])) for k in keys}
    summary.update({k + "_std": float(np.std([f[k] for f in fold_results])) for k in keys})

    return {
        "model_name": "ESM2-8M-Finetune",
        "fold_results": fold_results,
        "summary": summary,
        "oof_preds": oof_preds,
        "oof_true":  y,
        "oof_proba": oof_proba,
    }


# ── Full-dataset retrain ───────────────────────────────────────────────────────
def retrain_full(
    sequences: list[str],
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr_backbone: float,
    lr_head: float,
    device: torch.device,
    save_path: Path,
) -> None:
    import esm as esm_lib

    print(f"\n{'='*72}")
    print(f"  FINAL RETRAIN on full dataset  ({len(sequences):,} sequences, {epochs} epochs)")
    print(f"{'='*72}")

    esm_model, alphabet = esm_lib.pretrained.esm2_t6_8M_UR50D()
    model = ESM2Classifier(esm_model).to(device)
    collate = make_collate_fn(alphabet.get_batch_converter())

    loader = DataLoader(
        ProteinDataset(sequences, y.tolist()),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device)
    )
    optimizer = AdamW(
        [
            {"params": model.esm.parameters(),        "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[lr_backbone, lr_head],
        total_steps=epochs * len(loader),
        pct_start=0.1,
    )
    grad_scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        t = time.time()
        m = _run_epoch(model, loader, criterion, optimizer,
                       grad_scaler, device, train=True, scheduler=scheduler)
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"loss={m['loss']:.4f}  F1={m['macro_f1']:.4f}  "
              f"[{fmt_time(time.time()-t)}]", flush=True)

    torch.save(model.state_dict(), save_path)
    print(f"\n  Saved -> {save_path}")
    del model, esm_model
    torch.cuda.empty_cache()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM-2 8M for enzyme classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",       type=int,   default=10,   help="Max epochs per fold")
    parser.add_argument("--batch-size",   type=int,   default=32,   help="Training batch size")
    parser.add_argument("--lr-backbone",  type=float, default=2e-5, help="LR for ESM-2 backbone")
    parser.add_argument("--lr-head",      type=float, default=1e-4, help="LR for classifier head")
    parser.add_argument("--patience",     type=int,   default=5,    help="Early stopping patience")
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
    logging.getLogger("PIL").setLevel(logging.WARNING)

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

    project_root   = Path(__file__).resolve().parent.parent.parent
    figures_dir    = project_root / "outputs" / "figures"
    models_dir     = project_root / "outputs" / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sequences ────────────────────────────────────────────────────────
    t0 = time.time()
    print("\nLoading sequences...")
    df = load_all_sequences(project_root)
    sequences = df["sequence"].tolist()
    y = df["label"].values
    cv_splits = get_cv_splits(y)
    print(f"  Loaded {len(df):,} sequences  [{fmt_time(time.time()-t0)}]")
    print(f"  Class distribution: {np.bincount(y).tolist()}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    t_cv = time.time()
    cv_result = cross_validate_finetune(
        sequences, y, cv_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        patience=args.patience,
        device=device,
        figures_dir=figures_dir,
        checkpoints_dir=models_dir,
    )

    summary = cv_result["summary"]
    print(f"\n{'='*72}")
    print(f"  ESM-2 8M Fine-tune -- 5-Fold CV Summary")
    print(f"{'='*72}")
    print(f"  Accuracy:          {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Macro F1:          {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"  Balanced Accuracy: {summary['balanced_accuracy_mean']:.4f} +/- {summary['balanced_accuracy_std']:.4f}")
    print(f"  MCC:               {summary['mcc_mean']:.4f} +/- {summary['mcc_std']:.4f}")
    print(f"{'='*72}")
    print(f"  Total CV time: {fmt_time(time.time()-t_cv)}")

    # Classification report on OOF
    print("\n--- OOF Classification Report ---")
    print(classification_report(
        cv_result["oof_true"], cv_result["oof_preds"],
        target_names=CLASS_NAMES, zero_division=0,
    ))

    # Confusion matrix
    plot_confusion_matrix(
        cv_result["oof_true"], cv_result["oof_preds"],
        save_path=figures_dir / "cm_finetune.png",
        title="ESM-2 8M Fine-tuned -- OOF Confusion Matrix",
    )

    # ── Save CV results JSON ──────────────────────────────────────────────────
    results_path = project_root / "outputs" / "finetune_results.json"
    results_data = dict(cv_result)
    results_data["config"] = {
        "esm_model": ESM_MODEL_NAME, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
    }
    save_results_json(results_data, results_path)
    print(f"\nResults saved -> {results_path}")

    # ── Final retrain on full dataset ─────────────────────────────────────────
    retrain_ep = args.retrain_epochs if args.retrain_epochs else args.epochs
    final_model_path = models_dir / "finetune_final.pt"
    retrain_full(
        sequences, y,
        epochs=retrain_ep,
        batch_size=args.batch_size,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        device=device,
        save_path=final_model_path,
    )

    # ── Save joblib artifact for predict_blind.py ─────────────────────────────
    predictor = FinetunePredictor(final_model_path)
    artifact = {
        "model":           predictor,
        "scaler":          None,          # No external scaler — model handles raw sequences
        "feature_source":  "finetune",
        "esm_model_name":  ESM_MODEL_NAME,
        "esm_embedding_dim": EMB_DIM,
        "cv_scores":       summary,
        "model_name":      "ESM2-8M-Finetune",
    }
    artifact_path = models_dir / "finetune_artifact.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Joblib artifact saved -> {artifact_path}")

    print(f"\nTotal wall time: {fmt_time(time.time()-t0)}")
