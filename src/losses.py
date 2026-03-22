"""
Focal Loss for class-imbalanced classification.

Dynamically down-weights well-classified examples so the model focuses
its gradient updates on hard minority-class samples.  Replaces SMOTE and
simple class weighting with a mathematically principled loss function.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss with optional per-class alpha weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : Tensor of shape (C,) or None
        Per-class weights.  When ``None``, all classes are weighted equally.
        Typically set to inverse class frequency or sklearn's balanced weights.
    gamma : float
        Focusing parameter.  gamma=0 recovers standard cross-entropy.
        gamma=2 is the recommended default from the original paper.
    reduction : str
        ``'mean'`` (default), ``'sum'``, or ``'none'``.
    label_smoothing : float
        Optional label smoothing factor (0.0 = no smoothing).
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : (B, C)  raw logits (pre-softmax).
        targets : (B,)   integer class labels.
        """
        n_classes = logits.size(1)

        # Compute log-softmax for numerical stability
        log_p = F.log_softmax(logits, dim=1)  # (B, C)
        p = log_p.exp()  # (B, C)

        # Optional label smoothing: soft targets
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / n_classes
            targets_one_hot = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            targets_one_hot = (
                targets_one_hot * (1.0 - self.label_smoothing) + smooth
            )
            # Focal modulation on true-class probability
            p_t = (p * targets_one_hot).sum(dim=1)
            focal_weight = (1.0 - p_t) ** self.gamma
            loss = -(targets_one_hot * log_p).sum(dim=1)
            loss = focal_weight * loss
        else:
            # Gather true-class log-probability
            log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
            p_t = log_p_t.exp()  # (B,)

            focal_weight = (1.0 - p_t) ** self.gamma
            loss = -focal_weight * log_p_t  # (B,)

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # (B,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_focal_alpha(
    y: np.ndarray, n_classes: int = 7, method: str = "balanced"
) -> torch.Tensor:
    """Compute per-class alpha weights for focal loss.

    Parameters
    ----------
    y : array of integer labels.
    n_classes : number of classes.
    method : ``'balanced'`` (inverse frequency) or ``'effective'``
             (effective number of samples, Cui et al. 2019).

    Returns
    -------
    Tensor of shape (n_classes,) with per-class weights.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid division by zero

    if method == "balanced":
        # sklearn-style: n_samples / (n_classes * count_per_class)
        total = counts.sum()
        alpha = total / (n_classes * counts)
    elif method == "effective":
        # Effective number of samples (Cui et al., "Class-Balanced Loss
        # Based on Effective Number of Samples", CVPR 2019).
        # beta close to 1.0 gives more weight to rare classes; 0.9999
        # is the recommended default for highly imbalanced datasets.
        beta = 0.9999
        effective = 1.0 - np.power(beta, counts)
        alpha = (1.0 - beta) / effective
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalise so mean weight = 1.0
    alpha = alpha / alpha.mean()
    return torch.tensor(alpha, dtype=torch.float32)
