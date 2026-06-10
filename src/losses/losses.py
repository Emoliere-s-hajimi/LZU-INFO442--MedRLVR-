"""Losses for BrainTTNet.

The cleaned cohort uses BraTS-style multi-label segmentation — the
``label`` array is (3, D, H, W) uint8 with nested binary channels
[WT, TC, ET]. Every loss in this file operates on the same convention:
3-channel sigmoid predictions vs 3-channel binary targets.

Components:

  * ``MultiLabelDice``         — per-channel Dice on sigmoid outputs.
  * ``LogVolumeWeightedDice``  — per-channel Dice rescaled by log(1 + V),
                                 from Finding 2 (4-order volume long tail).
  * ``FocalBCEWithLogits``     — per-voxel focal loss for the segmentation
                                 head when used standalone.
  * ``NestingPenalty``         — soft enforcement of WT ⊇ TC ⊇ ET
                                 (Finding 6).
  * ``TopologyChiRegulariser`` — pulls the predicted χ scalar toward
                                 ``E[χ | class]`` (Finding 5).
  * ``FocalLoss``              — recurrence vs necrosis focal loss
                                 (Finding 12).
  * ``DeepSupervisionWrapper`` — auxiliary multi-scale seg supervision.
  * ``MultiTaskLoss``          — combines everything into a single
                                 trainable objective.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-label segmentation losses
# ---------------------------------------------------------------------------

class MultiLabelDice(nn.Module):
    """Per-channel Dice on sigmoid outputs.

    Both ``logits`` and ``target`` are (B, C, D, H, W). ``target`` is
    binary uint8/float. Returns mean over channels.
    """

    def __init__(self, smooth: float = 1.0,
                 channel_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.smooth = smooth
        self.register_buffer(
            "channel_weights",
            torch.tensor(channel_weights, dtype=torch.float32)
            if channel_weights is not None else torch.empty(0),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        dims = (0, 2, 3, 4)
        intersection = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss_per_c = 1.0 - dice
        if self.channel_weights.numel() == loss_per_c.numel():
            w = self.channel_weights.to(loss_per_c.device)
            return (w * loss_per_c).sum() / w.sum()
        return loss_per_c.mean()


class LogVolumeWeightedDice(nn.Module):
    """Per-channel Dice weighted by ``log(1 + lesion_volume)`` per case.

    The reweighting is computed *online* from the target lesion volume,
    so cases with small lesions contribute less raw Dice but more
    information per unit gradient. From Finding 2.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        target = target.float()
        # Per-case, per-channel
        dims = (2, 3, 4)
        intersection = (probs * target).sum(dim=dims)
        union = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        weights = torch.log1p(target.sum(dim=dims))                        # (B, C)
        weights = weights / (weights.sum() + 1e-6)
        return (weights * (1.0 - dice)).sum()


class FocalBCEWithLogits(nn.Module):
    """Focal BCE for the segmentation head — useful when most voxels are
    background. ``gamma`` defaults match Finding 12's focal-loss settings."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * target + (1 - probs) * (1 - target)
        focal = (self.alpha * (1 - p_t) ** self.gamma) * bce
        return focal.mean()


class NestingPenalty(nn.Module):
    """Soft enforcement of WT ⊇ TC ⊇ ET on sigmoid probabilities.

    For nested probabilities the constraint is ``p_WT ≥ p_TC ≥ p_ET``;
    we penalise positive ``max(0, p_TC - p_WT)`` and ``max(0, p_ET - p_TC)``
    per voxel. Finding 6.
    """

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        wt, tc, et = probs[:, 0], probs[:, 1], probs[:, 2]
        violation = torch.relu(tc - wt) + torch.relu(et - tc)
        return self.weight * violation.mean()


# ---------------------------------------------------------------------------
# Topology χ regulariser
# ---------------------------------------------------------------------------

class TopologyChiRegulariser(nn.Module):
    """Pulls the network-predicted χ scalar toward the class-conditional
    cohort mean (Finding 5: E[χ|recur] = +4, E[χ|necr] = −24).

    The mapping is configurable via ``per_class_chi``; missing entries
    fall back to ``default_chi`` (the cohort median of −3).
    """

    def __init__(self, per_class_chi: Optional[Dict[int, float]] = None,
                 default_chi: float = -3.0, weight: float = 0.05,
                 scale: float = 100.0) -> None:
        super().__init__()
        self.per_class_chi = per_class_chi or {0: 4.0, 1: -24.0}
        self.default_chi = default_chi
        self.weight = weight
        self.scale = scale  # divide by this so the loss term is O(1)

    def forward(self, chi_pred: torch.Tensor, cls_label: torch.Tensor) -> torch.Tensor:
        if chi_pred is None:
            return torch.zeros((), device=cls_label.device)
        target = torch.full_like(chi_pred, fill_value=self.default_chi)
        for cls_idx, chi in self.per_class_chi.items():
            target[cls_label == cls_idx] = chi
        return self.weight * F.smooth_l1_loss(
            chi_pred / self.scale, target / self.scale, reduction="mean",
        )


# ---------------------------------------------------------------------------
# Classification — focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 class_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None else torch.empty(0),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if self.class_weights.numel() > 0 else None
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        p_t = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Deep supervision + multi-task wrapper
# ---------------------------------------------------------------------------

class DeepSupervisionWrapper(nn.Module):
    """Apply a base segmentation loss to a list of multi-scale logits.

    ``weights[0]`` applies to the highest-resolution head; subsequent
    weights apply to coarser auxiliary outputs.
    """

    def __init__(self, base: nn.Module, weights: Sequence[float] = (1.0, 0.4, 0.3)) -> None:
        super().__init__()
        self.base = base
        self.weights = list(weights)

    def forward(self, outputs: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        w = self.weights[: len(outputs)]
        for o, wi in zip(outputs, w):
            tgt = target
            if o.shape[2:] != target.shape[2:]:
                tgt = F.interpolate(target.float(), size=o.shape[2:], mode="nearest")
            total = total + wi * self.base(o, tgt)
        return total / sum(w)


class MultiTaskLoss(nn.Module):
    """Composite training objective.

    Combines:
        cls_loss         FocalLoss on (B, 2) logits vs (B,) integer labels
        seg_loss         DeepSupervisionWrapper around MultiLabelDice + FocalBCE
        log_volume_loss  LogVolumeWeightedDice on the main seg head (no aux)
        nesting_loss     NestingPenalty on the main seg head
        chi_loss         TopologyChiRegulariser on the predicted χ scalar

    All sub-losses are computed and returned for logging. Only the
    weighted sum is back-propagated.
    """

    def __init__(
        self,
        cls_loss: nn.Module,
        seg_dice_loss: nn.Module,
        seg_focal_loss: Optional[nn.Module] = None,
        log_volume_loss: Optional[nn.Module] = None,
        nesting_loss: Optional[nn.Module] = None,
        chi_loss: Optional[nn.Module] = None,
        cls_weight: float = 1.0,
        seg_weight: float = 1.0,
        log_volume_weight: float = 0.5,
        ds_weights: Sequence[float] = (1.0, 0.4, 0.3),
    ) -> None:
        super().__init__()
        self.cls_loss = cls_loss
        self.seg_dice_loss = DeepSupervisionWrapper(seg_dice_loss, weights=ds_weights)
        self.seg_focal_loss = (
            DeepSupervisionWrapper(seg_focal_loss, weights=ds_weights)
            if seg_focal_loss is not None else None
        )
        self.log_volume_loss = log_volume_loss
        self.nesting_loss = nesting_loss
        self.chi_loss = chi_loss
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.log_volume_weight = log_volume_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # Classification
        loss_cls = self.cls_loss(outputs["cls"], targets["label"])

        # Segmentation — multi-scale Dice + optional focal BCE
        seg_logits: List[torch.Tensor] = [outputs["seg"]] + list(outputs.get("seg_aux", []))
        seg_target = targets.get("seg")
        zero = torch.zeros((), device=loss_cls.device)
        loss_seg_dice = self.seg_dice_loss(seg_logits, seg_target) if seg_target is not None else zero
        loss_seg_focal = (
            self.seg_focal_loss(seg_logits, seg_target)
            if (self.seg_focal_loss is not None and seg_target is not None) else zero
        )

        loss_log_vol = (
            self.log_volume_loss(outputs["seg"], seg_target)
            if (self.log_volume_loss is not None and seg_target is not None) else zero
        )

        loss_nest = (
            self.nesting_loss(outputs["seg"]) if self.nesting_loss is not None else zero
        )

        loss_chi = (
            self.chi_loss(outputs.get("chi_pred"), targets["label"])
            if (self.chi_loss is not None and outputs.get("chi_pred") is not None)
            else zero
        )

        total = (
            self.cls_weight * loss_cls
            + self.seg_weight * (loss_seg_dice + loss_seg_focal)
            + self.log_volume_weight * loss_log_vol
            + loss_nest
            + loss_chi
        )

        return {
            "total": total,
            "cls": loss_cls.detach(),
            "seg_dice": loss_seg_dice.detach(),
            "seg_focal": loss_seg_focal.detach(),
            "log_vol": loss_log_vol.detach(),
            "nest": loss_nest.detach(),
            "chi": loss_chi.detach(),
        }


# Backward-compatible alias retained for older callers
DiceLoss = MultiLabelDice
DiceCELoss = MultiLabelDice
