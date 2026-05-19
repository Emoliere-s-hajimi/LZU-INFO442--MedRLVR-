"""Losses for the multi-task glioma classification / segmentation network."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, C).permute(0, 4, 1, 2, 3).float()
        intersection = (probs * one_hot).sum(dim=(0, 2, 3, 4))
        union = probs.sum(dim=(0, 2, 3, 4)) + one_hot.sum(dim=(0, 2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, class_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.empty(0),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if self.class_weights.numel() > 0 else None
        ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
        p_t = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1).clamp_max(logits.shape[1] - 1)).squeeze(1)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


class DiceCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * F.cross_entropy(logits, target)


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, base: nn.Module, weights: Sequence[float] = (1.0, 0.4, 0.3)) -> None:
        super().__init__()
        self.base = base
        self.weights = list(weights)

    def forward(self, outputs: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        w = self.weights[: len(outputs)]
        for o, wi in zip(outputs, w):
            tgt = target
            if o.shape[2:] != target.shape[1:]:
                tgt = F.interpolate(target.unsqueeze(1).float(), size=o.shape[2:], mode="nearest").squeeze(1).long()
            total = total + wi * self.base(o, tgt)
        return total / sum(w)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        cls_loss: nn.Module,
        seg_loss: nn.Module,
        cls_weight: float = 1.0,
        seg_weight: float = 1.0,
        ds_weights: Sequence[float] = (1.0, 0.4, 0.3),
    ) -> None:
        super().__init__()
        self.cls_loss = cls_loss
        self.seg_loss = DeepSupervisionWrapper(seg_loss, weights=ds_weights)
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_cls = self.cls_loss(outputs["cls"], targets["label"])
        seg_outputs: List[torch.Tensor] = [outputs["seg"]] + list(outputs.get("seg_aux", []))
        loss_seg = self.seg_loss(seg_outputs, targets["seg"]) if "seg" in targets else torch.zeros_like(loss_cls)
        total = self.cls_weight * loss_cls + self.seg_weight * loss_seg
        return {"total": total, "cls": loss_cls.detach(), "seg": loss_seg.detach()}
