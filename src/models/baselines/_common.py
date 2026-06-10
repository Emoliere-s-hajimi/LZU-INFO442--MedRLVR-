"""Baseline models — common interface.

Each baseline:
  - Accepts ``(B, 4, H, W, D)`` 4-modality input
  - Produces a dict ``{seg: (B, 3, H, W, D), cls: (B, 2)}``
  - Has a single ``forward(image, missing_mask=None, aux_features=None, return_aux=False)`` signature
  - Reports total parameter count via ``param_count()``

This common contract lets the smoke-test and training scripts swap baselines
in / out with zero code change.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, max_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, min(max_groups, c // 4)), c)


class BaselineWrapper(nn.Module):
    """Common API for every baseline. Subclasses fill in ``self.backbone``,
    ``self.cls_head``, ``self.seg_head``."""

    def __init__(self) -> None:
        super().__init__()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        image: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
        aux_features: Optional[torch.Tensor] = None,
        return_aux: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ClassificationHead3D(nn.Module):
    """GAP → MLP → cls logits."""
    def __init__(self, in_c: int, n_classes: int = 2, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(x))


class SegmentationHead3D(nn.Module):
    """1x1x1 conv → seg logits, with output upsampling to input size."""
    def __init__(self, in_c: int, n_classes: int = 3):
        super().__init__()
        self.head = nn.Conv3d(in_c, n_classes, 1)

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        out = self.head(x)
        if out.shape[2:] != tuple(target_size):
            out = F.interpolate(out, size=target_size, mode="trilinear", align_corners=False)
        return out
