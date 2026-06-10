"""3-D U-Net backbone for BrainTT.

A clean encoder/decoder that the prior modules plug into. The backbone
itself has no medical priors baked in — it is the shape and feature
ladder; ``src.models.priors`` provides the inductive biases.

Layout::

    Input  → stem → enc1 → enc2 → enc3 → bottleneck
                                                    ↓
                                              dec3 ← skip(enc3)
                                              dec2 ← skip(enc2)
                                              dec1 ← skip(enc1)
                                                    ↓
                                              seg_head  (3-channel sigmoid: WT, TC, ET)
                                              cls_head  (2-class softmax: recurrence vs necrosis)

Two block flavours are exposed:

    * ``ResidualBlock3D``           — isotropic 3×3×3, used in the encoder
    * ``AnisotropicResidualBlock3D`` — 3×3×1 + 1×1×3 factorisation, used in
                                       the decoder per Finding 3 (no tumour
                                       is spherical, so isotropic decoder
                                       kernels under-segment elongated
                                       lesions).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm/activation helpers
# ---------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, channels // 8), channels)


def _conv_bn_act(in_c: int, out_c: int, kernel: Tuple[int, ...] = (3, 3, 3),
                 padding: Tuple[int, ...] = (1, 1, 1)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel, padding=padding, bias=False),
        _gn(out_c),
        nn.GELU(),
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock3D(nn.Module):
    """Isotropic 3×3×3 residual block — used in the encoder."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = _conv_bn_act(in_c, out_c)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            _gn(out_c),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv3d(in_c, out_c, 1, bias=False), _gn(out_c))
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv2(self.conv1(x)) + self.shortcut(x))


class AnisotropicResidualBlock3D(nn.Module):
    """Decoder block with axis-asymmetric factorisation (3×3×1 + 1×1×3).

    Motivation (Finding 3): the cohort has median sphericity 0.362 and
    bbox fill 0.241 — lesions are highly elongated along an arbitrary
    axis. A factorised kernel doubles the receptive field along two axes
    at no FLOP cost.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_c, out_c, (3, 3, 1), padding=(1, 1, 0), bias=False),
            _gn(out_c),
            nn.GELU(),
            nn.Conv3d(out_c, out_c, (1, 1, 3), padding=(0, 0, 1), bias=False),
            _gn(out_c),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv3d(in_c, out_c, 1, bias=False), _gn(out_c))
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spatial(x) + self.shortcut(x))


class Downsample(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = _conv_bn_act(in_c, out_c, kernel=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.Conv3d(out_c, out_c, 2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


class Upsample(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class NestedSegmentationHead(nn.Module):
    """Outputs 3 channels [WT, TC, ET] as nested logits.

    The forward pass returns *raw* logits per channel; the nesting
    constraint (Finding 6) is enforced by ``losses.NestingPenalty`` rather
    than by a structural ``min`` chain so gradients still reach every
    channel cleanly.
    """

    def __init__(self, in_c: int, num_classes: int = 3) -> None:
        super().__init__()
        hidden = max(8, in_c // 2)
        self.head = nn.Sequential(
            nn.Conv3d(in_c, hidden, 3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        out = self.head(x)
        if target_size is not None and out.shape[2:] != tuple(target_size):
            out = F.interpolate(out, size=target_size, mode="trilinear", align_corners=False)
        return out


class ClassificationHead(nn.Module):
    """GAP → MLP → 2-class logits.

    Accepts an optional ``aux_features`` tensor concatenated to the
    pooled vector. We feed the T1ce in/out ratio scalar (Finding 8,
    Cohen's d = 0.94) and the predicted lesion volume here.
    """

    def __init__(self, in_c: int, num_classes: int = 2,
                 aux_dim: int = 0, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.aux_dim = aux_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_c + aux_dim, in_c // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_c // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor = None) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        if self.aux_dim > 0:
            if aux is None:
                aux = torch.zeros(pooled.shape[0], self.aux_dim,
                                  device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, aux], dim=1)
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# U-Net backbone
# ---------------------------------------------------------------------------

class UNetBackbone(nn.Module):
    """4-stage 3-D U-Net with isotropic encoder + anisotropic decoder.

    Returns a dict so the surrounding prior modules can hook into each
    stage cleanly::

        {
          "enc1": (B, c1, D,    H,    W),
          "enc2": (B, c2, D/2,  H/2,  W/2),
          "enc3": (B, c3, D/4,  H/4,  W/4),
          "bottleneck": (B, c4, D/8,  H/8,  W/8),
          "dec3": (B, c3, D/4,  H/4,  W/4),
          "dec2": (B, c2, D/2,  H/2,  W/2),
          "dec1": (B, c1, D,    H,    W),
        }
    """

    def __init__(self, stem_channels: int = 32, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.channels = (c1, c2, c3, c4)

        self.stem = _conv_bn_act(stem_channels, c1)

        # Encoder
        self.enc1 = ResidualBlock3D(c1, c1)
        self.down1 = Downsample(c1, c2)
        self.enc2 = ResidualBlock3D(c2, c2)
        self.down2 = Downsample(c2, c3)
        self.enc3 = ResidualBlock3D(c3, c3)
        self.down3 = Downsample(c3, c4)
        self.bottleneck = ResidualBlock3D(c4, c4)

        # Decoder (anisotropic refinement)
        self.up3 = Upsample(c4, c3)
        self.dec3 = AnisotropicResidualBlock3D(c3 + c3, c3)
        self.up2 = Upsample(c3, c2)
        self.dec2 = AnisotropicResidualBlock3D(c2 + c2, c2)
        self.up1 = Upsample(c2, c1)
        self.dec1 = AnisotropicResidualBlock3D(c1 + c1, c1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stem = self.stem(x)
        enc1 = self.enc1(stem)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        bottleneck = self.bottleneck(self.down3(enc3))

        dec3 = self.dec3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return {
            "enc1": enc1, "enc2": enc2, "enc3": enc3,
            "bottleneck": bottleneck,
            "dec3": dec3, "dec2": dec2, "dec1": dec1,
        }
