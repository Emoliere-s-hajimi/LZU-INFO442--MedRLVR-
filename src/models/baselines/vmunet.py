"""3D VM-UNet (Vision Mamba U-Net) baseline.

Reference:
  - Mamba: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces", 2023
  - VM-UNet: Ruan & Xiang, "VM-UNet: Vision Mamba UNet for Medical Image
    Segmentation", 2024

Implementation note:
  Full SSM (selective scan) requires a CUDA kernel. To keep the baseline
  pure PyTorch and runnable on CPU for the smoke test, we implement a
  simplified "MambaBlock" using:
    - Depthwise conv along each spatial axis (3 directional scans)
    - Linear gating (SiLU-style)
    - 1D residual state via cumulative sum (approximates state-space recurrence)

  This captures the spirit of VM-UNet (multi-directional sequence modeling
  on flattened volume tokens) while staying portable.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D, _gn


class SimplifiedMambaBlock3D(nn.Module):
    """Approximated selective-scan over 3 spatial axes.

    For each axis: depthwise causal conv + linear gate + sigmoid mask.
    Aggregated across axes to mimic multi-directional scanning.
    """

    def __init__(self, channels: int, expand: int = 2):
        super().__init__()
        inner = channels * expand
        self.norm = _gn(channels)
        self.in_proj = nn.Conv3d(channels, inner * 2, 1)  # for x and gate

        # Three directional depthwise convs (one per axis)
        self.dw_d = nn.Conv3d(inner, inner, (5, 1, 1), padding=(2, 0, 0),
                              groups=inner, bias=False)
        self.dw_h = nn.Conv3d(inner, inner, (1, 5, 1), padding=(0, 2, 0),
                              groups=inner, bias=False)
        self.dw_w = nn.Conv3d(inner, inner, (1, 1, 5), padding=(0, 0, 2),
                              groups=inner, bias=False)

        # Cumulative state (approx SSM recurrence)
        self.state_proj = nn.Conv3d(inner, inner, 1, bias=False)
        self.out_proj = nn.Conv3d(inner, channels, 1)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x_proj = self.in_proj(x)
        x_main, gate = x_proj.chunk(2, dim=1)

        # Multi-directional scanning
        s = self.dw_d(x_main) + self.dw_h(x_main) + self.dw_w(x_main)
        s = F.silu(s)

        # Lightweight state via gated cumulative sum along depth axis
        state = self.state_proj(s)
        state = state.cumsum(dim=2) / (
            torch.arange(1, x.shape[2] + 1, device=x.device, dtype=x.dtype)
            .view(1, 1, -1, 1, 1)
        )
        s = s + 0.1 * state

        # Gate and project
        y = self.out_proj(s * torch.sigmoid(gate))
        return identity + y


class VSSLayer3D(nn.Module):
    """Stack of Mamba blocks at one stage."""
    def __init__(self, channels: int, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([SimplifiedMambaBlock3D(channels) for _ in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class VMUNet3D(BaselineWrapper):
    """3D VM-UNet: U-shaped network with Mamba blocks instead of conv at each stage."""

    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_c: int = 16, depths=(2, 2, 2, 2)):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, c1, 3, padding=1, bias=False),
            _gn(c1), nn.GELU(),
        )

        # Encoder
        self.enc1 = VSSLayer3D(c1, depths[0])
        self.down1 = nn.Conv3d(c1, c2, 2, stride=2)
        self.enc2 = VSSLayer3D(c2, depths[1])
        self.down2 = nn.Conv3d(c2, c3, 2, stride=2)
        self.enc3 = VSSLayer3D(c3, depths[2])
        self.down3 = nn.Conv3d(c3, c4, 2, stride=2)
        self.enc4 = VSSLayer3D(c4, depths[3])

        # Decoder
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = VSSLayer3D(c3, depths[2])
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = VSSLayer3D(c2, depths[1])
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = VSSLayer3D(c1, depths[0])

        # Skip connection fusion convs
        self.fuse3 = nn.Conv3d(c3 * 2, c3, 1)
        self.fuse2 = nn.Conv3d(c2 * 2, c2, 1)
        self.fuse1 = nn.Conv3d(c1 * 2, c1, 1)

        self.cls_head = ClassificationHead3D(c4, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(c1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        x = self.patch_embed(image)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        d3 = self.dec3(self.fuse3(torch.cat([self.up3(e4), e3], dim=1)))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d2), e1], dim=1)))

        cls = self.cls_head(e4)
        seg = self.seg_head(d1, target_size)
        return {"seg": seg, "cls": cls}


def build_vmunet(config: Optional[Dict] = None) -> VMUNet3D:
    cfg = (config or {}).get("model", {})
    return VMUNet3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
        depths=cfg.get("depths", (2, 2, 2, 2)),
    )
