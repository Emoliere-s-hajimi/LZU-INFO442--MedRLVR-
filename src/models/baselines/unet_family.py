"""U-Net family 3D baselines.

References:
  - U-Net: Ronneberger et al., MICCAI 2015
  - 3D U-Net: Çiçek et al., MICCAI 2016
  - Attention U-Net: Oktay et al., 2018
  - UNet++ (Nested U-Net): Zhou et al., DLMIA 2018
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D, _gn


class ConvBlock3D(nn.Module):
    """Conv-GN-GELU × 2 with optional residual."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            _gn(out_c), nn.GELU(),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            _gn(out_c), nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Standard 3D U-Net
# ---------------------------------------------------------------------------

class UNet3D(BaselineWrapper):
    """Standard 4-stage 3D U-Net."""

    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_c: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.enc1 = ConvBlock3D(in_channels, c1)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(c1, c2)
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.down3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(c3, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock3D(c3 * 2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock3D(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock3D(c1 * 2, c1)

        self.cls_head = ClassificationHead3D(c4, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(c1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        e1 = self.enc1(image)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        cls = self.cls_head(e4)
        seg = self.seg_head(d1, target_size)
        return {"seg": seg, "cls": cls}


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------

class AttentionGate3D(nn.Module):
    """Gating signal × skip feature, with learned soft attention."""
    def __init__(self, gate_c: int, skip_c: int, inter_c: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(gate_c, inter_c, 1, bias=False), _gn(inter_c))
        self.W_x = nn.Sequential(nn.Conv3d(skip_c, inter_c, 1, bias=False), _gn(inter_c))
        self.psi = nn.Sequential(nn.Conv3d(inter_c, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, gate, skip):
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        psi = self.psi(F.gelu(g1 + x1))
        return skip * psi


class AttentionUNet3D(BaselineWrapper):
    """U-Net with attention gates at every skip connection."""

    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_c: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.enc1 = ConvBlock3D(in_channels, c1)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(c1, c2)
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(c2, c3)
        self.down3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(c3, c4)

        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.att3 = AttentionGate3D(c3, c3, c3 // 2)
        self.dec3 = ConvBlock3D(c3 * 2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.att2 = AttentionGate3D(c2, c2, c2 // 2)
        self.dec2 = ConvBlock3D(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.att1 = AttentionGate3D(c1, c1, c1 // 2)
        self.dec1 = ConvBlock3D(c1 * 2, c1)

        self.cls_head = ClassificationHead3D(c4, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(c1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        e1 = self.enc1(image)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        u3 = self.up3(e4)
        d3 = self.dec3(torch.cat([u3, self.att3(u3, e3)], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.att2(u2, e2)], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.att1(u1, e1)], dim=1))

        cls = self.cls_head(e4)
        seg = self.seg_head(d1, target_size)
        return {"seg": seg, "cls": cls}


# ---------------------------------------------------------------------------
# UNet++ (Nested U-Net)
# ---------------------------------------------------------------------------

class NestedUNet3D(BaselineWrapper):
    """UNet++ — dense skip connections via nested decoder paths.

    For tractability we use a 3-stage version of UNet++ (not 4-stage)."""

    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_c: int = 16):
        super().__init__()
        c1, c2, c3 = base_c, base_c * 2, base_c * 4

        self.conv0_0 = ConvBlock3D(in_channels, c1)
        self.conv1_0 = ConvBlock3D(c1, c2)
        self.conv2_0 = ConvBlock3D(c2, c3)

        self.conv0_1 = ConvBlock3D(c1 + c2, c1)
        self.conv1_1 = ConvBlock3D(c2 + c3, c2)
        self.conv0_2 = ConvBlock3D(c1 * 2 + c2, c1)

        self.pool = nn.MaxPool3d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)

        self.cls_head = ClassificationHead3D(c3, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(c1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        x0_0 = self.conv0_0(image)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        cls = self.cls_head(x2_0)
        seg = self.seg_head(x0_2, target_size)
        return {"seg": seg, "cls": cls}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_unet(config: Optional[Dict] = None) -> UNet3D:
    cfg = (config or {}).get("model", {})
    return UNet3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
    )


def build_attention_unet(config: Optional[Dict] = None) -> AttentionUNet3D:
    cfg = (config or {}).get("model", {})
    return AttentionUNet3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
    )


def build_nested_unet(config: Optional[Dict] = None) -> NestedUNet3D:
    cfg = (config or {}).get("model", {})
    return NestedUNet3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
    )
