"""3D ResNet baselines (ResNet10, ResNet50, MResNet).

References:
  - ResNet: He et al., CVPR 2016
  - 3D ResNet (Med3D): Chen et al., 2019 — pretrained on medical 3D volumes
  - MResNet: Medical-ResNet, adapted with multi-scale receptive fields

Variants:
  - ResNet10: 4 stages, BasicBlock × [1, 1, 1, 1]
  - ResNet50: 4 stages, Bottleneck × [3, 4, 6, 3]
  - MResNet:  ResNet18 backbone + multi-scale lateral connections
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D, _gn


class BasicBlock3D(nn.Module):
    """ResNet BasicBlock: Conv-BN-ReLU-Conv-BN + residual."""
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _gn(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=False)
        self.norm2 = _gn(out_c)
        self.act = nn.GELU()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False), _gn(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class Bottleneck3D(nn.Module):
    """ResNet Bottleneck: 1x1 → 3x3 → 1x1 + residual."""
    expansion = 4

    def __init__(self, in_c: int, mid_c: int, stride: int = 1):
        super().__init__()
        out_c = mid_c * self.expansion
        self.conv1 = nn.Conv3d(in_c, mid_c, 1, bias=False)
        self.norm1 = _gn(mid_c)
        self.conv2 = nn.Conv3d(mid_c, mid_c, 3, stride=stride, padding=1, bias=False)
        self.norm2 = _gn(mid_c)
        self.conv3 = nn.Conv3d(mid_c, out_c, 1, bias=False)
        self.norm3 = _gn(out_c)
        self.act = nn.GELU()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False), _gn(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return self.act(out + identity)


def _make_layer(block, in_c, mid_c, num_blocks, stride):
    """Build a sequence of blocks for one stage."""
    layers = [block(in_c, mid_c, stride=stride)]
    out_c = mid_c * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(out_c, mid_c, stride=1))
    return nn.Sequential(*layers), out_c


class ResNet3D(BaselineWrapper):
    """Configurable 3D ResNet with multi-task heads + lightweight seg decoder."""

    def __init__(self, block, layers: List[int], in_channels: int = 4,
                 n_classes: int = 2, seg_classes: int = 3, base_c: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, 7, stride=2, padding=3, bias=False),
            _gn(c1), nn.GELU(),
            nn.MaxPool3d(3, stride=2, padding=1),
        )

        self.layer1, out1 = _make_layer(block, c1, c1, layers[0], stride=1)
        self.layer2, out2 = _make_layer(block, out1, c2, layers[1], stride=2)
        self.layer3, out3 = _make_layer(block, out2, c3, layers[2], stride=2)
        self.layer4, out4 = _make_layer(block, out3, c4, layers[3], stride=2)

        # Lightweight seg decoder (upsample 16x with 4 stages)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(out4, out3, 2, stride=2),
            nn.Conv3d(out3, out3, 3, padding=1), _gn(out3), nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(out3, out2, 2, stride=2),
            nn.Conv3d(out2, out2, 3, padding=1), _gn(out2), nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(out2, out1, 2, stride=2),
            nn.Conv3d(out1, out1, 3, padding=1), _gn(out1), nn.GELU(),
        )
        self.dec0 = nn.Sequential(
            nn.ConvTranspose3d(out1, out1, 2, stride=2),
            nn.Conv3d(out1, out1, 3, padding=1), _gn(out1), nn.GELU(),
        )

        self.cls_head = ClassificationHead3D(out4, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(out1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        s = self.stem(image)
        l1 = self.layer1(s)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        d3 = self.dec3(l4)
        if d3.shape[2:] != l3.shape[2:]:
            d3 = F.interpolate(d3, size=l3.shape[2:], mode="trilinear", align_corners=False)
        d3 = d3 + l3

        d2 = self.dec2(d3)
        if d2.shape[2:] != l2.shape[2:]:
            d2 = F.interpolate(d2, size=l2.shape[2:], mode="trilinear", align_corners=False)
        d2 = d2 + l2

        d1 = self.dec1(d2)
        if d1.shape[2:] != l1.shape[2:]:
            d1 = F.interpolate(d1, size=l1.shape[2:], mode="trilinear", align_corners=False)
        d1 = d1 + l1

        d0 = self.dec0(d1)

        cls = self.cls_head(l4)
        seg = self.seg_head(d0, target_size)
        return {"seg": seg, "cls": cls}


class ResNet10_3D(ResNet3D):
    def __init__(self, in_channels=4, n_classes=2, seg_classes=3, base_c=16):
        super().__init__(BasicBlock3D, [1, 1, 1, 1], in_channels, n_classes, seg_classes, base_c)


class ResNet50_3D(ResNet3D):
    def __init__(self, in_channels=4, n_classes=2, seg_classes=3, base_c=16):
        super().__init__(Bottleneck3D, [3, 4, 6, 3], in_channels, n_classes, seg_classes, base_c)


# ---------------------------------------------------------------------------
# MResNet — multi-scale ResNet with lateral connections
# ---------------------------------------------------------------------------

class MResNet3D(BaselineWrapper):
    """Multi-scale ResNet: ResNet18 backbone + lateral multi-scale fusion.

    Inspired by HRNet and Pyramid Networks — instead of strict top-down
    decoder, we maintain multi-resolution feature maps in parallel and
    fuse them via lateral connections."""

    def __init__(self, in_channels=4, n_classes=2, seg_classes=3, base_c=16):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        # Stem (single downsample, not 4x like ResNet)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, 7, stride=2, padding=3, bias=False),
            _gn(c1), nn.GELU(),
        )

        # 4 parallel scale branches via BasicBlock
        self.branch1 = BasicBlock3D(c1, c1)
        self.down1 = nn.Conv3d(c1, c2, 2, stride=2, bias=False)
        self.branch2 = BasicBlock3D(c2, c2)
        self.down2 = nn.Conv3d(c2, c3, 2, stride=2, bias=False)
        self.branch3 = BasicBlock3D(c3, c3)
        self.down3 = nn.Conv3d(c3, c4, 2, stride=2, bias=False)
        self.branch4 = BasicBlock3D(c4, c4)

        # Lateral multi-scale fusion modules (1×1×1 channel adapters)
        self.lat4_to_1 = nn.Conv3d(c4, c1, 1)
        self.lat3_to_1 = nn.Conv3d(c3, c1, 1)
        self.lat2_to_1 = nn.Conv3d(c2, c1, 1)

        # Final fusion conv
        self.fuse = nn.Sequential(
            nn.Conv3d(c1 * 4, c1, 3, padding=1, bias=False), _gn(c1), nn.GELU(),
        )

        self.cls_head = ClassificationHead3D(c4, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(c1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        s = self.stem(image)
        b1 = self.branch1(s)
        b2 = self.branch2(self.down1(b1))
        b3 = self.branch3(self.down2(b2))
        b4 = self.branch4(self.down3(b3))

        # Upsample everything to b1's resolution and concatenate
        def upto(x, target):
            if x.shape[2:] != target.shape[2:]:
                return F.interpolate(x, size=target.shape[2:], mode="trilinear",
                                      align_corners=False)
            return x

        l4 = upto(self.lat4_to_1(b4), b1)
        l3 = upto(self.lat3_to_1(b3), b1)
        l2 = upto(self.lat2_to_1(b2), b1)
        fused = self.fuse(torch.cat([b1, l2, l3, l4], dim=1))

        cls = self.cls_head(b4)
        seg = self.seg_head(fused, target_size)
        return {"seg": seg, "cls": cls}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_resnet10(config: Optional[Dict] = None) -> ResNet10_3D:
    cfg = (config or {}).get("model", {})
    return ResNet10_3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
    )


def build_resnet50(config: Optional[Dict] = None) -> ResNet50_3D:
    cfg = (config or {}).get("model", {})
    return ResNet50_3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
    )


def build_mresnet(config: Optional[Dict] = None) -> MResNet3D:
    cfg = (config or {}).get("model", {})
    return MResNet3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
    )
