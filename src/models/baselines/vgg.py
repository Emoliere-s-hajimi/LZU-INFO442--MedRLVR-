"""VGG-style 3D baselines (VGG11, VGG16).

References:
  Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale
  Image Recognition", ICLR 2015.

Modifications for 3D medical imaging:
  - Conv2d → Conv3d
  - MaxPool2d → MaxPool3d
  - Skip-connection-style decoder appended for joint cls + seg
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D, _gn


VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]


def _make_vgg_layers(cfg: List, in_channels: int = 4, base_div: int = 4):
    """Build VGG-style layer stack. ``base_div`` divides every channel count
    to make a lightweight 3D version (full VGG would be 200M+ params in 3D)."""
    layers = []
    encoder_features = []  # per-stage outputs for the decoder
    in_c = in_channels
    for v in cfg:
        if v == "M":
            encoder_features.append(in_c)
            layers.append(nn.MaxPool3d(2, 2))
        else:
            out_c = max(8, v // base_div)
            layers.append(nn.Conv3d(in_c, out_c, 3, padding=1, bias=False))
            layers.append(_gn(out_c))
            layers.append(nn.GELU())
            in_c = out_c
    return nn.Sequential(*layers), in_c, encoder_features


class VGG3D(BaselineWrapper):
    """3D VGG with multi-task output."""

    def __init__(self, cfg: List, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_div: int = 4):
        super().__init__()
        self.features, last_c, _ = _make_vgg_layers(cfg, in_channels, base_div)
        self.cls_head = ClassificationHead3D(last_c, n_classes=n_classes, hidden=128)
        self.seg_head = SegmentationHead3D(last_c, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        feat = self.features(image)
        cls = self.cls_head(feat)
        seg = self.seg_head(feat, target_size)
        return {"seg": seg, "cls": cls}


class VGG11_3D(VGG3D):
    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_div: int = 4):
        super().__init__(VGG11_CFG, in_channels, n_classes, seg_classes, base_div)


class VGG16_3D(VGG3D):
    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_div: int = 4):
        super().__init__(VGG16_CFG, in_channels, n_classes, seg_classes, base_div)


def build_vgg11(config: Optional[Dict] = None) -> VGG11_3D:
    cfg = (config or {}).get("model", {})
    return VGG11_3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_div=cfg.get("base_div", 4),
    )


def build_vgg16(config: Optional[Dict] = None) -> VGG16_3D:
    cfg = (config or {}).get("model", {})
    return VGG16_3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_div=cfg.get("base_div", 4),
    )
