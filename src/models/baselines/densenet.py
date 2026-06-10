"""3D DenseNet baseline (DenseNet121).

Reference:
  Huang et al., "Densely Connected Convolutional Networks", CVPR 2017.

Each DenseLayer concatenates its output to all previous layers' outputs
within the same DenseBlock, encouraging feature reuse.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D, _gn


class DenseLayer3D(nn.Module):
    """BN-ReLU-1x1conv-BN-ReLU-3x3conv, output concatenated to input."""
    def __init__(self, in_c: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        inter_c = bn_size * growth_rate
        self.norm1 = _gn(in_c)
        self.conv1 = nn.Conv3d(in_c, inter_c, 1, bias=False)
        self.norm2 = _gn(inter_c)
        self.conv2 = nn.Conv3d(inter_c, growth_rate, 3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.norm1(x))
        y = self.conv1(y)
        y = self.act(self.norm2(y))
        y = self.conv2(y)
        return torch.cat([x, y], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_c: int, num_layers: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        c = in_c
        for _ in range(num_layers):
            self.layers.append(DenseLayer3D(c, growth_rate, bn_size))
            c += growth_rate
        self.out_channels = c

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer3D(nn.Module):
    """BN-ReLU-1x1conv-2x2pool — halves channels and spatial dims."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm = _gn(in_c)
        self.conv = nn.Conv3d(in_c, out_c, 1, bias=False)
        self.pool = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.gelu(self.norm(x))))


class DenseNet3D(BaselineWrapper):
    """3D DenseNet121 with multi-task heads.

    Block config (DenseNet121): [6, 12, 24, 16] layers per block."""

    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, growth_rate: int = 12,
                 block_config=(6, 12, 24, 16), init_features: int = 32):
        super().__init__()

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_features, 7, stride=2, padding=3, bias=False),
            _gn(init_features), nn.GELU(),
            nn.MaxPool3d(3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        c = init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        skip_channels = []   # for seg decoder

        for i, n_layers in enumerate(block_config):
            block = DenseBlock3D(c, n_layers, growth_rate)
            self.dense_blocks.append(block)
            c = block.out_channels
            skip_channels.append(c)
            if i != len(block_config) - 1:
                self.transitions.append(TransitionLayer3D(c, c // 2))
                c = c // 2

        self.bottleneck_c = c
        self.skip_channels = skip_channels

        # Seg decoder — 4-stage upsampling
        self.dec_layers = nn.ModuleList()
        prev_c = c
        # Bring back to size of largest dense block (16× upsample needed)
        for tgt_c in [32, 32, 32, 32]:
            self.dec_layers.append(nn.Sequential(
                nn.ConvTranspose3d(prev_c, tgt_c, 2, stride=2),
                nn.Conv3d(tgt_c, tgt_c, 3, padding=1), _gn(tgt_c), nn.GELU(),
            ))
            prev_c = tgt_c

        self.cls_head = ClassificationHead3D(c, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(prev_c, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        x = self.stem(image)
        skips = []
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        cls = self.cls_head(x)

        # Decode
        for dec in self.dec_layers:
            x = dec(x)
        seg = self.seg_head(x, target_size)
        return {"seg": seg, "cls": cls}


def build_densenet121(config: Optional[Dict] = None) -> DenseNet3D:
    cfg = (config or {}).get("model", {})
    return DenseNet3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        growth_rate=cfg.get("growth_rate", 12),
        block_config=cfg.get("block_config", (6, 12, 24, 16)),
        init_features=cfg.get("init_features", 32),
    )
