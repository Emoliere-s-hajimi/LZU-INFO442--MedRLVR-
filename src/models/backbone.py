"""Generic 3D encoder/decoder blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, channels // 8), channels)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False), _gn(out_channels), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False), _gn(out_channels), nn.GELU()
        )
        self.shortcut = (
            nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, bias=False), _gn(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.gelu(self.gamma * x + identity)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False), _gn(out_channels), nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False), _gn(out_channels), nn.GELU()
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.gamma * x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False), _gn(out_channels), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        hidden = max(8, in_channels // 2)
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        out = self.head(x)
        if target_size is not None and out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="trilinear", align_corners=False)
        return out


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x))
