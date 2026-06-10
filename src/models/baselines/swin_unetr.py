"""3D Swin-UNETR baseline.

Reference:
  Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic
  Segmentation of Brain Tumors in MRI Images", BraTS 2021.

Implementation:
  Simplified Swin attention — we use 3D shifted-window attention with
  small window sizes to keep memory manageable on the smoke test.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D, _gn


def window_partition_3d(x, window_size: Tuple[int, int, int]):
    """(B, C, D, H, W) → (num_windows*B, C, wD, wH, wW)."""
    B, C, D, H, W = x.shape
    wD, wH, wW = window_size
    x = x.view(B, C, D // wD, wD, H // wH, wH, W // wW, wW)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    windows = windows.view(-1, C, wD, wH, wW)
    return windows


def window_reverse_3d(windows, window_size, B, D, H, W):
    wD, wH, wW = window_size
    C = windows.shape[1]
    x = windows.view(B, D // wD, H // wH, W // wW, C, wD, wH, wW)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    x = x.view(B, C, D, H, W)
    return x


class WindowAttention3D(nn.Module):
    """Multi-head self-attention within a 3D window."""

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int]):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B*nW, C, wD, wH, wW)
        B_, C, wD, wH, wW = x.shape
        N = wD * wH * wW
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B_, N, C)
        qkv = self.qkv(x_flat).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, h, N, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = out.reshape(B_, wD, wH, wW, C).permute(0, 4, 1, 2, 3).contiguous()
        return out


class SwinBlock3D(nn.Module):
    """Swin block: norm + W-MSA + norm + MLP, with optional cyclic shift."""

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int],
                 shift: bool = False, mlp_ratio: float = 2.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size) if shift else (0, 0, 0)

        self.norm1 = _gn(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size)
        self.norm2 = _gn(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, mlp_hidden, 1), nn.GELU(),
            nn.Conv3d(mlp_hidden, dim, 1),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        wD, wH, wW = self.window_size

        # Pad to multiple of window size
        pad_D = (wD - D % wD) % wD
        pad_H = (wH - H % wH) % wH
        pad_W = (wW - W % wW) % wW
        if pad_D or pad_H or pad_W:
            x = F.pad(x, (0, pad_W, 0, pad_H, 0, pad_D))

        identity = x
        x = self.norm1(x)

        # Cyclic shift
        if any(self.shift_size):
            x = torch.roll(x, shifts=tuple(-s for s in self.shift_size), dims=(2, 3, 4))

        # Window partition + attention
        windows = window_partition_3d(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse_3d(windows, self.window_size, B, x.shape[2], x.shape[3], x.shape[4])

        if any(self.shift_size):
            x = torch.roll(x, shifts=self.shift_size, dims=(2, 3, 4))

        # Crop pad
        if pad_D or pad_H or pad_W:
            x = x[:, :, :D, :H, :W]
            identity = identity[:, :, :D, :H, :W]

        x = identity + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage3D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock3D(dim, num_heads, window_size, shift=(i % 2 == 1))
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinUNETR3D(BaselineWrapper):
    """Simplified Swin-UNETR baseline.

    4-stage Swin encoder + CNN decoder."""

    def __init__(self, in_channels: int = 4, n_classes: int = 2,
                 seg_classes: int = 3, base_c: int = 16,
                 depths=(2, 2, 2, 2), num_heads=(2, 4, 4, 8),
                 window_size: Tuple[int, int, int] = (4, 4, 4)):
        super().__init__()
        c1, c2, c3, c4 = base_c, base_c * 2, base_c * 4, base_c * 8

        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, c1, 2, stride=2),
            _gn(c1), nn.GELU(),
        )

        self.stage1 = SwinStage3D(c1, depths[0], num_heads[0], window_size)
        self.down1 = nn.Conv3d(c1, c2, 2, stride=2)
        self.stage2 = SwinStage3D(c2, depths[1], num_heads[1], window_size)
        self.down2 = nn.Conv3d(c2, c3, 2, stride=2)
        self.stage3 = SwinStage3D(c3, depths[2], num_heads[2], window_size)
        self.down3 = nn.Conv3d(c3, c4, 2, stride=2)
        self.stage4 = SwinStage3D(c4, depths[3], num_heads[3], window_size)

        # CNN decoder
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(c3 * 2, c3, 3, padding=1, bias=False), _gn(c3), nn.GELU(),
            nn.Conv3d(c3, c3, 3, padding=1, bias=False), _gn(c3), nn.GELU(),
        )
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(c2 * 2, c2, 3, padding=1, bias=False), _gn(c2), nn.GELU(),
            nn.Conv3d(c2, c2, 3, padding=1, bias=False), _gn(c2), nn.GELU(),
        )
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(c1 * 2, c1, 3, padding=1, bias=False), _gn(c1), nn.GELU(),
            nn.Conv3d(c1, c1, 3, padding=1, bias=False), _gn(c1), nn.GELU(),
        )
        self.up0 = nn.ConvTranspose3d(c1, c1, 2, stride=2)

        self.cls_head = ClassificationHead3D(c4, n_classes=n_classes)
        self.seg_head = SegmentationHead3D(c1, n_classes=seg_classes)

    def forward(self, image, missing_mask=None, aux_features=None, return_aux=False):
        target_size = image.shape[2:]
        x0 = self.patch_embed(image)        # /2
        s1 = self.stage1(x0)                # /2
        s2 = self.stage2(self.down1(s1))    # /4
        s3 = self.stage3(self.down2(s2))    # /8
        s4 = self.stage4(self.down3(s3))    # /16

        d3 = self.dec3(torch.cat([self._align(self.up3(s4), s3), s3], dim=1))
        d2 = self.dec2(torch.cat([self._align(self.up2(d3), s2), s2], dim=1))
        d1 = self.dec1(torch.cat([self._align(self.up1(d2), s1), s1], dim=1))
        d0 = self.up0(d1)

        cls = self.cls_head(s4)
        seg = self.seg_head(d0, target_size)
        return {"seg": seg, "cls": cls}

    @staticmethod
    def _align(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x


def build_swin_unetr(config: Optional[Dict] = None) -> SwinUNETR3D:
    cfg = (config or {}).get("model", {})
    return SwinUNETR3D(
        in_channels=cfg.get("in_channels", 4),
        n_classes=cfg.get("num_classes", 2),
        seg_classes=cfg.get("seg_classes", 3),
        base_c=cfg.get("base_c", 16),
        depths=cfg.get("depths", (2, 2, 2, 2)),
        num_heads=cfg.get("num_heads", (2, 4, 4, 8)),
        window_size=cfg.get("window_size", (4, 4, 4)),
    )
