"""Volume-level preprocessing utilities for multimodal brain MRI."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def normalize_zscore(volume: np.ndarray, foreground_only: bool = True) -> np.ndarray:
    """Z-score normalization. When `foreground_only` is True, statistics are
    computed over the non-zero region only — the typical recipe for brain MRI
    after skull stripping."""
    volume = volume.astype(np.float32)
    if foreground_only:
        mask = volume > 0
        if mask.sum() == 0:
            return volume
        mean = volume[mask].mean()
        std = volume[mask].std() + 1e-8
        out = (volume - mean) / std
        out[~mask] = 0.0
        return out
    mean = volume.mean()
    std = volume.std() + 1e-8
    return (volume - mean) / std


def percentile_clip(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(volume[volume > 0], [low, high]) if (volume > 0).any() else (volume.min(), volume.max())
    return np.clip(volume, lo, hi)


def resample_volume(volume: np.ndarray, src_spacing: Sequence[float], dst_spacing: Sequence[float], order: int = 1) -> np.ndarray:
    """Resample a 3D volume to the target voxel spacing via affine zoom."""
    from scipy.ndimage import zoom

    factors = [s / d for s, d in zip(src_spacing, dst_spacing)]
    return zoom(volume, factors, order=order)


def crop_or_pad(volume: np.ndarray, target: Tuple[int, int, int], constant: float = 0.0) -> np.ndarray:
    out = volume
    pad_width = []
    for i in range(3):
        diff = target[i] - out.shape[i]
        if diff > 0:
            pad_width.append((diff // 2, diff - diff // 2))
        else:
            pad_width.append((0, 0))
    if any(p != (0, 0) for p in pad_width):
        out = np.pad(out, pad_width, mode="constant", constant_values=constant)

    slices = []
    for i in range(3):
        diff = out.shape[i] - target[i]
        if diff > 0:
            start = diff // 2
            slices.append(slice(start, start + target[i]))
        else:
            slices.append(slice(None))
    return out[tuple(slices)]


def center_on_lesion(seg: np.ndarray, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if seg is not None and seg.any():
        coords = np.argwhere(seg > 0)
        return tuple(int(c) for c in coords.mean(axis=0))
    return fallback
