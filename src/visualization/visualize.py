"""Visualization helpers for multimodal MRI slices and segmentation overlays."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _safe_import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def grid_2d_slices(slices: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None, ncols: int = 3, out_path: Optional[str] = None):
    plt = _safe_import_plt()
    n = len(slices)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i, slc in enumerate(slices):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        ax.imshow(slc, cmap="gray")
        ax.axis("off")
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def overlay_segmentation(image: np.ndarray, seg: np.ndarray, alpha: float = 0.4, out_path: Optional[str] = None):
    plt = _safe_import_plt()
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    masked = np.ma.masked_where(seg == 0, seg)
    ax.imshow(masked, cmap="autumn", alpha=alpha)
    ax.axis("off")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def save_modality_panel(volumes_by_modality: dict, slice_index: int, out_path: str):
    titles = list(volumes_by_modality.keys())
    slices = [v[:, :, slice_index] if v.ndim == 3 else v for v in volumes_by_modality.values()]
    return grid_2d_slices(slices, titles=titles, ncols=len(slices), out_path=out_path)
