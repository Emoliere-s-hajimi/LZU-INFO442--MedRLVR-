"""Shared utilities for the case-study visualisation pack."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# --- Constants shared by every script ------------------------------------
MODALITY_NAMES = ("T1", "T1ce", "T2", "FLAIR")
LABEL_NAMES = ("WT", "TC", "ET")
LABEL_COLORS = ("#d6604d", "#5aae61", "#3690c0")  # red, green, blue
MODALITY_CMAPS = {
    "T1":    "gray",
    "T1ce":  "gray",
    "T2":    "gray",
    "FLAIR": "gray",
}
DPI = 150
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "visualization" / "case_study"


def use_clean_style(plt) -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9,
        "legend.frameon": False,
    })


# --- Loading -------------------------------------------------------------

def load_case(npz_path: str):
    """Load image (4, H, W, D) and label (3, H, W, D) from a cleaned npz."""
    data = np.load(npz_path)
    return data["image"], data["label"]


def case_id_from_path(npz_path: str) -> str:
    return Path(npz_path).stem


def case_out_dir(case_id: str, out_root: str | Path | None = None) -> Path:
    root = Path(out_root) if out_root is not None else DEFAULT_OUT_ROOT
    out = root / case_id
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- Geometry ------------------------------------------------------------

def lesion_centroid(wt: np.ndarray, fallback: Tuple[int, int, int] | None = None) -> Tuple[int, int, int]:
    """Centre of mass of the WT mask (integer voxel index) or fallback."""
    if wt.any():
        coords = np.argwhere(wt > 0)
        c = coords.mean(0)
        return tuple(int(v) for v in c)
    if fallback is None:
        return tuple(s // 2 for s in wt.shape)
    return fallback


def brain_mask(image: np.ndarray) -> np.ndarray:
    """Non-zero across any modality channel."""
    return np.any(image != 0, axis=0)


def clip_to_brain_vrange(slice_2d: np.ndarray, brain_2d: np.ndarray) -> Tuple[float, float]:
    """Return a robust vmin/vmax based on the brain interior of one slice."""
    fg = slice_2d[brain_2d > 0]
    if fg.size == 0:
        return float(slice_2d.min()), float(slice_2d.max())
    lo, hi = np.percentile(fg, [2, 98])
    return float(lo), float(hi)


def stack_label_rgba(label_slice_3xhw: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Convert a (3, H, W) binary nested label slice to (H, W, 4) RGBA overlay.

    Channel order [WT, TC, ET]; each painted with its colour. Outer mask
    is drawn first so inner masks override it.
    """
    import matplotlib.colors as mcolors

    H, W = label_slice_3xhw.shape[1:]
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for ch, color in zip(range(3), LABEL_COLORS):
        mask = label_slice_3xhw[ch] > 0
        if not mask.any():
            continue
        rgb = np.array(mcolors.to_rgb(color), dtype=np.float32)
        for c in range(3):
            rgba[..., c][mask] = rgb[c]
        rgba[..., 3][mask] = alpha
    return rgba
