"""Morphology case study — sphericity, principal axes, surface roughness.

A 2 × 3 panel:

    (a) WT axial mid-slice with the equivalent-volume sphere outline overlaid
        — visualising the gap between actual shape and the round-tumour prior
    (b) Sagittal slice with bounding-box edges marked
        — shows the bbox-fill ratio in pixels
    (c) PCA principal axes (3-D scatter sample + 3 coloured eigen-vectors)
    (d) Surface roughness map: per-surface-voxel local std of the morphological
        gradient. Bright = highly irregular boundary (recurrence-like).
    (e) Numeric panel: sphericity, elongation, bbox fill, surface area
    (f) Polar diagram of lesion extent along 16 radial directions —
        an at-a-glance "shape fingerprint"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import helpers as H


def _surface_area_voxels(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    sa = 0
    for axis in range(3):
        diff = np.diff(mask.astype(np.int8), axis=axis)
        sa += int((diff != 0).sum())
    return float(sa)


def _sphericity(volume: int, surface: float) -> float:
    if volume <= 0 or surface <= 0:
        return 0.0
    return float(np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface)


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.min(0), coords.max(0) + 1


def _equivalent_sphere_radius(volume: int) -> float:
    return float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def _pca_axes(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.shape[0] < 3:
        return None, None, None
    centred = coords - coords.mean(0)
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return coords.mean(0), eigvals[order], eigvecs[:, order]


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Voxels in mask that have at least one zero 6-neighbour."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    return mask & ~eroded


def _radial_extent_polar(mask: np.ndarray, n_dirs: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Max radial reach of the mask along n_dirs angles in the axial plane,
    averaged over slices that contain the lesion."""
    if not mask.any():
        return np.linspace(0, 2 * np.pi, n_dirs, endpoint=False), np.zeros(n_dirs)
    coords = np.argwhere(mask)
    centroid = coords.mean(0)
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    radii = np.zeros(n_dirs, dtype=np.float32)
    cy, cx = centroid[1], centroid[2]
    yy, xx = np.argwhere(mask.any(0)).T
    if yy.size == 0:
        return angles, radii
    dy = yy - cy; dx = xx - cx
    rr = np.sqrt(dy ** 2 + dx ** 2)
    aa = np.arctan2(dy, dx) % (2 * np.pi)
    for i, a in enumerate(angles):
        in_wedge = (np.abs((aa - a + np.pi) % (2 * np.pi) - np.pi) < (np.pi / n_dirs))
        if in_wedge.any():
            radii[i] = float(rr[in_wedge].max())
    return angles, radii


def render(npz_path: str, out_root: Optional[str] = None) -> Path:
    H.use_clean_style(plt)
    case_id = H.case_id_from_path(npz_path)
    image, label = H.load_case(npz_path)
    wt = label[0] > 0
    if not wt.any():
        raise ValueError(f"{npz_path}: empty WT mask")

    cz, cy, cx = H.lesion_centroid(wt)
    brain = H.brain_mask(image)
    surface_area = _surface_area_voxels(wt)
    vol = int(wt.sum())
    sphericity = _sphericity(vol, surface_area)
    bbox = _bbox(wt)
    bbox_fill = float(vol / np.prod(bbox[1] - bbox[0])) if bbox is not None else 0.0
    elongation = float(np.max(bbox[1] - bbox[0]) / max(1, np.min(bbox[1] - bbox[0]))) if bbox is not None else 0.0
    eq_r = _equivalent_sphere_radius(vol)

    fig = plt.figure(figsize=(13, 8))

    # (a) axial slice + equivalent-volume sphere outline
    ax = fig.add_subplot(2, 3, 1)
    t1ce_axial = image[1][cz, :, :]
    vmin, vmax = H.clip_to_brain_vrange(t1ce_axial, brain[cz, :, :])
    ax.imshow(t1ce_axial.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    wt_slc = wt[cz, :, :]
    ax.imshow(np.ma.masked_where(~wt_slc, np.ones_like(wt_slc)).T,
              cmap="autumn", alpha=0.45, origin="lower")
    circ = plt.Circle((cx, cy), eq_r, fill=False, color="#3690c0", linewidth=1.6,
                      linestyle="--", label="equivalent-volume sphere")
    ax.add_patch(circ)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"(a) WT vs equivalent sphere  ψ={sphericity:.3f}")
    ax.axis("off")

    # (b) sagittal slice with bbox edges
    ax = fig.add_subplot(2, 3, 2)
    t1ce_sag = image[1][:, :, cx]
    vmin, vmax = H.clip_to_brain_vrange(t1ce_sag, brain[:, :, cx])
    ax.imshow(t1ce_sag.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    if bbox is not None:
        zmin, ymin, _ = bbox[0]; zmax, ymax, _ = bbox[1]
        ax.add_patch(plt.Rectangle((zmin, ymin), zmax - zmin, ymax - ymin,
                                   fill=False, edgecolor="#5aae61", linewidth=1.8,
                                   label=f"bbox fill {bbox_fill:.2%}"))
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"(b) bbox fill ratio  =  {bbox_fill:.3f}")
    ax.axis("off")

    # (c) PCA principal axes
    ax = fig.add_subplot(2, 3, 3, projection="3d")
    centre, eigvals, eigvecs = _pca_axes(wt)
    if centre is not None:
        coords = np.argwhere(wt)
        rng = np.random.default_rng(0)
        sub = coords[rng.choice(coords.shape[0],
                                size=min(3000, coords.shape[0]), replace=False)]
        ax.scatter(sub[:, 2], sub[:, 1], sub[:, 0], s=2, c="#d6d6d6", alpha=0.4)
        axis_colors = ("#d6604d", "#5aae61", "#3690c0")
        for i in range(3):
            length = float(np.sqrt(eigvals[i])) * 2.5
            vec = eigvecs[:, i] * length
            ax.plot([centre[2] - vec[2], centre[2] + vec[2]],
                    [centre[1] - vec[1], centre[1] + vec[1]],
                    [centre[0] - vec[0], centre[0] + vec[0]],
                    color=axis_colors[i], lw=2.5,
                    label=f"PC{i+1} (λ={eigvals[i]:.1f})")
        ax.legend(fontsize=7, loc="upper left")
    ax.set_axis_off()
    ax.set_title("(c) PCA principal axes")

    # (d) surface roughness map (axial slice, mean of |morph gradient|)
    ax = fig.add_subplot(2, 3, 4)
    from scipy.ndimage import grey_dilation, grey_erosion
    morph = grey_dilation(wt.astype(np.float32), size=3) - grey_erosion(wt.astype(np.float32), size=3)
    surf_voxels = _surface_voxels(wt)
    rough = np.zeros_like(wt, dtype=np.float32)
    rough[surf_voxels] = morph[surf_voxels]
    # Local std of roughness within a 5-voxel window on the surface
    from scipy.ndimage import generic_filter
    surf_slc = rough[cz, :, :]
    ax.imshow(t1ce_axial.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.imshow(np.ma.masked_where(surf_slc <= 0, surf_slc).T, cmap="autumn", origin="lower",
              alpha=0.8, vmin=0, vmax=1.0)
    ax.set_title(f"(d) surface roughness (axial)  ·  area={surface_area:.0f}")
    ax.axis("off")

    # (e) numeric panel
    ax = fig.add_subplot(2, 3, 5)
    ax.axis("off")
    lines = [
        f"volume voxels     : {vol:>10,}",
        f"surface area      : {surface_area:>10.0f}",
        f"sphericity ψ      : {sphericity:>10.3f}",
        f"elongation        : {elongation:>10.2f}",
        f"bbox fill         : {bbox_fill:>10.3f}",
        f"equiv-sphere r    : {eq_r:>10.2f} voxels",
        "",
        "cohort priors:",
        "  median ψ = 0.362",
        "  no case ψ ≥ 0.65",
        "  median bbox fill = 0.241",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11)

    # (f) polar shape fingerprint
    ax = fig.add_subplot(2, 3, 6, projection="polar")
    angles, radii = _radial_extent_polar(wt, n_dirs=24)
    angles_closed = np.append(angles, angles[0])
    radii_closed = np.append(radii, radii[0])
    ax.plot(angles_closed, radii_closed, color="#d6604d", lw=1.6)
    ax.fill(angles_closed, radii_closed, color="#d6604d", alpha=0.18)
    ax.set_title("(f) axial radial extent fingerprint", fontsize=10, pad=12)
    ax.set_yticks([])

    fig.suptitle(
        f"Case {case_id} — morphology study  ·  ψ={sphericity:.3f}  "
        f"elong={elongation:.2f}  bbox_fill={bbox_fill:.3f}",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir = H.case_out_dir(case_id, out_root)
    out_path = out_dir / "04_morphology.png"
    fig.savefig(out_path, dpi=H.DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/some_cleaned_examples/001.npz")
    p.add_argument("--out_root", default=None)
    args = p.parse_args()
    print(render(args.npz, args.out_root))


if __name__ == "__main__":
    main()
