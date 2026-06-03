"""Topology case study — connected components, Euler χ, cavity highlighting.

A 2 × 3 panel showing, for one case:

    (top-left)     axial slice with WT connected components colour-coded
    (top-middle)   axial slice with internal cavities (β₁ holes) highlighted
    (top-right)    histogram of per-component voxel counts (multifocality)
    (bottom-left)  3-D scatter sample of all WT components in a different colour
    (bottom-middle) text panel with χ, β₀, surrogate β₁, components, hole count
    (bottom-right) per-axis distance-transform profile through centroid

This is the case-level realisation of Findings 4 and 5 — multifocality
and Euler χ as recurrence-vs-necrosis discriminators.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import helpers as H


def _connected_components(mask: np.ndarray):
    from scipy.ndimage import label as ndi_label
    lab, n = ndi_label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:] if n > 0 else np.array([], dtype=int)
    return lab.astype(np.int32), int(n), sizes


def _euler_chi_and_holes(mask: np.ndarray) -> Tuple[int, int]:
    if not mask.any():
        return 0, 0
    m = mask.astype(np.int8)
    cubes = m[:-1, :-1, :-1] & m[1:, :-1, :-1] & m[:-1, 1:, :-1] & m[:-1, :-1, 1:] \
        & m[1:, 1:, :-1] & m[1:, :-1, 1:] & m[:-1, 1:, 1:] & m[1:, 1:, 1:]
    fx = m[:-1, :-1, :] & m[1:, :-1, :] & m[:-1, 1:, :] & m[1:, 1:, :]
    fy = m[:-1, :, :-1] & m[1:, :, :-1] & m[:-1, :, 1:] & m[1:, :, 1:]
    fz = m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:] & m[:, 1:, 1:]
    ex = m[:-1, :, :] & m[1:, :, :]
    ey = m[:, :-1, :] & m[:, 1:, :]
    ez = m[:, :, :-1] & m[:, :, 1:]
    v = int(m.sum())
    e = int(ex.sum() + ey.sum() + ez.sum())
    f = int(fx.sum() + fy.sum() + fz.sum())
    c = int(cubes.sum())
    chi = v - e + f - c
    return chi, max(0, 0 - chi)


def _cavities(mask: np.ndarray):
    """Find background-hole regions strictly inside the (closed) lesion."""
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(mask)
    return filled & (~mask)


def render(npz_path: str, out_root: Optional[str] = None) -> Path:
    H.use_clean_style(plt)
    case_id = H.case_id_from_path(npz_path)
    image, label = H.load_case(npz_path)
    wt = label[0] > 0
    if not wt.any():
        raise ValueError(f"{npz_path} has empty WT mask — nothing to visualise")

    lab, n_comp, sizes = _connected_components(wt)
    chi, n_holes_surrogate = _euler_chi_and_holes(wt)
    cavities = _cavities(wt)

    cz, cy, cx = H.lesion_centroid(wt)
    brain = H.brain_mask(image)
    t1ce_axial = image[1][cz, :, :]
    vmin, vmax = H.clip_to_brain_vrange(t1ce_axial, brain[cz, :, :])

    fig = plt.figure(figsize=(13, 8))

    # (a) components on axial slice
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(t1ce_axial.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    overlay = lab[cz, :, :]
    ax.imshow(np.ma.masked_where(overlay == 0, overlay).T, cmap="tab20",
              origin="lower", alpha=0.7)
    ax.set_title(f"(a) {n_comp} connected components — axial z={cz}")
    ax.axis("off")

    # (b) cavities (β₁ holes) on axial slice
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(t1ce_axial.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    cav_slc = cavities[cz, :, :]
    if cav_slc.any():
        ax.imshow(np.ma.masked_where(~cav_slc, np.ones_like(cav_slc)).T,
                  cmap="autumn", origin="lower", alpha=0.7)
        ax.set_title(f"(b) internal cavities — {cavities.sum():,} voxels")
    else:
        ax.set_title("(b) no internal cavities in this slice")
    ax.axis("off")

    # (c) per-component size histogram
    ax = fig.add_subplot(2, 3, 3)
    if sizes.size > 0:
        ranks = np.arange(1, sizes.size + 1)
        ax.bar(ranks, np.sort(sizes)[::-1], color="#3690c0", edgecolor="#444")
        for r, s in zip(ranks[:min(5, sizes.size)], np.sort(sizes)[::-1][:5]):
            ax.text(r, s, f"{s:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("component rank (largest first)")
    ax.set_ylabel("voxels (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"(c) component voxel counts  ·  largest = {sizes.max() if sizes.size else 0:,}")

    # (d) 3-D scatter of all components with per-component colour
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    if n_comp > 0:
        keep = max(1, lab.size // 20000)
        zz, yy, xx = np.where(lab > 0)
        # subsample for plot speed
        idx = np.arange(zz.size)
        if zz.size > 4000:
            idx = np.random.default_rng(0).choice(zz.size, size=4000, replace=False)
        labels = lab[zz[idx], yy[idx], xx[idx]]
        ax.scatter(xx[idx], yy[idx], zz[idx], c=labels, cmap="tab20", s=2, alpha=0.6)
    ax.set_axis_off()
    ax.set_title("(d) component spatial distribution")

    # (e) numeric text panel
    ax = fig.add_subplot(2, 3, 5)
    ax.axis("off")
    largest_share = float(sizes.max() / sizes.sum()) if sizes.size else 0.0
    lines = [
        f"n_components          : {n_comp}",
        f"largest_share         : {largest_share:.3f}",
        f"Euler  χ(WT)          : {chi:+d}",
        f"surrogate holes β₁    : {n_holes_surrogate:,}",
        f"cavity voxels         : {int(cavities.sum()):,}",
        "",
        "class signature (cohort priors):",
        "  χ ≤ −20  →  cavitated necrosis",
        "  χ ≥ +5   →  compact recurrence",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11)

    # (f) distance-transform profile through centroid
    ax = fig.add_subplot(2, 3, 6)
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(wt)
    profile_z = dt[:, cy, cx]
    profile_y = dt[cz, :, cx]
    profile_x = dt[cz, cy, :]
    ax.plot(profile_x, label="x axis (sagittal)", color="#d6604d")
    ax.plot(profile_y, label="y axis (coronal)", color="#5aae61")
    ax.plot(profile_z, label="z axis (axial)", color="#3690c0")
    ax.set_xlabel("voxel along axis")
    ax.set_ylabel("distance to lesion boundary (voxels)")
    ax.set_title(f"(f) distance-to-boundary through centroid")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Case {case_id} — topology study  ·  n_components={n_comp}  χ(WT)={chi:+d}  "
        f"surrogate β₁={n_holes_surrogate}",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir = H.case_out_dir(case_id, out_root)
    out_path = out_dir / "03_topology.png"
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
