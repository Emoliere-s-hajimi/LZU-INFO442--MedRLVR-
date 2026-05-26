"""3-D tumour surface rendering — the WT ⊇ TC ⊇ ET nesting prior in space.

Uses ``skimage.measure.marching_cubes`` to triangulate each of the three
nested binary masks, then renders all three from four angles on a single
canvas. WT is drawn semi-transparent so the inner TC and ET surfaces
remain visible. The result is the most direct visual proof of the
Finding 6 nesting invariant — a viewer can see WT enclose TC enclose ET
in any rotation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import helpers as H


def _marching_cubes(mask: np.ndarray, downsample: int = 1):
    from skimage.measure import marching_cubes

    m = mask.astype(np.float32)
    if downsample > 1:
        m = m[::downsample, ::downsample, ::downsample]
    if m.sum() < 4:
        return None, None
    try:
        verts, faces, _, _ = marching_cubes(m, level=0.5)
    except (ValueError, RuntimeError):
        return None, None
    return verts * downsample, faces


def _draw_surface(ax, verts, faces, color, alpha):
    if verts is None:
        return
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = Poly3DCollection(verts[faces], alpha=alpha, edgecolor=None)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)


def render(npz_path: str, out_root: Optional[str] = None, downsample: int = 2) -> Path:
    H.use_clean_style(plt)
    case_id = H.case_id_from_path(npz_path)
    _, label = H.load_case(npz_path)

    surfaces = []
    for ch, name in enumerate(H.LABEL_NAMES):
        v, f = _marching_cubes(label[ch] > 0, downsample=downsample)
        surfaces.append((name, v, f, label[ch].sum()))

    # Two-row layout — top: four-angle WT + TC + ET overlays
    #                  bottom: three single-channel panels for clarity
    fig = plt.figure(figsize=(12, 9))
    angles = [(30, 30), (30, 120), (30, 210), (30, 300)]
    for i, (elev, azim) in enumerate(angles, start=1):
        ax = fig.add_subplot(2, 4, i, projection="3d")
        _draw_surface(ax, surfaces[0][1], surfaces[0][2], H.LABEL_COLORS[0], alpha=0.12)
        _draw_surface(ax, surfaces[1][1], surfaces[1][2], H.LABEL_COLORS[1], alpha=0.40)
        _draw_surface(ax, surfaces[2][1], surfaces[2][2], H.LABEL_COLORS[2], alpha=0.85)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"WT ⊇ TC ⊇ ET   ({azim}°)", fontsize=9)
        # Auto-fit to lesion bbox
        verts_all = [v for _, v, _, _ in surfaces if v is not None]
        if verts_all:
            all_v = np.concatenate(verts_all, axis=0)
            for setter, axis in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
                setter(all_v[:, axis].min(), all_v[:, axis].max())

    for i, (name, verts, faces, vox) in enumerate(surfaces):
        ax = fig.add_subplot(2, 4, 5 + i, projection="3d")
        _draw_surface(ax, verts, faces, H.LABEL_COLORS[i], alpha=0.65)
        ax.view_init(elev=25, azim=45)
        ax.set_axis_off()
        ax.set_title(f"{name} only  ·  {vox:,} voxels", fontsize=9)

    # Numeric inset
    ax = fig.add_subplot(2, 4, 8)
    ax.axis("off")
    wt, tc, et = surfaces[0][3], surfaces[1][3], surfaces[2][3]
    if wt > 0:
        tc_in_wt = float(((label[0] > 0) & (label[1] > 0)).sum() / max(label[1].sum(), 1))
        et_in_tc = float(((label[1] > 0) & (label[2] > 0)).sum() / max(label[2].sum(), 1))
        lines = [
            f"WT voxels:  {wt:>10,}",
            f"TC voxels:  {tc:>10,}   ({100 * tc / max(wt, 1):4.1f}% of WT)",
            f"ET voxels:  {et:>10,}   ({100 * et / max(wt, 1):4.1f}% of WT)",
            "",
            f"TC ⊆ WT share : {tc_in_wt:.3f}",
            f"ET ⊆ TC share : {et_in_tc:.3f}",
        ]
        ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=10)

    fig.suptitle(f"Case {case_id} — 3-D nested tumour surfaces (Finding 6 prior)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir = H.case_out_dir(case_id, out_root)
    out_path = out_dir / "02_tumor_3d_nesting.png"
    fig.savefig(out_path, dpi=H.DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/some_cleaned_examples/001.npz")
    p.add_argument("--out_root", default=None)
    p.add_argument("--downsample", type=int, default=2)
    args = p.parse_args()
    print(render(args.npz, args.out_root, downsample=args.downsample))


if __name__ == "__main__":
    main()
