"""Anatomy panel — three orthogonal views through the lesion centroid.

For one case, produces a 5 × 3 grid:

    rows   : T1, T1ce, T2, FLAIR, seg overlay on T1ce
    cols   : axial, coronal, sagittal slice (through WT centroid)

Each slice is contrast-windowed by the brain-interior 2nd/98th percentile
so multi-vendor intensity variation does not wash the panel out.
This is the radiology-standard view our clinical collaborators read,
and it grounds every downstream "what is the model looking at" question.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import helpers as H


def render(npz_path: str, out_root: str | Path | None = None) -> Path:
    H.use_clean_style(plt)
    case_id = H.case_id_from_path(npz_path)
    image, label = H.load_case(npz_path)
    brain = H.brain_mask(image)
    cz, cy, cx = H.lesion_centroid(label[0] > 0, fallback=tuple(s // 2 for s in image.shape[1:]))

    # Axial: z (first axis); Coronal: y (second); Sagittal: x (third)
    def slc(volume3d, axis: int, idx: int):
        if axis == 0: return volume3d[idx, :, :]
        if axis == 1: return volume3d[:, idx, :]
        return volume3d[:, :, idx]

    views = [("axial (z=%d)" % cz, 0, cz),
             ("coronal (y=%d)" % cy, 1, cy),
             ("sagittal (x=%d)" % cx, 2, cx)]

    fig, axes = plt.subplots(5, 3, figsize=(11, 16))
    for col, (view_name, axis, idx) in enumerate(views):
        brain_slc = slc(brain, axis, idx)
        for row, mod_name in enumerate(H.MODALITY_NAMES):
            ax = axes[row, col]
            sl = slc(image[row], axis, idx)
            vmin, vmax = H.clip_to_brain_vrange(sl, brain_slc)
            ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_title(f"{mod_name} — {view_name}" if row == 0 else mod_name, fontsize=9)
            if (image[row] == 0).all():
                ax.text(0.5, 0.5, "modality missing", color="#d6604d",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, fontweight="bold")
            ax.axis("off")

        # Bottom row: T1ce + seg overlay
        ax = axes[4, col]
        sl = slc(image[1], axis, idx)
        vmin, vmax = H.clip_to_brain_vrange(sl, brain_slc)
        ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        # Stack label channels for this slice
        if axis == 0: lab_slc = label[:, idx, :, :]
        elif axis == 1: lab_slc = label[:, :, idx, :]
        else: lab_slc = label[:, :, :, idx]
        rgba = H.stack_label_rgba(lab_slc)
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")
        ax.set_title(f"T1ce + [WT, TC, ET] overlay", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Case {case_id} — three-orthogonal-view anatomy panel",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_dir = H.case_out_dir(case_id, out_root)
    out_path = out_dir / "01_anatomy_orthogonal.png"
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
