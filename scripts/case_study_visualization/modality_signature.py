"""Modality signature case study — Finding 7 and Finding 8 made concrete.

A 2 × 4 panel:

    Top row    : axial mid-slice through the lesion for each modality, with
                 the WT contour overlaid. The viewer sees with their own
                 eyes which modality lights up the lesion best.
    Bottom row : per-modality intensity histograms — inside-lesion (red) vs
                 outside-lesion-but-inside-brain (grey). The Bhattacharyya
                 distance between the two is printed; this is the case's
                 own measurement of the FLAIR > T2 > T1ce > T1 prior.

Then a short text panel beneath calls out the T1ce in/out ratio and where
the case sits on the cohort's recurrence (+1.42) vs necrosis (+0.88)
priors, which is Finding 8.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import helpers as H


def _bhattacharyya(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = p / (p.sum() + eps); q = q / (q.sum() + eps)
    return -float(np.log(np.sum(np.sqrt(p * q)) + eps))


def _contour_overlay(ax, mask_slice: np.ndarray, color: str = "#fde725") -> None:
    if mask_slice.any():
        ax.contour(mask_slice.T, levels=[0.5], colors=color, linewidths=1.0)


def render(npz_path: str, out_root: Optional[str] = None) -> Path:
    H.use_clean_style(plt)
    case_id = H.case_id_from_path(npz_path)
    image, label = H.load_case(npz_path)
    wt = label[0] > 0
    if not wt.any():
        raise ValueError(f"{npz_path}: empty WT mask")

    cz, cy, cx = H.lesion_centroid(wt)
    brain = H.brain_mask(image)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5),
                             gridspec_kw=dict(height_ratios=[1.05, 1.0]))

    in_out_means: list[Tuple[str, float, float, float]] = []
    bhat_distances: list[Tuple[str, float]] = []

    for col, mod_name in enumerate(H.MODALITY_NAMES):
        sl = image[col][cz, :, :]
        brain_slc = brain[cz, :, :]
        present = not (image[col] == 0).all()
        ax_top = axes[0, col]
        if present:
            vmin, vmax = H.clip_to_brain_vrange(sl, brain_slc)
            ax_top.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            _contour_overlay(ax_top, wt[cz, :, :])
            ax_top.set_title(mod_name, fontsize=10)
        else:
            ax_top.imshow(np.zeros_like(sl).T, cmap="gray", origin="lower")
            ax_top.text(0.5, 0.5, "modality missing", color="#d6604d",
                        ha="center", va="center", transform=ax_top.transAxes,
                        fontsize=11, fontweight="bold")
            ax_top.set_title(mod_name, fontsize=10)
        ax_top.axis("off")

        ax_bot = axes[1, col]
        if present:
            inside = image[col][wt & brain]
            outside = image[col][(~wt) & brain]
            if inside.size > 0 and outside.size > 0:
                lo = float(min(inside.min(), outside.min()))
                hi = float(max(inside.max(), outside.max()))
                bins = np.linspace(lo, hi, 60)
                ax_bot.hist(outside, bins=bins, color="#cccccc",
                            edgecolor="white", label="outside brain mask")
                ax_bot.hist(inside, bins=bins, color="#d6604d",
                            alpha=0.7, edgecolor="white", label="inside WT")
                ax_bot.set_xlabel("foreground z")
                ax_bot.set_ylabel("voxel count")
                ax_bot.legend(fontsize=7, loc="upper right")
                # Bhattacharyya distance
                hist_in, _ = np.histogram(inside, bins=bins)
                hist_out, _ = np.histogram(outside, bins=bins)
                bd = _bhattacharyya(hist_in.astype(np.float64), hist_out.astype(np.float64))
                ratio = float(inside.mean() / outside.mean()) if abs(outside.mean()) > 1e-6 else 0.0
                in_out_means.append((mod_name, float(inside.mean()), float(outside.mean()), ratio))
                bhat_distances.append((mod_name, bd))
                ax_bot.set_title(f"{mod_name}  ·  Δmean = {inside.mean()-outside.mean():+.2f}  ·  Bh = {bd:.2f}",
                                 fontsize=9)
            else:
                ax_bot.set_visible(False)
        else:
            ax_bot.set_visible(False)
            in_out_means.append((mod_name, 0.0, 0.0, 0.0))
            bhat_distances.append((mod_name, 0.0))

    # Sub-caption — case-level summary aligned with the cohort priors
    ranking = sorted(bhat_distances, key=lambda kv: -kv[1])
    rank_str = " > ".join(f"{m}({d:.2f})" for m, d in ranking if d > 0)
    t1ce_ratio = next((r for n, mi, mo, r in in_out_means if n == "T1ce"), 0.0)
    cohort_recur, cohort_necr = 1.42, 0.88
    closer = "recurrence" if abs(t1ce_ratio - cohort_recur) < abs(t1ce_ratio - cohort_necr) else "necrosis"

    caption = (
        f"Inside-vs-outside separation ranking for this case (Bhattacharyya):  "
        f"{rank_str or '— modalities missing —'}\n"
        f"Cohort prior (Finding 7):  FLAIR(3.33σ) > T2(2.16σ) > T1ce(1.39σ) > T1(0.23σ).\n"
        f"T1ce inside/outside ratio (Finding 8):  this case = {t1ce_ratio:+.2f}   "
        f"·  cohort recurrence median = {cohort_recur:+.2f}   "
        f"·  cohort necrosis median = {cohort_necr:+.2f}   "
        f"·  closer to **{closer}**"
    )
    fig.suptitle(f"Case {case_id} — modality signature", fontsize=13, y=0.995)
    fig.text(0.5, -0.01, caption, ha="center", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_dir = H.case_out_dir(case_id, out_root)
    out_path = out_dir / "05_modality_signature.png"
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
