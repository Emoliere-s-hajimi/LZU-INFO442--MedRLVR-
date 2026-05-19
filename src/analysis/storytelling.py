"""Storytelling EDA: turn ``cohort_scan`` output into figures and a
markdown narrative the team can drop straight into the M2/M3 report.

The figures are intentionally a small, opinionated set rather than an
exhaustive grid — each one answers one question that drives a downstream
modelling decision:

    fig01_class_balance.png       — how imbalanced is the cohort?
    fig02_modality_coverage.png   — which modalities are missing, and where?
    fig03_missingness_heatmap.png — heatmap of (case × modality) presence
    fig04_shape_distribution.png  — per-axis volume-shape histograms
    fig05_spacing_distribution.png — voxel-spacing histograms per modality
    fig06_dataset_storage.png     — bytes consumed per modality / per class
    fig07_intensity_distribution.png — intensity ranges sampled per modality
    fig08_modality_panel.png      — representative slice panel (one case)
    fig09_invalid_id_strip.png    — explicit visual call-out of dropped IDs
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..data.modality_map import CANONICAL_MODALITIES
from .cohort_scan import CaseScan


def _safe_import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

@dataclass
class CohortStats:
    n_total: int = 0
    n_invalid: int = 0
    n_kept: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    missing_modality_counts: Dict[str, int] = field(default_factory=dict)
    n_complete_4mod: int = 0
    n_with_pwi: int = 0
    n_with_seg: int = 0
    total_bytes: int = 0
    bytes_per_modality: Dict[str, int] = field(default_factory=dict)
    bytes_per_class: Dict[str, int] = field(default_factory=dict)
    shape_distribution: Dict[str, List[List[int]]] = field(default_factory=dict)
    spacing_distribution: Dict[str, List[List[float]]] = field(default_factory=dict)
    invalid_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_total": self.n_total,
            "n_invalid": self.n_invalid,
            "n_kept": self.n_kept,
            "class_counts": self.class_counts,
            "missing_modality_counts": self.missing_modality_counts,
            "n_complete_4mod": self.n_complete_4mod,
            "n_with_pwi": self.n_with_pwi,
            "n_with_seg": self.n_with_seg,
            "total_bytes": self.total_bytes,
            "bytes_per_modality": self.bytes_per_modality,
            "bytes_per_class": self.bytes_per_class,
            "shape_distribution": self.shape_distribution,
            "spacing_distribution": self.spacing_distribution,
            "invalid_ids": self.invalid_ids,
        }


def aggregate(scans: Iterable[CaseScan]) -> CohortStats:
    stats = CohortStats()
    for s in scans:
        stats.n_total += 1
        if s.invalid_reason is not None:
            stats.n_invalid += 1
            stats.invalid_ids.append(s.case_id)
            continue
        stats.n_kept += 1
        label = s.label or "unlabeled"
        stats.class_counts[label] = stats.class_counts.get(label, 0) + 1
        if s.has_pwi:
            stats.n_with_pwi += 1
        if s.has_seg:
            stats.n_with_seg += 1
        if not s.missing_modalities:
            stats.n_complete_4mod += 1
        for m in s.missing_modalities:
            stats.missing_modality_counts[m] = stats.missing_modality_counts.get(m, 0) + 1
        for m, b in s.modality_bytes.items():
            stats.bytes_per_modality[m] = stats.bytes_per_modality.get(m, 0) + b
            stats.total_bytes += b
            stats.bytes_per_class[label] = stats.bytes_per_class.get(label, 0) + b
        for m, shape in s.modality_shape.items():
            stats.shape_distribution.setdefault(m, []).append(list(shape))
        for m, spacing in s.modality_spacing.items():
            stats.spacing_distribution.setdefault(m, []).append(list(spacing))
    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_CLASS_COLORS = {
    "recurrence": "#d6604d",
    "necrosis": "#4393c3",
    "necrosis+recurrence": "#9970ab",
    "unlabeled": "#bdbdbd",
}


def fig_class_balance(stats: CohortStats, out: Path) -> None:
    plt = _safe_import_plt()
    classes = list(stats.class_counts.keys())
    counts = [stats.class_counts[c] for c in classes]
    colors = [_CLASS_COLORS.get(c, "#888888") for c in classes]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(classes, counts, color=colors)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=9)
    ax.set_title("Class balance — kept cohort")
    ax.set_ylabel("# cases")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig01_class_balance.png", dpi=150)
    plt.close(fig)


def fig_modality_coverage(stats: CohortStats, out: Path) -> None:
    plt = _safe_import_plt()
    miss = {m: stats.missing_modality_counts.get(m, 0) for m in CANONICAL_MODALITIES}
    present = {m: stats.n_kept - miss[m] for m in CANONICAL_MODALITIES}
    x = np.arange(len(CANONICAL_MODALITIES))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x, [present[m] for m in CANONICAL_MODALITIES], color="#5aae61", label="present")
    ax.bar(x, [miss[m] for m in CANONICAL_MODALITIES],
           bottom=[present[m] for m in CANONICAL_MODALITIES],
           color="#d6604d", label="missing")
    ax.set_xticks(x)
    ax.set_xticklabels(CANONICAL_MODALITIES)
    ax.set_ylabel("# cases")
    ax.set_title("Modality coverage across the kept cohort")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig02_modality_coverage.png", dpi=150)
    plt.close(fig)


def fig_missingness_heatmap(scans: List[CaseScan], out: Path) -> None:
    plt = _safe_import_plt()
    scans = [s for s in scans if s.invalid_reason is None]
    scans.sort(key=lambda s: (s.label or "z", s.case_id))
    M = np.zeros((len(scans), len(CANONICAL_MODALITIES)), dtype=np.uint8)
    for i, s in enumerate(scans):
        for j, m in enumerate(CANONICAL_MODALITIES):
            M[i, j] = 1 if m in s.modality_shape else 0
    fig, ax = plt.subplots(figsize=(4.5, max(3.5, 0.04 * len(scans))))
    ax.imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(CANONICAL_MODALITIES)))
    ax.set_xticklabels(CANONICAL_MODALITIES)
    ax.set_ylabel(f"case (n={len(scans)})")
    ax.set_title("Per-case modality presence (green = present)")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "fig03_missingness_heatmap.png", dpi=150)
    plt.close(fig)


def fig_shape_distribution(stats: CohortStats, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    axis_names = ["axis-0", "axis-1", "axis-2"]
    for ax_idx, ax in enumerate(axes):
        for m in CANONICAL_MODALITIES:
            shapes = stats.shape_distribution.get(m, [])
            if not shapes:
                continue
            arr = np.array(shapes)[:, ax_idx]
            ax.hist(arr, bins=20, alpha=0.45, label=m)
        ax.set_title(axis_names[ax_idx])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("# cases")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Volume shape distribution per modality", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig04_shape_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_spacing_distribution(stats: CohortStats, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    axis_names = ["slice", "in-plane y", "in-plane x"]
    for ax_idx, ax in enumerate(axes):
        for m in CANONICAL_MODALITIES:
            spc = stats.spacing_distribution.get(m, [])
            if not spc:
                continue
            arr = np.array(spc)[:, ax_idx]
            ax.hist(arr, bins=20, alpha=0.45, label=m)
        ax.set_title(axis_names[ax_idx])
        ax.set_xlabel("mm")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("# cases")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Voxel spacing distribution per modality", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig05_spacing_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_storage_bar(stats: CohortStats, out: Path) -> None:
    plt = _safe_import_plt()
    GB = 1024 ** 3
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    mods = list(stats.bytes_per_modality.keys())
    axes[0].bar(mods, [stats.bytes_per_modality[m] / GB for m in mods], color="#3690c0")
    axes[0].set_ylabel("GB"); axes[0].set_title("Storage by modality")
    axes[0].spines[["top", "right"]].set_visible(False)

    classes = list(stats.bytes_per_class.keys())
    axes[1].bar(classes,
                [stats.bytes_per_class[c] / GB for c in classes],
                color=[_CLASS_COLORS.get(c, "#888888") for c in classes])
    axes[1].set_ylabel("GB"); axes[1].set_title("Storage by class")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig06_dataset_storage.png", dpi=150)
    plt.close(fig)


def fig_intensity_distribution(intensity_samples: Dict[str, np.ndarray], out: Path) -> None:
    plt = _safe_import_plt()
    if not intensity_samples:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m, samples in intensity_samples.items():
        if samples.size == 0:
            continue
        ax.hist(samples, bins=80, alpha=0.45, label=m, density=True)
    ax.set_title("Foreground-voxel intensity distribution (sampled)")
    ax.set_xlabel("z-scored intensity")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig07_intensity_distribution.png", dpi=150)
    plt.close(fig)


def fig_modality_panel(npz_path: Optional[str], out: Path) -> Optional[str]:
    """One representative case → axial mid-slice in all 4 modalities + the
    WT/TC/ET seg overlay. Skipped if no preprocessed npz is available."""
    if npz_path is None or not Path(npz_path).exists():
        return None
    plt = _safe_import_plt()
    data = np.load(npz_path)
    img, lab = data["image"], data["label"]
    z = img.shape[3] // 2
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    for i, m in enumerate(CANONICAL_MODALITIES):
        axes[i].imshow(img[i, :, :, z].T, cmap="gray", origin="lower")
        axes[i].set_title(m); axes[i].axis("off")
    overlay = np.stack([
        lab[0, :, :, z] * 0.5,    # WT (red)
        lab[1, :, :, z] * 0.5,    # TC (green)
        lab[2, :, :, z] * 0.7,    # ET (blue)
    ], axis=-1)
    axes[4].imshow(img[1, :, :, z].T, cmap="gray", origin="lower")
    axes[4].imshow(np.transpose(overlay, (1, 0, 2)), alpha=0.55, origin="lower")
    axes[4].set_title("seg overlay on T1ce"); axes[4].axis("off")
    fig.tight_layout()
    out_file = out / "fig08_modality_panel.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)


def fig_invalid_strip(stats: CohortStats, out: Path) -> None:
    plt = _safe_import_plt()
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ids = sorted(stats.invalid_ids)
    ax.scatter(range(len(ids)), [0] * len(ids), s=80, color="#d6604d")
    for i, cid in enumerate(ids):
        ax.text(i, 0.04, cid, ha="center", fontsize=8)
    ax.set_xlim(-1, len(ids))
    ax.set_ylim(-0.1, 0.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(
        f"{len(ids)} invalid case IDs dropped before any analysis "
        "(patients who did not undergo radiation therapy)"
    )
    for sp in ["top", "right", "left", "bottom"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig09_invalid_id_strip.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Intensity subsampling (slow — only run on a small sample)
# ---------------------------------------------------------------------------

def sample_intensities(
    scans: List[CaseScan],
    sample_n: int = 12,
    voxels_per_case: int = 50000,
    seed: int = 442,
) -> Dict[str, np.ndarray]:
    """Load a small random subset of full NIfTI volumes and z-score their
    foreground. Returns a per-modality 1-D array of sampled intensities for
    the density plot. DICOM-only cases are skipped (would be too slow).
    """
    import nibabel as nib

    rng = np.random.default_rng(seed)
    nifti_scans = [s for s in scans if s.invalid_reason is None
                   and any(p.endswith(".nii") or p.endswith(".nii.gz")
                           for p in s.modality_source.values())]
    if not nifti_scans:
        return {}
    pick = rng.choice(len(nifti_scans), size=min(sample_n, len(nifti_scans)), replace=False)
    samples: Dict[str, List[np.ndarray]] = {m: [] for m in CANONICAL_MODALITIES}
    for idx in pick:
        s = nifti_scans[int(idx)]
        for m in CANONICAL_MODALITIES:
            p = s.modality_source.get(m)
            if not p or not (p.endswith(".nii") or p.endswith(".nii.gz")):
                continue
            try:
                vol = nib.load(p).get_fdata().astype(np.float32)
            except Exception:
                continue
            fg = vol[vol > 0]
            if fg.size == 0:
                continue
            z = (fg - fg.mean()) / (fg.std() + 1e-8)
            if z.size > voxels_per_case:
                z = rng.choice(z, size=voxels_per_case, replace=False)
            samples[m].append(z)
    return {m: np.concatenate(v) if v else np.array([]) for m, v in samples.items()}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = """# Cohort EDA — story of the Tiantan glioma post-RT dataset

*Auto-generated by ``src.analysis.storytelling``. The numbers below are
read straight off the raw dump headers; no full volumes are loaded
except for the small intensity subsample.*

## 1. Where do we start

We start from a raw dump of {n_total} unique case IDs spread across two
physical layouts — BraTS-style NIfTI per-case folders and a per-modality
DICOM tree (``4个常规结构像/{{T1,T1ce,T2,FLAIR}}/{{复发,放坏,放坏+复发}}``).
The dump occupies **{total_gb:.1f} GB on disk** ({total_bytes:,} bytes).

Before any model training we drop **{n_invalid} case IDs** flagged by the
clinical collaborators as not having actually undergone radiation therapy
— these cases violate the post-radiation inclusion criterion of the
study. The dropped IDs are: {invalid_ids_str}. See
``fig09_invalid_id_strip.png``.

After this fixed exclusion list, **{n_kept} cases remain** for the rest
of the analysis.

## 2. Class balance

| class | # cases |
|---|---|
{class_table}

The cohort is structurally imbalanced — class imbalance handling
(weighted sampler at the input side, focal-style loss at the output side)
is therefore a first-class part of the training pipeline rather than an
afterthought. See ``fig01_class_balance.png``.

## 3. Modality coverage

Of the {n_kept} retained cases, **{n_complete} ({pct_complete:.1f}%)
carry the full 4-modality set** (T1, T1ce, T2, FLAIR). The remaining
cases are missing at least one modality:

| modality | # cases missing it |
|---|---|
{missing_table}

**{n_with_pwi} cases** additionally carry perfusion-weighted imaging
(PWI); **{n_with_seg} cases** carry a per-voxel lesion segmentation.

See ``fig02_modality_coverage.png`` and ``fig03_missingness_heatmap.png``.

The cleaning policy (configurable via ``PreprocessConfig``) is to **keep
cases with missing modalities** but fill the missing channel with zeros,
so the model can learn modality dropout robustness rather than forcing us
to discard ~{drop_share:.0%} of an already small cohort.

## 4. Shape and spacing heterogeneity

Volume shape varies across cases and modalities — the raw cohort is not
shape-aligned. Per-axis shape histograms are in
``fig04_shape_distribution.png``; per-axis voxel-spacing histograms are
in ``fig05_spacing_distribution.png``.

Per-modality summary (median and inter-quartile range):

| modality | median shape | median spacing (mm) |
|---|---|---|
{shape_table}

The preprocessing pipeline resamples every modality to the case's T1ce
reference grid (priority T1ce → T1 → T2 → FLAIR if T1ce is absent), then
crops to the union foreground bounding box and z-scores per modality.

## 5. Storage footprint

Disk usage is dominated by the structural sequences:

| modality | GB |
|---|---|
{bytes_modality_table}

By class:

| class | GB |
|---|---|
{bytes_class_table}

See ``fig06_dataset_storage.png``.

## 6. Intensity ranges

We z-score each modality over its foreground voxels (skull-stripped MR
already has near-zero background). The density plot in
``fig07_intensity_distribution.png`` shows the resulting distributions on
a subset of {sample_n} cases — modalities overlap heavily around zero
(intended) but their tails carry the discriminative signal.

## 7. A representative case

``fig08_modality_panel.png`` shows mid-axial slices through one
preprocessed case in all four modalities, plus the WT / TC / ET
segmentation overlay on T1ce. If this figure is missing, no preprocessed
``.npz`` was available at report time — run
``scripts/preprocess_cohort.py`` first.

## 8. Implications for modelling

* **Class imbalance** is structural, not incidental → use weighted sampling
  and focal-style loss together rather than relying on either alone.
* **Modality dropout is the norm**, not the exception → the model must
  tolerate zero-channels at inference time; train with random modality
  dropout augmentation.
* **Shape and spacing vary** → preprocessing must own resampling and
  cropping; do not rely on raw shape.
* **Segmentation labels are sparse** ({n_with_seg}/{n_kept} cases) → the
  segmentation head is auxiliary; the primary loss target is the
  classification head, with seg contributing where available.

"""


def _format_class_table(stats: CohortStats) -> str:
    items = sorted(stats.class_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "\n".join(f"| {k} | {v} |" for k, v in items)


def _format_missing_table(stats: CohortStats) -> str:
    items = [(m, stats.missing_modality_counts.get(m, 0)) for m in CANONICAL_MODALITIES]
    return "\n".join(f"| {k} | {v} |" for k, v in items)


def _format_shape_table(stats: CohortStats) -> str:
    rows = []
    for m in CANONICAL_MODALITIES:
        shapes = np.array(stats.shape_distribution.get(m, []) or [[0, 0, 0]])
        spc = np.array(stats.spacing_distribution.get(m, []) or [[0.0, 0.0, 0.0]])
        med_shape = np.median(shapes, axis=0).astype(int).tolist()
        med_spc = np.median(spc, axis=0).round(2).tolist()
        rows.append(f"| {m} | {med_shape} | {med_spc} |")
    return "\n".join(rows)


def _format_bytes_table(d: Dict[str, int]) -> str:
    GB = 1024 ** 3
    items = sorted(d.items(), key=lambda kv: -kv[1])
    return "\n".join(f"| {k} | {v / GB:.2f} |" for k, v in items)


def write_markdown(
    stats: CohortStats,
    out_dir: str,
    intensity_sample_n: int = 0,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    GB = 1024 ** 3
    miss_total = sum(stats.missing_modality_counts.values())
    drop_share = miss_total / max(stats.n_kept * len(CANONICAL_MODALITIES), 1)
    body = _REPORT_TEMPLATE.format(
        n_total=stats.n_total,
        total_gb=stats.total_bytes / GB,
        total_bytes=stats.total_bytes,
        n_invalid=stats.n_invalid,
        n_kept=stats.n_kept,
        invalid_ids_str=", ".join(stats.invalid_ids) or "(none)",
        class_table=_format_class_table(stats) or "| (no cases) | 0 |",
        n_complete=stats.n_complete_4mod,
        pct_complete=(100.0 * stats.n_complete_4mod / max(stats.n_kept, 1)),
        missing_table=_format_missing_table(stats),
        n_with_pwi=stats.n_with_pwi,
        n_with_seg=stats.n_with_seg,
        drop_share=drop_share,
        shape_table=_format_shape_table(stats),
        bytes_modality_table=_format_bytes_table(stats.bytes_per_modality) or "| (none) | 0 |",
        bytes_class_table=_format_bytes_table(stats.bytes_per_class) or "| (none) | 0 |",
        sample_n=intensity_sample_n,
    )
    p = out / "cohort_story.md"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_storytelling(
    scans: List[CaseScan],
    out_dir: str,
    representative_npz: Optional[str] = None,
    intensity_sample_n: int = 12,
) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = aggregate(scans)
    fig_class_balance(stats, out)
    fig_modality_coverage(stats, out)
    fig_missingness_heatmap(scans, out)
    fig_shape_distribution(stats, out)
    fig_spacing_distribution(stats, out)
    fig_storage_bar(stats, out)
    fig_invalid_strip(stats, out)
    fig_modality_panel(representative_npz, out)

    intensity_samples = sample_intensities(scans, sample_n=intensity_sample_n)
    fig_intensity_distribution(intensity_samples, out)

    write_markdown(stats, out_dir, intensity_sample_n=intensity_sample_n)

    with open(out / "cohort_stats.json", "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    return stats.to_dict()
