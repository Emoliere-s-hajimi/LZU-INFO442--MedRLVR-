"""Exploratory data analysis for the cleaned multimodal MRI cohort.

Produces summary tables and figures: class balance, voxel-spacing histograms,
intensity distributions per modality, and per-case lesion-volume statistics.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


def _safe_import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def class_balance(manifest: List[Dict]) -> Counter:
    return Counter(r["label"] for r in manifest)


def modality_intensity_stats(manifest: List[Dict], modalities, sample_n: int = 20) -> Dict[str, Dict[str, float]]:
    import nibabel as nib

    stats: Dict[str, Dict[str, float]] = {m: {"mean": 0.0, "std": 0.0, "n": 0} for m in modalities}
    rng = np.random.default_rng(442)
    sample = rng.choice(len(manifest), size=min(sample_n, len(manifest)), replace=False)
    for idx in sample:
        record = manifest[int(idx)]
        for m in modalities:
            path = record["modalities"].get(m)
            if path is None or not isinstance(path, str):
                continue
            try:
                vol = nib.load(path).get_fdata().astype(np.float32)
            except Exception:
                continue
            fg = vol[vol > 0]
            if fg.size == 0:
                continue
            stats[m]["mean"] += float(fg.mean())
            stats[m]["std"] += float(fg.std())
            stats[m]["n"] += 1
    for m in modalities:
        n = max(stats[m]["n"], 1)
        stats[m]["mean"] /= n
        stats[m]["std"] /= n
    return stats


def lesion_volume_stats(manifest: List[Dict]) -> Dict[str, float]:
    import nibabel as nib

    volumes = []
    for r in manifest:
        seg_path = r.get("seg")
        if seg_path is None or not isinstance(seg_path, str):
            continue
        try:
            seg = nib.load(seg_path).get_fdata()
        except Exception:
            continue
        volumes.append(int((seg > 0).sum()))
    if not volumes:
        return {}
    arr = np.array(volumes, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_voxels": float(arr.mean()),
        "median_voxels": float(np.median(arr)),
        "min_voxels": float(arr.min()),
        "max_voxels": float(arr.max()),
    }


def run_eda(manifest_path: str, modalities, out_dir: str) -> Dict:
    plt = _safe_import_plt()
    with open(manifest_path) as f:
        manifest = json.load(f)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    balance = class_balance(manifest)
    fig, ax = plt.subplots()
    ax.bar(balance.keys(), balance.values())
    ax.set_title("Class balance")
    ax.set_ylabel("# cases")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)

    intensity = modality_intensity_stats(manifest, modalities)
    lesion = lesion_volume_stats(manifest)

    summary = {
        "class_balance": dict(balance),
        "modality_intensity": intensity,
        "lesion_volume": lesion,
    }
    with open(out / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
