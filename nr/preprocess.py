"""Offline preprocessing for the NR cohort.

Walks ``SourcePreprocess_SegLabel_202110`` (NIfTI per-patient tree where
the parent folder N/R/RN encodes the class), runs each case through the
shared cleaning pipeline in ``src/data/pipeline.py``, synthesises any
missing modalities, then writes:

  - ``processed/<case_id>.npz``                — image + seg + affine
  - ``processed/train_manifest.json``          — 80 % stratified by class
  - ``processed/val_manifest.json``            — 20 % stratified by class
  - ``processed/preprocess_report.json``       — kept / dropped per split

Usage::

    python -m nr_subproject.nr.preprocess \\
        --config nr_subproject/configs/nr.yaml

Override the root or output on the command line if you don't want to
edit the config.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.data.pipeline import CaseRecord, DropRecord, PreprocessConfig, process_one_case
from src.utils import load_config

from .discover import iter_seg_patients


def _build_preprocess_config(cfg: Dict, out_dir: Path) -> PreprocessConfig:
    pp = cfg.get("preprocess", {})
    return PreprocessConfig(
        out_dir=str(out_dir),
        drop_if_no_modalities=pp.get("drop_if_no_modalities", True),
        drop_if_missing_modality=pp.get("drop_if_missing_modality", False),
        zscore_foreground=pp.get("zscore_foreground", True),
        fg_threshold=pp.get("fg_threshold", 0.0),
        synthesize_missing=pp.get("synthesize_missing", False),
    )


def _stratified_split(
    cases: List[Tuple[str, Path, str]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Tuple[str, Path, str]], List[Tuple[str, Path, str]]]:
    """Per-class shuffle then deterministic slice. Guarantees both splits
    are non-empty as long as each class has at least 2 cases."""
    by_class: Dict[str, List[Tuple[str, Path, str]]] = defaultdict(list)
    for item in cases:
        by_class[item[2]].append(item)

    rng = random.Random(seed)
    train: List[Tuple[str, Path, str]] = []
    val: List[Tuple[str, Path, str]] = []
    for label, bucket in sorted(by_class.items()):
        bucket_sorted = sorted(bucket, key=lambda x: x[0])
        rng.shuffle(bucket_sorted)
        n = len(bucket_sorted)
        n_val = int(round(n * val_fraction))
        if n >= 2:
            n_val = max(1, min(n_val, n - 1))
        else:
            n_val = 0
        val.extend(bucket_sorted[:n_val])
        train.extend(bucket_sorted[n_val:])
    train.sort(key=lambda x: x[0])
    val.sort(key=lambda x: x[0])
    return train, val


def _process_split(
    patients: List[Tuple[str, Path, str]],
    cfg_pp: PreprocessConfig,
    force: bool,
    limit: Optional[int],
    desc: str,
) -> Tuple[List[Dict], List[Dict]]:
    kept: List[Dict] = []
    dropped: List[Dict] = []
    todo = patients[:limit] if limit else patients

    pbar = tqdm(todo, desc=desc, unit="pt")
    for case_id, p_dir, label in pbar:
        pbar.set_postfix_str(case_id)
        npz_path = Path(cfg_pp.out_dir) / f"{case_id}.npz"

        if npz_path.exists() and not force:
            kept.append({
                "case_id": case_id,
                "patient_folder": p_dir.name,
                "npz": str(npz_path),
                "label": label,
                "cached": True,
            })
            continue

        try:
            result = process_one_case(case_id, str(p_dir), label, cfg_pp)
        except Exception as e:
            dropped.append({"case_id": case_id, "patient_folder": p_dir.name,
                            "reason": f"preprocess_error:{e}"})
            continue

        if isinstance(result, CaseRecord):
            row = result.to_dict()
            row["patient_folder"] = p_dir.name
            row["npz"] = row.pop("out_path")
            kept.append(row)
        elif isinstance(result, DropRecord):
            dropped.append({"case_id": case_id, "patient_folder": p_dir.name,
                            "reason": result.reason})
        else:  # pragma: no cover - defensive
            dropped.append({"case_id": case_id, "patient_folder": p_dir.name,
                            "reason": f"unexpected_result_type:{type(result).__name__}"})

    return kept, dropped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="nr_subproject/configs/nr.yaml")
    ap.add_argument("--seg_root", type=str, default=None,
                    help="Override data.seg_root (e.g. server: /root/autodm-tmp/SourcePreprocess_SegLabel_202110)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Override data.processed_dir")
    ap.add_argument("--val_fraction", type=float, default=None,
                    help="Override data.val_fraction")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N patients (smoke test). Split is computed before limiting.")
    ap.add_argument("--force", action="store_true",
                    help="Re-preprocess patients even if the .npz already exists")
    ap.add_argument("--split", choices=("train", "val", "both"), default="both")
    ap.add_argument("--no-prefer-revised", action="store_true",
                    help="Use plain N/ folders instead of N（坏死）(修改版）/N/")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})

    seg_root = args.seg_root or data_cfg.get("seg_root")
    if not seg_root:
        raise SystemExit("missing seg_root (set data.seg_root in config or pass --seg_root)")

    val_fraction = args.val_fraction if args.val_fraction is not None else \
        float(data_cfg.get("val_fraction", 0.2))
    prefer_revised = (not args.no_prefer_revised) and bool(data_cfg.get("prefer_revised", True))
    seed = int(cfg.get("project", {}).get("seed", 442))

    out_root = Path(args.out_dir or data_cfg.get("processed_dir",
                                                 "nr_subproject/processed"))
    out_root.mkdir(parents=True, exist_ok=True)

    cases = list(iter_seg_patients(seg_root, prefer_revised=prefer_revised))
    print(f"discovered {len(cases)} patients under {seg_root}")
    train_cases, val_cases = _stratified_split(cases, val_fraction, seed)
    print(f"split: train={len(train_cases)}  val={len(val_cases)}  "
          f"(val_fraction={val_fraction}, seed={seed})")

    report: Dict[str, Dict] = {}

    cfg_pp = _build_preprocess_config(cfg, out_root)
    cfg_pp.require_seg = False  # some patients lack a seg file; still keep them
    # All npz land in the same processed_dir; manifests partition them by split.

    if args.split in ("train", "both"):
        kept, dropped = _process_split(
            patients=train_cases, cfg_pp=cfg_pp, force=args.force,
            limit=args.limit, desc="train",
        )
        with open(out_root / "train_manifest.json", "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        report["train"] = {"kept": len(kept), "dropped": len(dropped),
                           "drop_records": dropped}
        print(f"[train] kept {len(kept)}  dropped {len(dropped)}")

    if args.split in ("val", "both"):
        kept, dropped = _process_split(
            patients=val_cases, cfg_pp=cfg_pp, force=args.force,
            limit=args.limit, desc="val",
        )
        with open(out_root / "val_manifest.json", "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)
        report["val"] = {"kept": len(kept), "dropped": len(dropped),
                         "drop_records": dropped}
        print(f"[val] kept {len(kept)}  dropped {len(dropped)}")

    report["config"] = {
        "seg_root": seg_root,
        "prefer_revised": prefer_revised,
        "val_fraction": val_fraction,
        "seed": seed,
        "synthesize_missing": cfg_pp.synthesize_missing,
    }
    with open(out_root / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
