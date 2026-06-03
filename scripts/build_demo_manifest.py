"""Build a manifest.json from a folder of cleaned ``.npz`` files.

Useful when working with public sample data (e.g. ``data/some_cleaned_examples/``)
that does not ship with a manifest — turns the folder into a drop-in
input for ``src.train`` / ``src.evaluate``.

Class labels are assigned by a deterministic hash of the case id so the
same case always lands in the same class, while the distribution sits
close to the cohort's 3.5 : 1 recurrence : necrosis ratio.

Usage::

    python scripts/build_demo_manifest.py \\
        --in_dir data/some_cleaned_examples \\
        --out_path data/some_cleaned_examples/manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

import numpy as np


def _label_for(case_id: str) -> str:
    """Deterministic recurrence/necrosis split via SHA-1 hash. Roughly 3.5:1."""
    h = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    return "necrosis" if int(h, 16) % 100 < 22 else "recurrence"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/some_cleaned_examples")
    parser.add_argument("--out_path", default="data/some_cleaned_examples/manifest.json")
    parser.add_argument("--labels", choices=["hash", "all_recurrence", "all_necrosis"],
                        default="hash",
                        help="Label assignment policy. 'hash' is deterministic per case.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    npz_files = sorted(in_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"no .npz files under {in_dir}")

    manifest: List[dict] = []
    for p in npz_files:
        case_id = p.stem
        d = np.load(p)
        image = d["image"]
        label = d["label"]
        shape = list(image.shape[1:])
        missing = []
        for ch, name in enumerate(("t1", "t1ce", "t2", "flair")):
            if ch < image.shape[0] and (image[ch] == 0).all():
                missing.append(name)
        available = [m for m in ("t1", "t1ce", "t2", "flair") if m not in missing]

        if args.labels == "hash":
            label_str = _label_for(case_id)
        elif args.labels == "all_recurrence":
            label_str = "recurrence"
        else:
            label_str = "necrosis"

        manifest.append({
            "case_id": case_id,
            "npz": str(p),
            "label": label_str,
            "available_modalities": available,
            "missing_modalities": missing,
            "shape": shape,
            "has_seg": bool(label.sum() > 0),
        })

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    by_class = {}
    for r in manifest:
        by_class[r["label"]] = by_class.get(r["label"], 0) + 1
    print(json.dumps({
        "n_cases": len(manifest),
        "by_class": by_class,
        "out_path": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
