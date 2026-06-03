"""Run every case-study script over one or more cleaned ``.npz`` files.

Usage::

    # Single case
    python -m fv.case_study.run_case_study --npz data/some_cleaned_examples/001.npz

    # All cases under a folder
    python -m fv.case_study.run_case_study --in_dir data/some_cleaned_examples

    # Custom output root
    python -m fv.case_study.run_case_study --in_dir data/processed \\
        --out_root visualization/case_study
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from . import anatomy, helpers, modality_signature, morphology, topology, tumor_3d


SCRIPTS = [
    ("anatomy",            anatomy.render),
    ("tumor_3d",           tumor_3d.render),
    ("topology",           topology.render),
    ("morphology",         morphology.render),
    ("modality_signature", modality_signature.render),
]


def run_one(npz_path: str, out_root: str | Path | None = None) -> dict:
    case_id = helpers.case_id_from_path(npz_path)
    results = {"case_id": case_id, "figures": {}, "errors": {}}
    for name, fn in SCRIPTS:
        try:
            out = fn(npz_path, out_root=out_root)
            results["figures"][name] = str(out)
            print(f"  ✓ {name}: {out}")
        except Exception as e:
            results["errors"][name] = f"{type(e).__name__}: {e}"
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=None, help="One .npz file")
    p.add_argument("--in_dir", default=None, help="Folder of .npz files")
    p.add_argument("--out_root", default=None,
                   help="Defaults to visualization/case_study/")
    args = p.parse_args()

    cases: List[str] = []
    if args.npz:
        cases.append(args.npz)
    if args.in_dir:
        cases.extend(str(p) for p in sorted(Path(args.in_dir).glob("*.npz")))
    if not cases:
        raise SystemExit("provide --npz or --in_dir")

    all_results = []
    for npz in cases:
        print(f"\n--- {npz} ---")
        all_results.append(run_one(npz, out_root=args.out_root))

    out_root = Path(args.out_root) if args.out_root else helpers.DEFAULT_OUT_ROOT
    summary_path = out_root / "case_study_summary.json"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
