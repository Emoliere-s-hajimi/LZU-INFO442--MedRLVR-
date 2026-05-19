"""Fill missing modality channels in cleaned ``.npz`` cases via donor-based
synthesis. Three zero-training strategies plus a hook for a learned model:

    atlas_mean          — average of donor cases, brain-masked
    linear_regression   — ridge regression observed → missing (default)
    patch_knn           — voxel-wise nearest-neighbour lookup in
                          observed-modality intensity space

Donors are cleaned ``.npz`` cases that already carry the full 4-modality
set. They are auto-detected from ``--donor_dir``.

A ``--in_dir`` folder of cleaned cases is scanned, every case with at
least one zero channel is run through the chosen strategy, and the
output is written to ``--out_dir`` together with per-case
``<id>_synth.json`` reports and a cohort-level ``synthesis_report.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthesis import DonorPool, synthesize_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="data/processed",
                        help="Folder of cleaned .npz cases (some with missing channels). "
                             "Defaults to data/processed.")
    parser.add_argument("--donor_dir", default="data/processed",
                        help="Folder of cleaned .npz cases used as donor pool. "
                             "Incomplete donors are filtered out automatically. "
                             "Defaults to data/processed (same folder as --in_dir).")
    parser.add_argument("--out_dir", default="data/processed_synth",
                        help="Where to write synthesized .npz + per-case reports. "
                             "Defaults to data/processed_synth.")
    parser.add_argument("--viz_dir", default="visualization/synthesis",
                        help="Where to write per-case before/after panel PNGs. "
                             "Defaults to visualization/synthesis.")
    parser.add_argument("--strategy", default="linear_regression",
                        choices=["atlas_mean", "linear_regression", "patch_knn"])
    parser.add_argument("--max_donors", type=int, default=None,
                        help="If set, only use the first N donor cases (cheaper).")
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="Ridge penalty for the linear_regression strategy.")
    parser.add_argument("--subsample_voxels", type=int, default=200000,
                        help="Voxel subsample for linear_regression fit.")
    parser.add_argument("--knn_k", type=int, default=5,
                        help="k for the patch_knn strategy.")
    parser.add_argument("--max_donor_voxels", type=int, default=200000,
                        help="Max donor voxels held in memory for patch_knn.")
    args = parser.parse_args()

    donors = DonorPool.from_dir(args.donor_dir, max_donors=args.max_donors)
    kwargs = {}
    if args.strategy == "linear_regression":
        kwargs.update(ridge=args.ridge, subsample_voxels=args.subsample_voxels)
    elif args.strategy == "patch_knn":
        kwargs.update(k=args.knn_k, max_donor_voxels=args.max_donor_voxels)

    reports = synthesize_cohort(
        in_dir=args.in_dir,
        donors=donors,
        strategy=args.strategy,
        out_dir=args.out_dir,
        viz_dir=args.viz_dir,
        **kwargs,
    )
    summary = {
        "n_total": len(reports),
        "n_synthesized": sum(1 for r in reports if r.strategy != "noop_complete"),
        "n_complete": sum(1 for r in reports if r.strategy == "noop_complete"),
        "out_dir": args.out_dir,
        "strategy": args.strategy,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
