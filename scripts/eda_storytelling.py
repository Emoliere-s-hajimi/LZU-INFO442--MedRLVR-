"""Metadata-only EDA over the raw cohort + a narrative markdown report.

The full Tiantan dump is ~90 GB; this script never loads a full volume
unless asked (``--intensity_sample_n > 0``). It produces:

    <out_dir>/
      cohort_stats.json
      cohort_story.md
      fig01_class_balance.png
      fig02_modality_coverage.png
      fig03_missingness_heatmap.png
      fig04_shape_distribution.png
      fig05_spacing_distribution.png
      fig06_dataset_storage.png
      fig07_intensity_distribution.png       (only if --intensity_sample_n > 0)
      fig08_modality_panel.png               (only if --representative_npz given)
      fig09_invalid_id_strip.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.cohort_scan import scan_dataset, write_scans  # noqa: E402
from src.analysis.storytelling import run_storytelling  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_root", default="data/uncleaned_examples",
                        help="BraTS-style NIfTI root. Defaults to data/uncleaned_examples.")
    parser.add_argument("--structural_root", default=None,
                        help="Optional DICOM dump root (<MOD>/<class>/<case>/*.dcm).")
    parser.add_argument("--pwi_root", default=None,
                        help="Optional PWI DICOM root.")
    parser.add_argument("--out_dir", default="visualization/eda",
                        help="Where to write figures and cohort_story.md. "
                             "Defaults to visualization/eda.")
    parser.add_argument("--representative_npz", default="data/processed/171.npz",
                        help="Optional path to a preprocessed .npz used for "
                             "the modality-panel figure (fig08). Skipped if absent.")
    parser.add_argument("--intensity_sample_n", type=int, default=4,
                        help="If > 0, load this many full NIfTI volumes to "
                             "compute the intensity-density figure.")
    args = parser.parse_args()

    if not any([args.nifti_root, args.structural_root, args.pwi_root]):
        parser.error("Provide at least one of --nifti_root, --structural_root, --pwi_root.")

    scans = scan_dataset(
        nifti_root=args.nifti_root,
        structural_root=args.structural_root,
        pwi_root=args.pwi_root,
    )
    print(f"[scan] {len(scans)} unique case IDs discovered")
    write_scans(scans, str(Path(args.out_dir) / "cohort_scans.json"))

    stats = run_storytelling(
        list(scans.values()),
        out_dir=args.out_dir,
        representative_npz=args.representative_npz,
        intensity_sample_n=args.intensity_sample_n,
    )
    print(json.dumps({
        "n_total": stats["n_total"],
        "n_invalid": stats["n_invalid"],
        "n_kept": stats["n_kept"],
        "out_dir": args.out_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
