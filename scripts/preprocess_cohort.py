"""Map raw NIfTI / DICOM cases into the cleaned ``.npz`` format.

Two input shapes are supported:

    --nifti_root <dir>       — BraTS-style per-case folders. Each folder may
                               contain any subset of {t1, t1ce, t2, flair}
                               NIfTI files and an optional seg file.

    --structural_root <dir>  — DICOM dump in the
                               ``<root>/<MOD>/<class>/<case>/*.dcm``
                               layout used by Tiantan's hand-off.

If both are passed, NIfTI cases are processed first and DICOM cases are
processed after, with the same ``--out_dir`` so the per-case ``<id>.npz``
files end up next to each other.

Output layout::

    <out_dir>/
      <case_id>.npz          # image (4,H,W,D) + label (3,H,W,D) + affine
      manifest.json          # kept cases (case_id, npz, label, modalities)
      preprocess_report.json # full kept + dropped record
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import (  # noqa: E402
    PreprocessConfig,
    PreprocessReport,
    discover_dicom_cases,
    discover_nifti_cases,
    process_cohort,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_root", default="data/uncleaned_examples",
                        help="BraTS-style NIfTI per-case folder root. "
                             "Defaults to data/uncleaned_examples.")
    parser.add_argument("--structural_root", default=None,
                        help="DICOM dump root (<MOD>/<class>/<case>/*.dcm).")
    parser.add_argument("--out_dir", default="data/processed",
                        help="Destination for <case_id>.npz files and manifest. "
                             "Defaults to data/processed.")
    parser.add_argument("--drop_if_missing_modality", action="store_true",
                        help="If set, cases missing any of the 4 modalities are dropped. "
                             "Default keeps the case and zero-fills the channel.")
    parser.add_argument("--require_seg", action="store_true",
                        help="If set, cases without a segmentation file are dropped.")
    args = parser.parse_args()

    if not args.nifti_root and not args.structural_root:
        parser.error("Provide at least one of --nifti_root or --structural_root.")

    cfg = PreprocessConfig(
        out_dir=args.out_dir,
        drop_if_missing_modality=args.drop_if_missing_modality,
        require_seg=args.require_seg,
    )

    combined = PreprocessReport()

    if args.nifti_root:
        nifti_cases = discover_nifti_cases(args.nifti_root)
        print(f"[nifti] {len(nifti_cases)} candidate cases under {args.nifti_root}")
        r = process_cohort(nifti_cases, cfg, is_dicom_manifest=False)
        combined.kept.extend(r.kept)
        combined.dropped.extend(r.dropped)
        print(f"[nifti] kept={len(r.kept)}  dropped={len(r.dropped)}")

    if args.structural_root:
        dicom_cases = discover_dicom_cases(args.structural_root)
        print(f"[dicom] {len(dicom_cases)} candidate cases under {args.structural_root}")
        r = process_cohort(dicom_cases, cfg, is_dicom_manifest=True)
        combined.kept.extend(r.kept)
        combined.dropped.extend(r.dropped)
        print(f"[dicom] kept={len(r.kept)}  dropped={len(r.dropped)}")

    write_report(combined, args.out_dir)
    print(json.dumps({
        "kept": len(combined.kept),
        "dropped": len(combined.dropped),
        "out_dir": args.out_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
