"""Mine morphological + topological priors from the cleaned ``.npz`` cohort.

Produces:
    morphology_features.csv            — per-case flat feature table
    morphology_story.md                — narrative of the priors found
    morph_class_tests.json             — Mann-Whitney U tests recurrence vs necrosis
    morph_fig01..08*.png               — figures (see src/analysis/morphology.py)

Either a preprocessing manifest (so per-case class labels are known) or
a flat folder of ``.npz`` files can be passed; the manifest gives
class-stratified figures, the flat folder collapses everything into one
bucket.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.morphology import run_morphology_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.json",
                        help="Path to manifest.json from scripts/preprocess_cohort.py. "
                             "Defaults to data/processed/manifest.json.")
    parser.add_argument("--npz_dir", default="data/processed",
                        help="Folder of cleaned .npz files (used when manifest is absent). "
                             "Defaults to data/processed.")
    parser.add_argument("--out_dir", default="visualization/morphology",
                        help="Where to write feature CSV, figures and morphology_story.md. "
                             "Defaults to visualization/morphology.")
    args = parser.parse_args()

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        with open(args.manifest) as f:
            manifest = json.load(f)

    summary = run_morphology_analysis(
        manifest=manifest,
        npz_dir=args.npz_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
