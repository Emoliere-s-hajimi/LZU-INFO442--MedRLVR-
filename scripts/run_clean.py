"""Run the cleaning pipeline against the raw private cohort."""
from __future__ import annotations

import argparse

from src.data.cleaning import CleaningPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_root", default="data/processed")
    args = parser.parse_args()

    pipe = CleaningPipeline(raw_root=args.raw_root, out_root=args.out_root)
    report = pipe.run()
    print(f"kept {report.kept_cases}/{report.total_cases} cases")
    print("class counts:", report.class_counts)
    if report.dropped_cases:
        print(f"dropped {len(report.dropped_cases)} cases — see {args.out_root}/cleaning_report.json")


if __name__ == "__main__":
    main()
