"""Run the exploratory data analysis stage on the cleaned manifest."""
from __future__ import annotations

import argparse
import yaml

from src.analysis import run_eda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", default="data/processed/manifest.json")
    parser.add_argument("--out_dir", default="outputs/eda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    summary = run_eda(args.manifest, cfg["data"]["modalities"], args.out_dir)
    print(summary)


if __name__ == "__main__":
    main()
