"""Cleaning pipeline for the raw private Tiantan glioma post-radiation cohort.

The raw cohort contains heterogeneous per-case folders with mixed modality naming,
inconsistent voxel spacing, occasional missing modalities, and a non-trivial
class imbalance between recurrence and radiation-necrosis cases. This module
provides a single entry point that turns the raw dump into a curated manifest
with a uniform on-disk layout.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


CANONICAL_MODALITIES = ["t1", "t1ce", "t2", "flair"]
MODALITY_ALIASES = {
    "t1": {"t1", "t1w", "t1_pre", "pret1"},
    "t1ce": {"t1ce", "t1c", "t1gd", "t1_post", "post_t1", "t1+c"},
    "t2": {"t2", "t2w"},
    "flair": {"flair", "t2flair", "fla"},
}


@dataclass
class CleaningReport:
    total_cases: int = 0
    kept_cases: int = 0
    dropped_cases: List[Dict] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_cases": self.total_cases,
            "kept_cases": self.kept_cases,
            "dropped_cases": self.dropped_cases,
            "class_counts": self.class_counts,
        }


def _detect_modality(filename: str) -> Optional[str]:
    name = filename.lower()
    for canonical, aliases in MODALITY_ALIASES.items():
        for token in aliases:
            if re.search(rf"(^|[_\-]){re.escape(token)}([_\-\.]|$)", name):
                return canonical
    return None


def _infer_label_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    if any("recur" in p or p == "r" for p in parts):
        return "recurrence"
    if any("necro" in p or "rn" in p or p == "n" for p in parts):
        return "necrosis"
    return None


class CleaningPipeline:
    """Walks a raw case directory tree and produces a clean manifest."""

    def __init__(
        self,
        raw_root: str,
        out_root: str,
        required_modalities: Optional[List[str]] = None,
        min_lesion_voxels: int = 50,
        write_manifest: bool = True,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.required_modalities = required_modalities or CANONICAL_MODALITIES
        self.min_lesion_voxels = min_lesion_voxels
        self.write_manifest = write_manifest

    def discover_cases(self) -> List[Path]:
        cases: List[Path] = []
        for path in sorted(self.raw_root.rglob("*")):
            if not path.is_dir():
                continue
            files = list(path.glob("*"))
            modalities_found = {_detect_modality(f.name) for f in files if f.is_file()}
            if any(m is not None for m in modalities_found):
                cases.append(path)
        return cases

    def _index_case(self, case_dir: Path) -> Dict:
        record: Dict = {"case_id": case_dir.name, "path": str(case_dir), "modalities": {}, "label": None, "seg": None}
        for f in case_dir.iterdir():
            if not f.is_file():
                continue
            modality = _detect_modality(f.name)
            if modality is not None:
                record["modalities"][modality] = str(f)
            if "seg" in f.name.lower() or "mask" in f.name.lower() or "label" in f.name.lower():
                record["seg"] = str(f)
        record["label"] = _infer_label_from_path(case_dir)
        return record

    def _validate(self, record: Dict) -> Optional[str]:
        missing = [m for m in self.required_modalities if m not in record["modalities"]]
        if missing:
            return f"missing_modalities:{','.join(missing)}"
        if record["label"] is None:
            return "unknown_label"
        return None

    def run(self) -> CleaningReport:
        report = CleaningReport()
        manifest: List[Dict] = []
        for case_dir in self.discover_cases():
            report.total_cases += 1
            record = self._index_case(case_dir)
            reason = self._validate(record)
            if reason is not None:
                report.dropped_cases.append({"case_id": record["case_id"], "reason": reason})
                continue
            manifest.append(record)
            report.kept_cases += 1
            report.class_counts[record["label"]] = report.class_counts.get(record["label"], 0) + 1

        if self.write_manifest:
            self.out_root.mkdir(parents=True, exist_ok=True)
            with open(self.out_root / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            with open(self.out_root / "cleaning_report.json", "w") as f:
                json.dump(report.to_dict(), f, indent=2)
        return report


def load_manifest(processed_root: str) -> List[Dict]:
    with open(Path(processed_root) / "manifest.json") as f:
        return json.load(f)
