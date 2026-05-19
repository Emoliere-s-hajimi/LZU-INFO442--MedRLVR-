"""Metadata-only scan over the raw cohort.

The full Tiantan dump is ~90 GB. Loading every voxel just to write an EDA
report is wasteful, so this module walks the tree reading only NIfTI
headers and a single DICOM per series. It produces one ``CaseScan`` per
case which downstream storytelling code consumes.

The scanner accepts three roots:
    * ``nifti_root``     — BraTS-style ``<case>/BraTS2021_<id>_<mod>.nii.gz``
    * ``structural_root`` — DICOM tree ``{MOD}/{CLASS}/{case}/*.dcm``
    * ``pwi_root``        — DICOM tree ``PWI/{case}/<series>/*.dcm``

Any of them can be ``None``.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..data.modality_map import CANONICAL_MODALITIES, parse_chinese_label
from ..data.pipeline import INVALID_CASE_IDS, _case_id_from_dir


@dataclass
class CaseScan:
    case_id: str
    label: Optional[str] = None
    invalid_reason: Optional[str] = None
    # per-modality metadata
    modality_shape: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    modality_spacing: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    modality_source: Dict[str, str] = field(default_factory=dict)
    modality_bytes: Dict[str, int] = field(default_factory=dict)
    # convenience: presence flags
    has_pwi: bool = False
    has_seg: bool = False
    # set later: full-volume intensity stats, from a subsample of cases
    intensity_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def available_modalities(self) -> List[str]:
        return [m for m in CANONICAL_MODALITIES if m in self.modality_shape]

    @property
    def missing_modalities(self) -> List[str]:
        return [m for m in CANONICAL_MODALITIES if m not in self.modality_shape]

    def to_dict(self) -> Dict:
        d = asdict(self)
        for key in ("modality_shape", "modality_spacing"):
            d[key] = {k: list(v) for k, v in d[key].items()}
        d["available_modalities"] = self.available_modalities
        d["missing_modalities"] = self.missing_modalities
        return d


# ---------------------------------------------------------------------------
# NIfTI scan
# ---------------------------------------------------------------------------

def _scan_nifti_file(path: Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    shape = tuple(int(s) for s in img.shape[:3])  # type: ignore[arg-type]
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return shape, zooms


def _label_from_path(path: Path) -> Optional[str]:
    for part in path.parts[::-1]:
        lab = parse_chinese_label(part)
        if lab is not None:
            return lab
    return None


def scan_nifti_root(root: str) -> Dict[str, CaseScan]:
    """Per-case scan of a BraTS-style NIfTI dump."""
    from ..data.modality_map import classify_filename

    cases: Dict[str, CaseScan] = {}
    root_p = Path(root)
    if not root_p.exists():
        return cases

    for d in sorted(root_p.iterdir()):
        if not d.is_dir():
            continue
        cid = _case_id_from_dir(d.name)
        scan = cases.setdefault(cid, CaseScan(case_id=cid))
        if cid in INVALID_CASE_IDS:
            scan.invalid_reason = "invalid_id_not_irradiated"
        scan.label = scan.label or _label_from_path(d)
        for f in d.iterdir():
            if not f.is_file():
                continue
            n = f.name.lower()
            if not (n.endswith(".nii") or n.endswith(".nii.gz")):
                continue
            if "seg" in n or "mask" in n or "label" in n:
                scan.has_seg = True
                continue
            mod = classify_filename(f.name)
            if mod is None:
                continue
            try:
                shape, spacing = _scan_nifti_file(f)
            except Exception:
                continue
            scan.modality_shape[mod] = shape
            scan.modality_spacing[mod] = spacing
            scan.modality_source[mod] = str(f)
            try:
                scan.modality_bytes[mod] = f.stat().st_size
            except OSError:
                scan.modality_bytes[mod] = 0
    return cases


# ---------------------------------------------------------------------------
# DICOM scan (structural)
# ---------------------------------------------------------------------------

_STRUCT_DIR_MAP = {
    "FLAIR": "flair", "flair": "flair",
    "T1": "t1", "t1": "t1",
    "T1ce": "t1ce", "T1CE": "t1ce", "T1c": "t1ce", "t1ce": "t1ce", "t1c": "t1ce",
    "T2": "t2", "t2": "t2",
}


def _scan_dicom_series(series_dir: Path) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], int]:
    """Cheap DICOM-series scan: read header of first slice, count files for
    slice axis, sum byte size.
    """
    import pydicom

    files = sorted([f for f in series_dir.iterdir() if f.is_file()])
    if not files:
        return (0, 0, 0), (1.0, 1.0, 1.0), 0
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return (0, 0, 0), (1.0, 1.0, 1.0), sum(f.stat().st_size for f in files)
    rows = int(getattr(ds, "Rows", 0) or 0)
    cols = int(getattr(ds, "Columns", 0) or 0)
    ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
    thk = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
    return (len(files), rows, cols), (thk, float(ps[0]), float(ps[1])), sum(f.stat().st_size for f in files)


def scan_structural_dicom_root(root: str) -> Dict[str, CaseScan]:
    cases: Dict[str, CaseScan] = {}
    root_p = Path(root)
    if not root_p.exists():
        return cases

    for mod_dir in root_p.iterdir():
        if not mod_dir.is_dir():
            continue
        mod = _STRUCT_DIR_MAP.get(mod_dir.name)
        if mod is None:
            continue
        for class_dir in mod_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = parse_chinese_label(class_dir.name)
            for case_dir in class_dir.iterdir():
                if not case_dir.is_dir():
                    continue
                cid = _case_id_from_dir(case_dir.name)
                scan = cases.setdefault(cid, CaseScan(case_id=cid))
                scan.label = scan.label or label
                if cid in INVALID_CASE_IDS:
                    scan.invalid_reason = "invalid_id_not_irradiated"
                shape, spacing, byte_size = _scan_dicom_series(case_dir)
                scan.modality_shape[mod] = shape
                scan.modality_spacing[mod] = spacing
                scan.modality_source[mod] = str(case_dir)
                scan.modality_bytes[mod] = byte_size
    return cases


def scan_pwi_root(root: str) -> Dict[str, CaseScan]:
    cases: Dict[str, CaseScan] = {}
    root_p = Path(root)
    if not root_p.exists():
        return cases
    for case_dir in root_p.iterdir():
        if not case_dir.is_dir():
            continue
        cid = _case_id_from_dir(case_dir.name)
        scan = cases.setdefault(cid, CaseScan(case_id=cid))
        scan.has_pwi = True
    return cases


# ---------------------------------------------------------------------------
# Merge + write
# ---------------------------------------------------------------------------

def merge_scans(*partial: Dict[str, CaseScan]) -> Dict[str, CaseScan]:
    merged: Dict[str, CaseScan] = {}
    for part in partial:
        for cid, scan in part.items():
            if cid not in merged:
                merged[cid] = scan
                continue
            cur = merged[cid]
            cur.label = cur.label or scan.label
            cur.invalid_reason = cur.invalid_reason or scan.invalid_reason
            cur.has_pwi = cur.has_pwi or scan.has_pwi
            cur.has_seg = cur.has_seg or scan.has_seg
            for m, shape in scan.modality_shape.items():
                if m not in cur.modality_shape:
                    cur.modality_shape[m] = shape
                    cur.modality_spacing[m] = scan.modality_spacing.get(m, (1.0, 1.0, 1.0))
                    cur.modality_source[m] = scan.modality_source.get(m, "")
                    cur.modality_bytes[m] = scan.modality_bytes.get(m, 0)
    return merged


def write_scans(scans: Dict[str, CaseScan], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [scans[k].to_dict() for k in sorted(scans.keys())]
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def scan_dataset(
    nifti_root: Optional[str] = None,
    structural_root: Optional[str] = None,
    pwi_root: Optional[str] = None,
) -> Dict[str, CaseScan]:
    parts: List[Dict[str, CaseScan]] = []
    if nifti_root:
        parts.append(scan_nifti_root(nifti_root))
    if structural_root:
        parts.append(scan_structural_dicom_root(structural_root))
    if pwi_root:
        parts.append(scan_pwi_root(pwi_root))
    return merge_scans(*parts)
