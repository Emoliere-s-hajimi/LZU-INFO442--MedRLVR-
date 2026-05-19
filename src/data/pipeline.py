"""Per-case preprocessing pipeline that maps a raw case into the cleaned
``.npz`` format observed in ``data/some_cleaned_examples/``.

Target file layout::

    <case_id>.npz
      image  : float32, shape (4, H, W, D) — channel order [t1, t1ce, t2, flair]
      label  : uint8,   shape (3, H, W, D) — BraTS channels [WT, TC, ET]
      affine : float64, shape (4, 4)

Pipeline (per case):
    1. discover input files (NIfTI or DICOM)
    2. drop case if its ID is on the invalid-ID list
    3. load each modality; if a modality is missing we still keep the case
       (zeros channel) but the case's record carries a ``missing`` list
    4. choose a reference modality (priority: t1ce > t1 > t2 > flair) and
       resample every other modality to that reference grid
    5. crop to the foreground bounding box (union of foregrounds across
       available modalities)
    6. apply z-score normalisation per modality over its foreground voxels
    7. convert raw BraTS-style seg labels {0, 1, 2, 4} to the 3-channel
       [WT, TC, ET] one-hot representation
    8. save ``.npz``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .modality_map import CANONICAL_MODALITIES, parse_chinese_label
from .raw_io import (
    ModalityVolume,
    discover_segmentation,
    find_modalities_in_case_dir,
    load_modality,
)

# Invalid case IDs documented by the clinical collaborators: these
# patients did NOT undergo radiation therapy, so they violate the
# post-radiation inclusion criterion of the study and are excluded
# before any preprocessing runs. The list is fixed by the data hand-off.
INVALID_CASE_IDS: frozenset = frozenset({
    "047", "107", "214", "225", "311", "350", "354",
    "358", "374", "462", "463", "475", "481",
})

# Reference-modality preference for resampling
REFERENCE_PRIORITY = ("t1ce", "t1", "t2", "flair")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class CaseRecord:
    case_id: str
    out_path: str
    label: Optional[str]
    available_modalities: List[str]
    missing_modalities: List[str]
    shape: Tuple[int, int, int]
    reference_modality: str
    has_seg: bool
    raw_input: str

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "out_path": self.out_path,
            "label": self.label,
            "available_modalities": self.available_modalities,
            "missing_modalities": self.missing_modalities,
            "shape": list(self.shape),
            "reference_modality": self.reference_modality,
            "has_seg": self.has_seg,
            "raw_input": self.raw_input,
        }


@dataclass
class DropRecord:
    case_id: str
    reason: str
    raw_input: str

    def to_dict(self) -> Dict:
        return {"case_id": self.case_id, "reason": self.reason, "raw_input": self.raw_input}


@dataclass
class PreprocessReport:
    kept: List[CaseRecord] = field(default_factory=list)
    dropped: List[DropRecord] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "kept": [k.to_dict() for k in self.kept],
            "dropped": [d.to_dict() for d in self.dropped],
            "summary": {
                "n_kept": len(self.kept),
                "n_dropped": len(self.dropped),
            },
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _resample_to_grid(
    src: ModalityVolume,
    target_shape: Tuple[int, int, int],
    order: int = 1,
) -> np.ndarray:
    """Trilinear resample ``src.data`` to ``target_shape``.

    Cohort modalities arrive already roughly co-registered (the Tiantan
    NIfTI dumps follow the BraTS pipeline), so a simple shape-aligned zoom
    is adequate. For more rigorous cross-modality alignment, plug
    ``src/data/registration.py`` into the dataclass instead.
    """
    from scipy.ndimage import zoom

    if tuple(src.data.shape) == tuple(target_shape):
        return src.data.astype(np.float32, copy=False)
    factors = [t / s for t, s in zip(target_shape, src.data.shape)]
    return zoom(src.data, factors, order=order).astype(np.float32, copy=False)


def _bbox_union(masks: List[np.ndarray]) -> Tuple[slice, slice, slice]:
    if not masks:
        raise ValueError("empty mask list")
    acc = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        acc |= m
    if not acc.any():
        return tuple(slice(0, s) for s in acc.shape)  # type: ignore[return-value]
    coords = np.argwhere(acc)
    mn = coords.min(0)
    mx = coords.max(0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))  # type: ignore[return-value]


def _zscore_with_foreground_stats(crop: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Z-score using foreground statistics but apply to the whole crop.

    Matches the reference cleaned-cohort behaviour: background voxels
    keep their original 0, which after this transform becomes
    ``(0 - mean_fg) / std_fg`` — a strongly negative number that the
    network can still distinguish from a zero-filled missing channel
    (which stays exactly 0 because it is never z-scored).
    """
    crop = crop.astype(np.float32, copy=False)
    fg = crop > threshold
    if not fg.any():
        return crop
    mu = float(crop[fg].mean())
    sigma = float(crop[fg].std()) + 1e-8
    return (crop - mu) / sigma


def _seg_to_brats_channels(seg: np.ndarray) -> np.ndarray:
    """Convert raw BraTS seg labels to the 3-channel [WT, TC, ET] format.

    Raw label convention: 0 background, 1 NCR/NET, 2 ED, 4 ET.

      WT (whole tumor) = any label > 0
      TC (tumor core)  = labels {1, 4}
      ET (enhancing)   = label 4

    For Tiantan, where the raw seg may only mark a single tumour mass with
    binary {0, 1}, all three channels collapse to that mask, which is the
    behaviour we want for downstream Dice training.
    """
    seg = seg.astype(np.int32)
    wt = (seg > 0)
    tc = (seg == 1) | (seg == 4)
    et = (seg == 4)
    if not tc.any() and wt.any():
        tc = wt
    if not et.any() and wt.any():
        et = wt
    return np.stack([wt, tc, et], axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-case processor
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    out_dir: str
    drop_if_no_modalities: bool = True
    require_seg: bool = False
    # If True, a missing modality fails the case. If False, the missing
    # channel is filled with zeros and the case is still emitted.
    drop_if_missing_modality: bool = False
    zscore_foreground: bool = True
    fg_threshold: float = 0.0  # foreground = voxels strictly above this


def process_one_case(
    case_id: str,
    case_dir: str,
    label: Optional[str],
    cfg: PreprocessConfig,
) -> CaseRecord | DropRecord:
    if case_id in INVALID_CASE_IDS:
        return DropRecord(case_id=case_id, reason="invalid_id_not_irradiated", raw_input=case_dir)

    mod_paths = find_modalities_in_case_dir(case_dir)
    if not mod_paths and cfg.drop_if_no_modalities:
        return DropRecord(case_id=case_id, reason="no_recognised_modalities", raw_input=case_dir)

    seg_path = discover_segmentation(case_dir)
    if cfg.require_seg and seg_path is None:
        return DropRecord(case_id=case_id, reason="missing_segmentation", raw_input=case_dir)

    available = [m for m in CANONICAL_MODALITIES if m in mod_paths]
    missing = [m for m in CANONICAL_MODALITIES if m not in mod_paths]

    if missing and cfg.drop_if_missing_modality:
        return DropRecord(
            case_id=case_id,
            reason=f"missing_modalities:{','.join(missing)}",
            raw_input=case_dir,
        )

    ref_mod = next((m for m in REFERENCE_PRIORITY if m in mod_paths), None)
    if ref_mod is None:
        return DropRecord(case_id=case_id, reason="no_reference_modality", raw_input=case_dir)

    ref_vol = load_modality(mod_paths[ref_mod])
    target_shape = tuple(ref_vol.data.shape)
    affine = ref_vol.affine

    channels: Dict[str, np.ndarray] = {}
    channels[ref_mod] = ref_vol.data.astype(np.float32, copy=False)
    for m in available:
        if m == ref_mod:
            continue
        vol = load_modality(mod_paths[m])
        channels[m] = _resample_to_grid(vol, target_shape, order=1)

    # Foreground bbox is the union of per-modality foregrounds
    fg_masks = [c > cfg.fg_threshold for c in channels.values()]
    bbox = _bbox_union(fg_masks)

    # Cropped + z-scored stack in canonical channel order
    image = np.zeros((len(CANONICAL_MODALITIES),) + tuple(s.stop - s.start for s in bbox),
                    dtype=np.float32)
    for i, m in enumerate(CANONICAL_MODALITIES):
        if m in channels:
            crop = channels[m][bbox]
            if cfg.zscore_foreground:
                crop = _zscore_with_foreground_stats(crop)
            image[i] = crop
        # else: channel stays zeros (missing modality marker)

    # Segmentation
    if seg_path is not None:
        seg_vol = load_modality(seg_path)
        seg_resampled = _resample_to_grid(seg_vol, target_shape, order=0)
        seg_cropped = seg_resampled[bbox]
        label_arr = _seg_to_brats_channels(np.round(seg_cropped).astype(np.int32))
    else:
        label_arr = np.zeros((3,) + image.shape[1:], dtype=np.uint8)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.npz"
    np.savez_compressed(out_path, image=image, label=label_arr, affine=affine.astype(np.float64))

    return CaseRecord(
        case_id=case_id,
        out_path=str(out_path),
        label=label,
        available_modalities=available,
        missing_modalities=missing,
        shape=tuple(image.shape[1:]),
        reference_modality=ref_mod,
        has_seg=seg_path is not None,
        raw_input=case_dir,
    )


# ---------------------------------------------------------------------------
# Cohort-level entry points
# ---------------------------------------------------------------------------

def _case_id_from_dir(dir_name: str) -> str:
    """Extract the canonical 3-digit case ID from a folder name.

    Accepts both raw folder names (``006``, ``225``) and BraTS-style
    NIfTI parents (``BraTS2021_00171``).
    """
    import re

    m = re.search(r"(\d{3,5})", dir_name)
    if not m:
        return dir_name
    s = m.group(1)
    # canonicalise to 3 digits (006, not 6 or 00006)
    return s.lstrip("0").zfill(3) if int(s) > 0 else "000"


def discover_nifti_cases(root: str) -> List[Tuple[str, str, Optional[str]]]:
    """Walk a NIfTI-flavoured root (``data/uncleaned_examples`` style) and
    return ``[(case_id, case_dir, inferred_label_or_None), ...]``.
    """
    root_p = Path(root)
    out: List[Tuple[str, str, Optional[str]]] = []
    if not root_p.exists():
        return out
    for d in sorted(root_p.iterdir()):
        if not d.is_dir():
            continue
        # Has any .nii / .nii.gz under it?
        if any(f.suffix in {".nii"} or f.name.endswith(".nii.gz") for f in d.iterdir() if f.is_file()):
            out.append((_case_id_from_dir(d.name), str(d), None))
    return out


def discover_dicom_cases(structural_root: str) -> List[Tuple[str, str, Optional[str]]]:
    """Walk the DICOM dump (``4个常规结构像/{MOD}/{CLASS}/{case}/*.dcm``)
    and aggregate per-case modality folders into a single virtual case dir.

    Because the modality folders are not co-located on disk, we materialise
    a temporary on-the-fly view in ``process_one_case`` via a small
    side-effect: we collapse the structure by writing a manifest dict
    ``{modality: dicom_dir}`` and then call ``process_one_case`` with that.

    This function returns ``(case_id, virtual_case_dir, label)`` where
    ``virtual_case_dir`` is a Path-like that ``find_modalities_in_case_dir``
    would not find anything in. The cohort-level driver below substitutes
    in the discovered modality paths directly, sidestepping that issue.
    """
    structural_root = Path(structural_root)
    out: List[Tuple[str, str, Optional[str]]] = []
    if not structural_root.exists():
        return out
    by_case: Dict[str, Dict[str, str]] = {}
    label_of: Dict[str, Optional[str]] = {}

    canonical_dir_map = {
        "FLAIR": "flair", "flair": "flair",
        "T1": "t1", "t1": "t1",
        "T1ce": "t1ce", "T1CE": "t1ce", "T1c": "t1ce", "t1ce": "t1ce", "t1c": "t1ce",
        "T2": "t2", "t2": "t2",
    }

    for mod_dir in structural_root.iterdir():
        if not mod_dir.is_dir():
            continue
        mod = canonical_dir_map.get(mod_dir.name)
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
                by_case.setdefault(cid, {})[mod] = str(case_dir)
                if label is not None:
                    label_of[cid] = label

    for cid, mods in by_case.items():
        out.append((cid, json.dumps(mods), label_of.get(cid)))
    return out


def process_cohort(
    cases: List[Tuple[str, str, Optional[str]]],
    cfg: PreprocessConfig,
    is_dicom_manifest: bool = False,
) -> PreprocessReport:
    """Run ``process_one_case`` over a list of cases.

    When ``is_dicom_manifest`` is True the ``case_dir`` field is a JSON
    dict of ``{modality: dicom_dir}``; otherwise it is a real directory.
    """
    report = PreprocessReport()
    for cid, case_input, label in cases:
        if is_dicom_manifest:
            mod_paths = json.loads(case_input)
            result = _process_dicom_case(cid, mod_paths, label, cfg)
        else:
            result = process_one_case(cid, case_input, label, cfg)
        if isinstance(result, CaseRecord):
            report.kept.append(result)
        else:
            report.dropped.append(result)
    return report


def _process_dicom_case(
    case_id: str,
    mod_paths: Dict[str, str],
    label: Optional[str],
    cfg: PreprocessConfig,
) -> CaseRecord | DropRecord:
    """Variant of ``process_one_case`` for DICOM-tree cohorts where the
    per-case modality folders are scattered on disk."""
    if case_id in INVALID_CASE_IDS:
        return DropRecord(case_id=case_id, reason="invalid_id_not_irradiated",
                          raw_input=json.dumps(mod_paths))

    if not mod_paths and cfg.drop_if_no_modalities:
        return DropRecord(case_id=case_id, reason="no_recognised_modalities",
                          raw_input=json.dumps(mod_paths))

    available = [m for m in CANONICAL_MODALITIES if m in mod_paths]
    missing = [m for m in CANONICAL_MODALITIES if m not in mod_paths]
    if missing and cfg.drop_if_missing_modality:
        return DropRecord(case_id=case_id, reason=f"missing_modalities:{','.join(missing)}",
                          raw_input=json.dumps(mod_paths))

    ref_mod = next((m for m in REFERENCE_PRIORITY if m in mod_paths), None)
    if ref_mod is None:
        return DropRecord(case_id=case_id, reason="no_reference_modality",
                          raw_input=json.dumps(mod_paths))

    ref_vol = load_modality(mod_paths[ref_mod])
    target_shape = tuple(ref_vol.data.shape)
    affine = ref_vol.affine

    channels: Dict[str, np.ndarray] = {ref_mod: ref_vol.data.astype(np.float32, copy=False)}
    for m in available:
        if m == ref_mod:
            continue
        vol = load_modality(mod_paths[m])
        channels[m] = _resample_to_grid(vol, target_shape, order=1)

    fg_masks = [c > cfg.fg_threshold for c in channels.values()]
    bbox = _bbox_union(fg_masks)

    image = np.zeros((len(CANONICAL_MODALITIES),) + tuple(s.stop - s.start for s in bbox),
                    dtype=np.float32)
    for i, m in enumerate(CANONICAL_MODALITIES):
        if m in channels:
            crop = channels[m][bbox]
            if cfg.zscore_foreground:
                crop = _zscore_with_foreground_stats(crop)
            image[i] = crop

    # DICOM dump has no per-voxel segmentation in this layout
    label_arr = np.zeros((3,) + image.shape[1:], dtype=np.uint8)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}.npz"
    np.savez_compressed(out_path, image=image, label=label_arr, affine=affine.astype(np.float64))

    return CaseRecord(
        case_id=case_id,
        out_path=str(out_path),
        label=label,
        available_modalities=available,
        missing_modalities=missing,
        shape=tuple(image.shape[1:]),
        reference_modality=ref_mod,
        has_seg=False,
        raw_input=json.dumps(mod_paths),
    )


def write_report(report: PreprocessReport, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preprocess_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    # Also a flat manifest of kept cases, for downstream consumers
    manifest = [
        {
            "case_id": r.case_id,
            "npz": r.out_path,
            "label": r.label,
            "available_modalities": r.available_modalities,
            "missing_modalities": r.missing_modalities,
            "shape": list(r.shape),
            "has_seg": r.has_seg,
        }
        for r in report.kept
    ]
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
