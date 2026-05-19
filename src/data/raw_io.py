"""Format-agnostic readers for the raw cohort.

The Tiantan post-radiation cohort arrives in two physical forms:

  1. NIfTI dumps with BraTS-style filenames (one ``.nii.gz`` per modality,
     plus an optional ``_seg.nii.gz``).
  2. Per-patient DICOM trees where modalities live in sibling folders
     (``4个常规结构像/{FLAIR,T1,T1ce,T2}/{复发,放坏,放坏+复发}/<id>/*.dcm``)
     and the slice axis is encoded in the InstanceNumber / SliceLocation
     of each DICOM file.

This module turns both forms into a uniform in-memory representation:

    ``ModalityVolume(data: np.ndarray (Z, Y, X) | (H, W, D),
                    affine: np.ndarray (4, 4),
                    spacing: tuple[float, float, float],
                    source: str,
                    series_description: str | None)``

Downstream pipeline stages only know about ``ModalityVolume``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ModalityVolume:
    data: np.ndarray
    affine: np.ndarray
    spacing: Tuple[float, float, float]
    source: str
    series_description: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------

def load_nifti(path: str) -> ModalityVolume:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    aff = np.asarray(img.affine, dtype=np.float64)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return ModalityVolume(data=data, affine=aff, spacing=spacing, source=str(path))


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------

def _read_dicom_safely(path: Path):
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=False, force=True)


def assemble_dicom_series(dicom_dir: str) -> ModalityVolume:
    """Sort a directory of DICOM slices and stack them into a 3-D volume.

    Sort key precedence:
      ``ImagePositionPatient[2]`` (true z) → ``SliceLocation`` →
      ``InstanceNumber`` → filename. Ties are broken by InstanceNumber.

    Rescale slope/intercept are applied. The returned ``affine`` is a
    minimal LPS→RAS-ish transform built from ImageOrientationPatient and
    ImagePositionPatient if available, else from PixelSpacing/SliceThickness.
    """
    import pydicom

    p = Path(dicom_dir)
    files = sorted(p.glob("*.dcm"))
    if not files:
        files = sorted(p.glob("*"))
    if not files:
        raise FileNotFoundError(f"no DICOM files under {dicom_dir}")

    slices = []
    for f in files:
        try:
            ds = _read_dicom_safely(f)
            if not hasattr(ds, "PixelData"):
                continue
            slices.append((f, ds))
        except Exception:
            continue
    if not slices:
        raise RuntimeError(f"no readable DICOM slices under {dicom_dir}")

    def _key(item):
        _f, ds = item
        ipp = getattr(ds, "ImagePositionPatient", None)
        sl = getattr(ds, "SliceLocation", None)
        inst = getattr(ds, "InstanceNumber", None)
        primary = float(ipp[2]) if ipp is not None else (float(sl) if sl is not None else None)
        secondary = float(inst) if inst is not None else 0.0
        if primary is None:
            primary = secondary
        return (primary, secondary)

    slices.sort(key=_key)

    ref = slices[0][1]
    rows, cols = int(ref.Rows), int(ref.Columns)
    arr = np.zeros((len(slices), rows, cols), dtype=np.float32)
    for i, (_f, ds) in enumerate(slices):
        try:
            px = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if px.shape != (rows, cols):
            # Pad / crop mismatched slice to ref grid
            tmp = np.zeros((rows, cols), dtype=np.float32)
            r0 = min(rows, px.shape[0]); c0 = min(cols, px.shape[1])
            tmp[:r0, :c0] = px[:r0, :c0]
            px = tmp
        arr[i] = slope * px + intercept

    # Spacing (mm): in-plane PixelSpacing + through-plane via ipp delta or SliceThickness
    ps = getattr(ref, "PixelSpacing", [1.0, 1.0])
    in_plane = (float(ps[0]), float(ps[1]))
    if len(slices) >= 2:
        ipp0 = getattr(slices[0][1], "ImagePositionPatient", None)
        ipp1 = getattr(slices[1][1], "ImagePositionPatient", None)
        if ipp0 is not None and ipp1 is not None:
            dz = float(np.linalg.norm(np.array(ipp1, dtype=float) - np.array(ipp0, dtype=float)))
        else:
            dz = float(getattr(ref, "SliceThickness", 1.0) or 1.0)
    else:
        dz = float(getattr(ref, "SliceThickness", 1.0) or 1.0)
    spacing = (dz, in_plane[0], in_plane[1])  # axis order matches arr (Z, Y, X)

    affine = _affine_from_dicom(ref, slices, dz)
    series_desc = str(getattr(ref, "SeriesDescription", "")) or None

    return ModalityVolume(
        data=arr,
        affine=affine,
        spacing=spacing,
        source=str(p),
        series_description=series_desc,
    )


def _affine_from_dicom(ref, slices: Sequence, dz: float) -> np.ndarray:
    """Build a 4×4 affine from DICOM ImageOrientationPatient + ImagePositionPatient.

    Falls back to a diagonal spacing-only affine when the geometry tags are absent.
    """
    iop = getattr(ref, "ImageOrientationPatient", None)
    ipp = getattr(ref, "ImagePositionPatient", None)
    ps = getattr(ref, "PixelSpacing", [1.0, 1.0])
    if iop is not None and ipp is not None and len(iop) == 6:
        row_cos = np.array(iop[:3], dtype=float)
        col_cos = np.array(iop[3:], dtype=float)
        if len(slices) >= 2:
            ipp0 = np.array(getattr(slices[0][1], "ImagePositionPatient", ipp), dtype=float)
            ipp1 = np.array(getattr(slices[1][1], "ImagePositionPatient", ipp), dtype=float)
            slice_cos = ipp1 - ipp0
            n = np.linalg.norm(slice_cos)
            if n > 1e-6:
                slice_cos = slice_cos / n * dz
            else:
                slice_cos = np.cross(row_cos, col_cos) * dz
        else:
            slice_cos = np.cross(row_cos, col_cos) * dz
        aff = np.eye(4)
        aff[:3, 0] = row_cos * float(ps[1])
        aff[:3, 1] = col_cos * float(ps[0])
        aff[:3, 2] = slice_cos
        aff[:3, 3] = np.array(ipp, dtype=float)
        return aff
    return np.diag([float(ps[0]), float(ps[1]), float(dz), 1.0])


# ---------------------------------------------------------------------------
# Modality assembly for a case
# ---------------------------------------------------------------------------

def find_modalities_in_case_dir(case_dir: str) -> Dict[str, str]:
    """Return ``{modality: file_or_dir_path}`` discovered under ``case_dir``.

    Recognizes both ``BraTS2021_<id>_<mod>.nii.gz`` files and per-modality
    subfolders of DICOM slices.
    """
    from .modality_map import classify_filename

    case_dir = Path(case_dir)
    found: Dict[str, str] = {}

    for f in case_dir.iterdir():
        if not f.is_file():
            continue
        if not (f.name.endswith(".nii") or f.name.endswith(".nii.gz")):
            continue
        mod = classify_filename(f.name)
        if mod is not None:
            found[mod] = str(f)

    return found


def discover_segmentation(case_dir: str) -> Optional[str]:
    case_dir = Path(case_dir)
    for f in case_dir.iterdir():
        n = f.name.lower()
        if not f.is_file():
            continue
        if not (n.endswith(".nii") or n.endswith(".nii.gz")):
            continue
        if "seg" in n or "mask" in n or "label" in n:
            return str(f)
    return None


def load_modality(path_or_dir: str) -> ModalityVolume:
    p = Path(path_or_dir)
    if p.is_dir():
        return assemble_dicom_series(str(p))
    return load_nifti(str(p))
