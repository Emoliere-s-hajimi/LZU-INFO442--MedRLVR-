"""Map each ``<patient>/<series_number>/`` subfolder to one of
``{t1, t1ce, t2, flair}``.

Two sources of information are available, picked in this order:

1.  A parenthesized modality tag in the folder name itself, e.g.
    ``8（T1ce）`` or ``5（FLAIR）``. The validation set uses these heavily;
    the training set never does.
2.  The ``SeriesDescription`` DICOM header on the first slice of the
    series, parsed through the existing
    :func:`src.data.modality_map.classify_series_description` regex.

If two series resolve to the same canonical modality we keep the one with
the most slices (a standard tie-break for clinical MR — the longer
series is almost always the better-quality acquisition, and short series
are typically localisers, screen-saves, or DWI/PWI derived maps).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.data.modality_map import CANONICAL_MODALITIES, classify_series_description


# Group 1 captures the modality token inside half- or full-width parentheses
_PAREN_RE = re.compile(r"[（(]\s*([A-Za-z0-9+]+)\s*[)）]")


def _tag_from_folder(name: str) -> Optional[str]:
    m = _PAREN_RE.search(name)
    if not m:
        return None
    tok = m.group(1).strip().lower()
    # Normalise local variants
    if tok in ("t1ce", "t1c", "t1gd", "t1+c"):
        return "t1ce"
    if tok == "flair":
        return "flair"
    if tok == "t1":
        return "t1"
    if tok == "t2":
        return "t2"
    return None


def _read_first_dicom(series_dir: Path):
    import pydicom

    for f in sorted(series_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".dcm", ".ima", ""):
            continue
        try:
            return pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
    return None


def _count_dcm(series_dir: Path) -> int:
    n = 0
    for f in series_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".dcm", ".ima", ""):
            n += 1
    return n


def resolve_modalities(patient_dir: str, source: str) -> Dict[str, str]:
    """Return ``{modality: series_dir}`` for one patient.

    ``source`` is ``"train"`` or ``"val"``. The validation set's
    parenthesized folder tags are trusted absolutely; everything else
    falls back to a DICOM header read.
    """
    if source not in ("train", "val"):
        raise ValueError(f"source must be 'train' or 'val', got {source!r}")

    patient_dir_p = Path(patient_dir)
    # candidates :: list of (modality, series_dir, n_slices)
    candidates: list[Tuple[str, Path, int]] = []

    for series in patient_dir_p.iterdir():
        if not series.is_dir():
            continue
        n_slices = _count_dcm(series)
        if n_slices < 5:                               # localizers / scout
            continue

        modality: Optional[str] = None
        if source == "val":
            modality = _tag_from_folder(series.name)

        if modality is None:
            ds = _read_first_dicom(series)
            if ds is None:
                continue
            if getattr(ds, "Modality", "MR") != "MR":
                continue
            desc = getattr(ds, "SeriesDescription", None)
            if desc is None:
                continue
            modality = classify_series_description(str(desc))

        if modality in CANONICAL_MODALITIES:
            candidates.append((modality, series, n_slices))

    # Keep the longest series per modality
    best: Dict[str, Tuple[Path, int]] = {}
    for mod, series, n in candidates:
        if mod not in best or n > best[mod][1]:
            best[mod] = (series, n)

    return {mod: str(path) for mod, (path, _) in best.items()}
