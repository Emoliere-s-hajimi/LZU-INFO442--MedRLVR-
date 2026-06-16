"""Walk the NR data trees and yield patient directories.

Two layouts are supported:

  - ``iter_train_patients`` / ``iter_val_patients`` — the original DICOM
    per-patient trees. Retained for completeness but no longer used by
    the active pipeline.
  - ``iter_seg_patients`` — the cleaned ``SourcePreprocess_SegLabel_202110``
    tree (``{N,R,RN}/<patient>/<class>_<id>_<modality>.nii.gz``). This is
    the source of truth for training as of the 2026 refit; labels come
    from the class folder name and segmentations ship in-place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple


_TRAIN_JUNK = {"xin 录入"}


# Class folder name → canonical label in ``LABEL_MAP``
SEG_TREE_LABELS: Dict[str, str] = {
    "N":  "necrosis",
    "R":  "recurrence",
    "RN": "necrosis+recurrence",
}


def iter_train_patients(train_root: str) -> Iterator[Tuple[str, Path]]:
    """Yield ``(patient_folder_name, abs_path)`` for the per-patient train set.

    The expected layout is ``<train_root>/{1部分,2部分}/{1部分,2部分}/<patient>/``.
    Junk folders (entries in ``_TRAIN_JUNK``) and any non-directory entries are skipped.
    """
    root = Path(train_root)
    for part in sorted(root.iterdir()):
        if not part.is_dir():
            continue
        inner = part / part.name
        scan_dir = inner if inner.is_dir() else part
        for patient in sorted(scan_dir.iterdir()):
            if not patient.is_dir():
                continue
            if patient.name in _TRAIN_JUNK:
                continue
            yield patient.name, patient.resolve()


def iter_val_patients(val_root: str) -> Iterator[Tuple[str, Path]]:
    """Yield ``(case_id, abs_path)`` for the validation set.

    Layout: ``<val_root>/20220122坏死数据/<patient>/``. ``case_id`` is the
    folder name with leading ``N`` / ``N1`` necrosis markers and trailing
    "没有FLAIR" / "（没有FLAIR）" annotations stripped, so it stays a clean
    filesystem-safe identifier when used as the ``.npz`` filename.
    """
    root = Path(val_root)
    inner = root / "20220122坏死数据"
    scan_dir = inner if inner.is_dir() else root
    for patient in sorted(scan_dir.iterdir()):
        if not patient.is_dir():
            continue
        yield _val_case_id(patient.name), patient.resolve()


def iter_seg_patients(
    seg_root: str,
    prefer_revised: bool = True,
) -> Iterator[Tuple[str, Path, str]]:
    """Walk ``SourcePreprocess_SegLabel_202110`` and yield
    ``(case_id, abs_dir, label)`` per patient folder.

    Layout::

        <seg_root>/
            N/<id>/                       (52 patients, plain N)
            N_坏死_修改版/N/<id>/          (revised N — same IDs as above)
            R/<id>/                       (199 patients)
            RN/<id>/                      (71 patients, often missing
                                           one or more structural modalities)

    ``case_id`` is ``"<class>_<folder_name>"`` (e.g. ``N_005``, ``R_148``,
    ``RN_085_2``). When ``prefer_revised=True`` (the default), the revised
    ``N_坏死_修改版/N/`` folders shadow the plain ``N/`` IDs. The folder is
    also accepted under its original ``N（坏死）(修改版）`` name for
    backwards compatibility.

    Folders without any ``.nii.gz`` are skipped silently — leaving the
    drop reason to the preprocess step keeps discovery side-effect free.
    """
    root = Path(seg_root)
    if not root.exists():
        raise FileNotFoundError(f"seg_root not found: {seg_root}")

    revised_dir = next(
        (root / name / "N" for name in ("N_坏死_修改版", "N（坏死）(修改版）")
         if (root / name / "N").is_dir()),
        None,
    )
    revised_ids: set[str] = set()
    if prefer_revised and revised_dir is not None:
        for sub in sorted(revised_dir.iterdir()):
            if sub.is_dir() and _has_nifti(sub):
                revised_ids.add(sub.name)
                yield f"N_{sub.name}", sub.resolve(), "necrosis"

    for cls in ("N", "R", "RN"):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        label = SEG_TREE_LABELS[cls]
        for sub in sorted(cls_dir.iterdir()):
            if not sub.is_dir() or not _has_nifti(sub):
                continue
            if cls == "N" and sub.name in revised_ids:
                continue  # superseded by the revised version above
            yield f"{cls}_{sub.name}", sub.resolve(), label


def _has_nifti(d: Path) -> bool:
    for f in d.iterdir():
        if f.is_file() and (f.name.endswith(".nii") or f.name.endswith(".nii.gz")):
            return True
    return False


def _val_case_id(folder_name: str) -> str:
    name = folder_name
    for tag in ("（没有FLAIR）", "没有FLAIR"):
        name = name.replace(tag, "")
    # strip leading necrosis prefix ("N", "N1", ...) only when followed by a Chinese / digit char
    if name and name[0] in ("N", "n"):
        rest = name[1:]
        # leading digit after N is e.g. N1 王力 — strip the digit too
        while rest and rest[0].isdigit():
            rest = rest[1:]
        if rest:
            name = rest
    return name.strip() or folder_name
