"""Mapping from heterogeneous DICOM SeriesDescription strings and folder
labels to the canonical BraTS-style modality set: t1, t1ce, t2, flair.

The Tiantan cohort ships with scanner-specific SeriesDescription strings
(Siemens/GE/Philips/UIH all represented; see
``data/数据集/SourceData/SeriesDescription_Total.xlsx`` for the full vocabulary).
This module is the single source of truth for collapsing that vocabulary
into the four channels our model consumes.
"""
from __future__ import annotations

import re
from typing import Optional

CANONICAL_MODALITIES = ("t1", "t1ce", "t2", "flair")

# Class labels (Chinese names used in the raw cohort dump)
LABEL_RECURRENCE = "recurrence"
LABEL_NECROSIS = "necrosis"
LABEL_BOTH = "necrosis+recurrence"

_CHINESE_LABEL_MAP = {
    "复发": LABEL_RECURRENCE,
    "放坏": LABEL_NECROSIS,
    "放射性坏死": LABEL_NECROSIS,
    "放坏+复发": LABEL_BOTH,
    "复发+放坏": LABEL_BOTH,
}


def parse_chinese_label(token: str) -> Optional[str]:
    if token is None:
        return None
    return _CHINESE_LABEL_MAP.get(token.strip())


def classify_series_description(desc: str) -> Optional[str]:
    """Return one of {t1, t1ce, t2, flair} or None.

    Heuristics, in order:
      1. anything with a contrast marker (``+C``, ``C+``, ``GD``, ``post``,
         ``CE``, ``+c``) plus a T1 token  → t1ce
      2. anything with ``FLAIR`` or ``dark[-_]fluid`` / ``dark fluid``  → flair
      3. plain T1 / MPRAGE / TIRM  → t1
      4. plain T2 / TSE / FSE / PROPELLER  → t2
    Order matters: T1 FLAIR (post-contrast or not) must not be mis-routed
    to T1 or T1ce, and FLAIR contrast variants are still treated as FLAIR.
    """
    if desc is None:
        return None
    s = desc.strip().lower()
    if not s:
        return None

    has_contrast = bool(re.search(r"(\+c\b|c\+|gd|post[\s_-]?gd|\bce\b|contrast)", s))
    is_flair = bool(re.search(r"(flair|dark[\s_-]?fluid|tirm)", s))
    is_t1 = bool(re.search(r"(\bt1\b|mprage|fspgr|bravo|3dt1|v3dt1)", s))
    is_t2 = bool(
        re.search(r"(\bt2\b|propeller|prop\b|tse|fse|blade|fl/c)", s)
        and not is_flair
    )

    if is_flair:
        return "flair"
    if is_t1 and has_contrast:
        return "t1ce"
    if is_t1:
        return "t1"
    if is_t2:
        return "t2"
    return None


def classify_filename(name: str) -> Optional[str]:
    """Same idea as ``classify_series_description`` but tuned for filenames
    like ``BraTS2021_00171_t1ce.nii.gz``."""
    if name is None:
        return None
    s = name.lower()
    if re.search(r"(^|[_\-\.])(t1ce|t1c|t1gd|t1\+c|post[_\-]?t1)([_\-\.]|$)", s):
        return "t1ce"
    if re.search(r"(^|[_\-\.])flair([_\-\.]|$)", s):
        return "flair"
    if re.search(r"(^|[_\-\.])t1[w]?([_\-\.]|$)", s):
        return "t1"
    if re.search(r"(^|[_\-\.])t2[w]?([_\-\.]|$)", s):
        return "t2"
    return None
