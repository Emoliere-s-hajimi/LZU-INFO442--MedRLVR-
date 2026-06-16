"""Auto-derive training labels from the cleaned cohort indexes.

Two on-disk sources carry per-patient labels in their *filenames*:

  - ``data1/NR数据/训练集复发与坏死数据再清洗_20211102/Brain_Cnts_all/<L>_<PINYIN>_<id>_<date_8>_<modality>_<slice>.jpg``
    where ``L`` is one of ``{N, R, RN}``.
  - ``data1/数据集/SourceData/事实单病灶/<PINYIN>_<id>_<date_8>_<L>.jpg``
    where ``L`` is one of ``{N, R}``.

For each training patient ``<Chinese>YYYYMMDD<short_id>`` we transliterate
the Chinese name to pinyin and look it up in the combined index. A match
on ``(pinyin, date)`` is exact; a fallback on pinyin alone is accepted
only when that pinyin maps to a single label across both sources (no
class ambiguity). Everything else is reported as ``MANUAL`` so the user
can fill it in.

Empirically this auto-labels ≈ 83 % of the 442 training patients on the
Tiantan NR cohort, with 0 ambiguous-pinyin cases.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Filename-prefix → canonical label
_PREFIX_LABEL = {
    "N": "necrosis",
    "R": "recurrence",
    "RN": "necrosis+recurrence",
}

# Raw folder pattern: <chinese_name><date_8><tail_digits>
_RAW_RE = re.compile(r"^([^\d]+)(\d{8})\d+$")


@dataclass
class LabelLookup:
    by_pinyin_date: Dict[Tuple[str, str], str]
    by_pinyin: Dict[str, set]           # pinyin → {label, ...}

    @property
    def n_pairs(self) -> int:
        return len(self.by_pinyin_date)

    @property
    def n_pinyin(self) -> int:
        return len(self.by_pinyin)


def _pinyin_of(chinese: str) -> str:
    from pypinyin import Style, lazy_pinyin
    return "-".join(lazy_pinyin(chinese, style=Style.NORMAL)).upper()


def _parse_raw_folder(name: str) -> Tuple[Optional[str], Optional[str]]:
    m = _RAW_RE.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _add_entry(lookup: LabelLookup, pinyin: str, date: str, label: str) -> None:
    lookup.by_pinyin_date[(pinyin, date)] = label
    lookup.by_pinyin.setdefault(pinyin, set()).add(label)


def build_label_lookup(brain_cnts_dir: Optional[Path], shishi_dir: Optional[Path]) -> LabelLookup:
    """Aggregate (pinyin, date, label) triples from both filename sources."""
    lookup = LabelLookup(by_pinyin_date={}, by_pinyin=defaultdict(set))

    if brain_cnts_dir and brain_cnts_dir.is_dir():
        for f in brain_cnts_dir.iterdir():
            if not f.name.endswith(".jpg"):
                continue
            parts = f.stem.split("_")
            if len(parts) < 4:
                continue
            prefix, pinyin, _pid, date = parts[0], parts[1], parts[2], parts[3]
            label = _PREFIX_LABEL.get(prefix)
            if label and len(date) == 8 and date.isdigit():
                _add_entry(lookup, pinyin, date, label)

    if shishi_dir and shishi_dir.is_dir():
        for f in shishi_dir.iterdir():
            if not f.name.endswith(".jpg"):
                continue
            parts = f.stem.split("_")
            if len(parts) < 4:
                continue
            pinyin, _pid, date, suffix = parts[0], parts[1], parts[2], parts[3]
            label = _PREFIX_LABEL.get(suffix.upper())
            if label and len(date) == 8 and date.isdigit():
                _add_entry(lookup, pinyin, date, label)

    return lookup


def label_for_patient(folder_name: str, lookup: LabelLookup) -> Tuple[Optional[str], str]:
    """Return ``(label, source)``.

    ``source`` is one of:
      - ``exact``         : (pinyin, date) hit
      - ``pinyin_unique`` : pinyin hit with a single label across the index
      - ``pinyin_ambig``  : pinyin hit but multiple labels — label is None
      - ``no_pinyin``     : pinyin not in index — label is None
      - ``unparseable``   : folder name didn't match ``<Chinese><date_8><id>``
    """
    cn, date = _parse_raw_folder(folder_name)
    if cn is None:
        return None, "unparseable"

    pinyin = _pinyin_of(cn)
    if (pinyin, date) in lookup.by_pinyin_date:
        return lookup.by_pinyin_date[(pinyin, date)], "exact"

    labs = lookup.by_pinyin.get(pinyin)
    if not labs:
        return None, "no_pinyin"
    if len(labs) == 1:
        return next(iter(labs)), "pinyin_unique"
    return None, "pinyin_ambig"


def auto_label(
    train_root: str,
    brain_cnts_dir: str,
    shishi_dir: Optional[str],
) -> Tuple[List[Tuple[str, Optional[str], str]], LabelLookup]:
    """Return a list of ``(patient_folder, label_or_None, source)`` rows."""
    from .discover import iter_train_patients

    lookup = build_label_lookup(
        Path(brain_cnts_dir) if brain_cnts_dir else None,
        Path(shishi_dir) if shishi_dir else None,
    )

    rows: List[Tuple[str, Optional[str], str]] = []
    for patient, _ in iter_train_patients(train_root):
        label, source = label_for_patient(patient, lookup)
        rows.append((patient, label, source))
    return rows, lookup
