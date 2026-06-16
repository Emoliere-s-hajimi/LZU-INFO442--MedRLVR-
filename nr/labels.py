"""Label CSV loader + template / auto-label emitters.

The CSV has three columns: ``patient_folder`` (exact directory name on
disk), ``label`` (one of ``recurrence``, ``necrosis``, ``necrosis+recurrence``)
and ``source`` (``exact`` / ``pinyin_unique`` / ``MANUAL`` — informational
only, ignored at load time).

``load_labels`` is strict about typos so the training set never silently
mislabels patients. The CLI offers two modes:

  - ``--emit-template`` writes a CSV with one row per training patient and
    an empty label column — for the case where you want to fill everything
    in by hand.

  - ``--auto`` writes the same CSV but pre-populates labels by matching
    each Chinese patient name against the cleaned-cohort filename indexes
    (Brain_Cnts_all + 事实单病灶). Rows the matcher is unsure about get
    a blank label and ``source=MANUAL`` so you can tab to them quickly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

from .discover import iter_train_patients


VALID_LABELS = {"recurrence", "necrosis", "necrosis+recurrence"}


def load_labels(csv_path: str) -> Dict[str, str]:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"labels CSV not found: {csv_path}")
    out: Dict[str, str] = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "patient_folder" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: expected header 'patient_folder,label[,source]', got {reader.fieldnames!r}"
            )
        for i, row in enumerate(reader, start=2):
            pf = (row.get("patient_folder") or "").strip()
            lab = (row.get("label") or "").strip().lower()
            if not pf:
                continue
            if not lab:
                # silently skip — these are patients the user hasn't labelled yet
                # (preprocess will drop them with reason="no_label_in_csv")
                continue
            if lab not in VALID_LABELS:
                raise ValueError(
                    f"{csv_path}:{i}: invalid label '{lab}' for '{pf}' "
                    f"(allowed: {sorted(VALID_LABELS)})"
                )
            if pf in out:
                raise ValueError(f"{csv_path}:{i}: duplicate patient_folder '{pf}'")
            out[pf] = lab
    return out


def emit_template(train_root: str, out_path: str) -> int:
    """Write a labels CSV stub with one row per patient and an empty label
    column. Returns the number of patient rows written."""
    rows = [pid for pid, _ in iter_train_patients(train_root)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_folder", "label", "source"])
        for pid in rows:
            w.writerow([pid, "", "MANUAL"])
    return len(rows)


def emit_auto(
    train_root: str,
    brain_cnts_dir: str,
    shishi_dir: str,
    out_path: str,
) -> Dict[str, int]:
    """Walk the training tree and try to auto-label each patient against
    the filename indexes. Returns a dict of source → count for printing.

    Requires ``pypinyin`` to be installed. The label index is built from
    Brain_Cnts_all and 事实单病灶 — both ship inside ``data1/``.
    """
    from .auto_label import auto_label

    rows, lookup = auto_label(train_root, brain_cnts_dir, shishi_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_folder", "label", "source"])
        for patient, label, source in rows:
            display_source = source if label is not None else "MANUAL"
            w.writerow([patient, label or "", display_source])
            counts[display_source] = counts.get(display_source, 0) + 1
    counts["__index_pairs__"] = lookup.n_pairs
    counts["__index_pinyin__"] = lookup.n_pinyin
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-template", action="store_true",
                      help="Write a blank CSV (one row per training patient).")
    mode.add_argument("--auto", action="store_true",
                      help="Pre-populate labels by matching against the cleaned-cohort filename indexes.")
    ap.add_argument("--train_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--brain_cnts_dir", type=str,
                    default="data1/NR数据/训练集复发与坏死数据再清洗_20211102/Brain_Cnts_all")
    ap.add_argument("--shishi_dir", type=str,
                    default="data1/数据集/SourceData/事实单病灶")
    args = ap.parse_args()

    if args.emit_template:
        n = emit_template(args.train_root, args.out)
        print(f"wrote {n} patients to {args.out}")
        return

    if args.auto:
        counts = emit_auto(args.train_root, args.brain_cnts_dir, args.shishi_dir, args.out)
        index_pairs = counts.pop("__index_pairs__", 0)
        index_pinyin = counts.pop("__index_pinyin__", 0)
        total = sum(counts.values())
        print(f"label index: {index_pairs} (pinyin,date) pairs over {index_pinyin} unique pinyin names")
        print(f"auto-label results over {total} training patients:")
        for src in ("exact", "pinyin_unique", "MANUAL"):
            n = counts.get(src, 0)
            print(f"  {src:14s} : {n:4d}  ({n/total*100:.1f}%)")
        manual = counts.get("MANUAL", 0)
        if manual:
            print(f"\nNext step: open {args.out} and fill in the {manual} rows whose "
                  f"source is MANUAL.")


if __name__ == "__main__":
    main()
