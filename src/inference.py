"""Single-case inference for BrainTTNet.

Two input modes:

  (1) cleaned ``.npz`` — pass ``--npz path/to/<case>.npz`` and the
      pipeline runs straight on the pre-cropped, pre-normalised volume.

  (2) raw case directory — pass ``--case_dir path/to/<case>`` containing
      the four-modality NIfTI files delivered by the Tiantan partners
      (filenames follow a BraTS-2021-compatible convention, e.g.
      ``BraTS2021_<id>_<mod>.nii.gz``); the script invokes the same
      cleaning helpers used by ``scripts/preprocess_cohort.py`` and then
      runs the model.

Outputs:
    <out_dir>/<case_id>_prediction.json   recurrence probability + Dice
    <out_dir>/<case_id>_seg.npy           uint8 (3, H, W, D) predicted mask
    <out_dir>/<case_id>_overlay.png       optional T1ce + prediction overlay

Usage::

    python -m src.inference --checkpoint outputs/best_auc.pt \\
                            --npz data/processed/171.npz \\
                            --out_dir outputs/inference
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .data.pipeline import PreprocessConfig, process_one_case
from .models import build_model
from .utils import get_logger, load_checkpoint, load_config


LABEL_NAMES = {0: "necrosis", 1: "recurrence"}
MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")


def _load_npz(npz_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(npz_path)
    image = data["image"].astype(np.float32, copy=False)
    label = data["label"].astype(np.float32, copy=False) if "label" in data.files else None
    return image, label


def _clean_case_dir(case_dir: str, tmp_dir: str) -> str:
    case_id = Path(case_dir).name
    if "_" in case_id:
        case_id = case_id.split("_")[-1]
    record = process_one_case(
        case_id=case_id, case_dir=case_dir, label=None,
        cfg=PreprocessConfig(out_dir=tmp_dir, drop_if_missing_modality=False),
    )
    if not hasattr(record, "out_path"):
        raise RuntimeError(f"cleaning rejected the case: {record}")
    return record.out_path


def _crop_to_patch(image: np.ndarray, label: Optional[np.ndarray],
                   patch_size: Tuple[int, int, int]
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    fallback = tuple(s // 2 for s in image.shape[1:])
    if label is not None and label[0].any():
        coords = np.argwhere(label[0] > 0)
        center = tuple(int(c) for c in coords.mean(axis=0))
    else:
        center = fallback
    ps = tuple(patch_size)
    starts = [max(0, c - p // 2) for c, p in zip(center, ps)]

    def crop_pad(volume: np.ndarray) -> np.ndarray:
        out = np.zeros((volume.shape[0],) + ps, dtype=volume.dtype)
        for c in range(volume.shape[0]):
            slc = tuple(slice(s, min(volume.shape[i + 1], s + p))
                        for i, (s, p) in enumerate(zip(starts, ps)))
            chunk = volume[c][slc]
            out[c][tuple(slice(0, chunk.shape[i]) for i in range(3))] = chunk
        return out

    return crop_pad(image), (crop_pad(label) if label is not None else None)


def _load_aux(csv_path: Optional[str], case_id: str) -> np.ndarray:
    if not csv_path or not Path(csv_path).exists():
        return np.zeros(4, dtype=np.float32)
    import csv
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["case_id"] == case_id:
                return np.array([
                    float(row.get("volume_mm3_WT") or 0.0) / 1e5,
                    float(row.get("sphericity_WT") or 0.0),
                    float(row.get("n_components_WT") or 0.0) / 5.0,
                    float(row.get("intensity_ratio_in_over_out_t1ce") or 0.0),
                ], dtype=np.float32)
    return np.zeros(4, dtype=np.float32)


def _save_overlay(image: np.ndarray, seg_pred: np.ndarray, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = image.shape[3] // 2
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    for i, m in enumerate(MODALITY_ORDER):
        axes[i].imshow(image[i, :, :, z].T, cmap="gray", origin="lower")
        axes[i].set_title(m); axes[i].axis("off")
    overlay = np.stack([
        seg_pred[0, :, :, z] * 0.5,
        seg_pred[1, :, :, z] * 0.5,
        seg_pred[2, :, :, z] * 0.7,
    ], axis=-1)
    axes[4].imshow(image[1, :, :, z].T, cmap="gray", origin="lower")
    axes[4].imshow(np.transpose(overlay, (1, 0, 2)), alpha=0.55, origin="lower")
    axes[4].set_title("prediction on T1ce"); axes[4].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--case_dir", default=None)
    parser.add_argument("--aux_csv", default="visualization/morphology/morphology_features.csv")
    parser.add_argument("--out_dir", default="outputs/inference")
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_ema", action="store_true",
                        help="Load the EMA shadow weights from the checkpoint if present.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not args.npz and not args.case_dir:
        raise SystemExit("provide --npz or --case_dir")

    cfg = load_config(args.config)
    logger = get_logger("inference")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_dir:
        npz_path = _clean_case_dir(args.case_dir, str(out_dir / "tmp_cleaned"))
    else:
        npz_path = args.npz
    case_id = Path(npz_path).stem

    image, label = _load_npz(npz_path)
    image_patch, label_patch = _crop_to_patch(image, label, tuple(cfg["data"]["patch_size"]))
    missing_mask = np.array([(image_patch[c] == 0).all() for c in range(4)], dtype=bool)
    aux = _load_aux(args.aux_csv, case_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = load_checkpoint(args.checkpoint, model, map_location=str(device))
    if args.use_ema and "ema" in state:
        model.load_state_dict(state["ema"], strict=False)
        logger.info("loaded EMA shadow weights from checkpoint")
    model.eval()

    image_t = torch.from_numpy(image_patch).float().unsqueeze(0).to(device)
    missing_t = torch.from_numpy(missing_mask).unsqueeze(0).to(device)
    aux_t = torch.from_numpy(aux).unsqueeze(0).to(device)

    out = model(image_t, missing_mask=missing_t, aux_features=aux_t, return_aux=False)
    cls_probs = torch.softmax(out["cls"], dim=1).cpu().numpy()[0]
    pred_cls = int(cls_probs[1] >= args.threshold)
    seg_pred = (torch.sigmoid(out["seg"]) >= args.threshold).float().cpu().numpy()[0]

    result: Dict = {
        "case_id": case_id,
        "prob_recurrence": float(cls_probs[1]),
        "prediction": LABEL_NAMES[pred_cls],
        "missing_modalities": [m for m, miss in zip(MODALITY_ORDER, missing_mask) if miss],
        "chi_pred": float(out["chi_pred"][0]) if out.get("chi_pred") is not None else None,
        "checkpoint": args.checkpoint,
    }
    if label_patch is not None:
        dice = {}
        for c, name in enumerate(("WT", "TC", "ET")):
            inter = float((seg_pred[c] * label_patch[c]).sum())
            union = float(seg_pred[c].sum() + label_patch[c].sum())
            dice[name] = (2 * inter + 1e-6) / (union + 1e-6)
        result["dice"] = dice

    json_path = out_dir / f"{case_id}_prediction.json"
    json_path.write_text(json.dumps(result, indent=2))
    np.save(out_dir / f"{case_id}_seg.npy", seg_pred.astype(np.uint8))
    logger.info("wrote %s", json_path)
    print(json.dumps(result, indent=2))

    if args.save_overlay:
        overlay_path = out_dir / f"{case_id}_overlay.png"
        _save_overlay(image_patch, seg_pred, str(overlay_path))
        logger.info("wrote %s", overlay_path)


if __name__ == "__main__":
    main()
