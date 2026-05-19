"""Single-case inference. Given a case directory containing the four modality
NIfTI volumes (and optionally a seg mask), produces the recurrence probability,
the predicted segmentation, and an optional overlay PNG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .data.cleaning import _detect_modality
from .data.preprocessing import center_on_lesion, crop_or_pad, normalize_zscore
from .models import build_model
from .utils import get_logger, load_checkpoint, load_config


LABEL_NAMES = {0: "necrosis", 1: "recurrence"}


def _load_volume(path: str) -> np.ndarray:
    import nibabel as nib

    return nib.load(path).get_fdata().astype(np.float32)


def _gather_modality_paths(case_dir: Path, modalities) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for f in case_dir.iterdir():
        if not f.is_file():
            continue
        m = _detect_modality(f.name)
        if m is not None and m in modalities and m not in found:
            found[m] = str(f)
    missing = [m for m in modalities if m not in found]
    if missing:
        raise FileNotFoundError(f"missing modalities {missing} in {case_dir}")
    return found


def _save_nifti(volume: np.ndarray, reference_path: str, out_path: str) -> None:
    import nibabel as nib

    ref = nib.load(reference_path)
    img = nib.Nifti1Image(volume.astype(np.int16), affine=ref.affine, header=ref.header)
    nib.save(img, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case_dir", required=True, help="Directory holding the four modality volumes")
    parser.add_argument("--out_dir", default="outputs/inference")
    parser.add_argument("--save_overlay", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def run_inference(
    config_path: str,
    checkpoint_path: str,
    case_dir: str,
    out_dir: str,
    save_overlay: bool = False,
) -> Dict:
    cfg = load_config(config_path)
    logger = get_logger("inference")

    modalities = list(cfg["data"]["modalities"])
    patch = tuple(cfg["data"]["patch_size"])
    case_path = Path(case_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = _gather_modality_paths(case_path, modalities)
    volumes = {m: normalize_zscore(_load_volume(paths[m])) for m in modalities}
    ref_modality = "t1ce" if "t1ce" in volumes else modalities[0]
    ref_shape = volumes[ref_modality].shape

    seg_path: Optional[str] = None
    for f in case_path.iterdir():
        if f.is_file() and ("seg" in f.name.lower() or "mask" in f.name.lower()):
            seg_path = str(f)
            break

    seg_vol = _load_volume(seg_path).astype(np.int64) if seg_path else None
    center = center_on_lesion(seg_vol, fallback=tuple(s // 2 for s in ref_shape))

    starts = [max(0, c - p // 2) for c, p in zip(center, patch)]
    slc = tuple(slice(s, s + p) for s, p in zip(starts, patch))
    cropped = {m: crop_or_pad(volumes[m][slc], patch) for m in modalities}

    image = np.stack([cropped[m] for m in modalities], axis=0)[None]
    tensor = torch.from_numpy(image).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, map_location=str(device))
    model.eval()
    tensor = tensor.to(device)
    out = model(tensor, return_aux=False)
    probs = out["cls"].softmax(dim=1).cpu().numpy()[0]
    seg_pred = out["seg"].argmax(dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    result = {
        "case_id": case_path.name,
        "prob_necrosis": float(probs[0]),
        "prob_recurrence": float(probs[1]),
        "prediction": LABEL_NAMES[pred_class],
    }

    seg_out = out_path / f"{case_path.name}_pred_seg.nii.gz"
    _save_nifti(seg_pred, paths[ref_modality], str(seg_out))
    result["segmentation_path"] = str(seg_out)

    if save_overlay:
        from .visualization.predictions import gt_vs_pred_overlay

        z = patch[2] // 2
        overlay = gt_vs_pred_overlay(
            cropped[ref_modality][:, :, z],
            (seg_vol[slc][:, :, z] if seg_vol is not None else np.zeros_like(seg_pred[:, :, z])),
            seg_pred[:, :, z],
            out_path=str(out_path / f"{case_path.name}_overlay.png"),
        )
        del overlay
        result["overlay_path"] = str(out_path / f"{case_path.name}_overlay.png")

    with open(out_path / f"{case_path.name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info("inference complete: %s", json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        case_dir=args.case_dir,
        out_dir=args.out_dir,
        save_overlay=args.save_overlay,
    )


if __name__ == "__main__":
    main()
