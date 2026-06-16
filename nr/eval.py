"""One-shot evaluation against the validation manifest.

Loads a checkpoint, runs the validation loop once, and writes
``predictions.csv`` (one row per case) plus ``metrics.json``.

Usage::

    python -m nr.eval \\
        --config nr_subproject/configs/nr.yaml \\
        --checkpoint nr_subproject/outputs/run1/best_loss.pt \\
        --val_manifest nr_subproject/processed/val_manifest.json \\
        --out_dir nr_subproject/outputs/run1/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.data.dataset import MultiModalMRIDataset, _label_to_int
from src.metrics import classification_metrics
from src.models import build_model
from src.utils import get_logger, load_checkpoint, load_config


def _load_manifest(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="nr_subproject/configs/nr.yaml")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--val_manifest", type=str,
                    default="nr_subproject/processed/val_manifest.json")
    ap.add_argument("--out_dir", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or (Path(cfg["project"]["output_dir"]) / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("nr-eval")

    manifest = _load_manifest(args.val_manifest)
    if not manifest:
        raise SystemExit(f"empty manifest: {args.val_manifest}")

    ds = MultiModalMRIDataset(
        manifest=manifest,
        patch_size=tuple(cfg["data"]["patch_size"]),
        augment=False,
        modality_dropout_p=0.0,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg["train"].get("num_workers", 2),
                        pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = load_checkpoint(args.checkpoint, model, map_location=str(device))
    logger.info("loaded checkpoint %s @ epoch %s", args.checkpoint, state.get("epoch"))
    model.eval()

    rows: List[Dict] = []
    all_probs: List[float] = []
    all_labels: List[int] = []
    dice_acc = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    n_dice = 0

    use_amp = cfg["train"].get("amp", False) and torch.cuda.is_available()
    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        missing = batch["missing_mask"].to(device)
        aux = batch["aux"].to(device)
        label = int(batch["label"].item())
        case_id = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else batch["case_id"]

        with torch.no_grad(), autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=use_amp,
        ):
            out = model(image, missing_mask=missing, aux_features=aux, return_aux=False)
        probs = torch.softmax(out["cls"].float(), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())

        seg_pred = (torch.sigmoid(out["seg"].float()) >= 0.5).cpu().numpy()[0]
        seg_target = batch["seg"][0].numpy()
        dice_row: Dict[str, float] = {}
        if seg_target.any():
            for c, name in enumerate(("WT", "TC", "ET")):
                p = seg_pred[c].astype(np.float32)
                t = seg_target[c].astype(np.float32)
                inter = (p * t).sum()
                union = p.sum() + t.sum()
                d = float((2.0 * inter + 1e-6) / (union + 1e-6))
                dice_acc[name] += d
                dice_row[f"dice_{name}"] = d
            n_dice += 1

        all_probs.append(float(probs[1]))
        all_labels.append(label)
        rows.append({
            "case_id": case_id,
            "label": label,
            "prob_recurrence": float(probs[1]),
            "pred": pred,
            **dice_row,
        })

    cls_metrics = classification_metrics(
        probs=np.array(all_probs), labels=np.array(all_labels), threshold=0.5,
    )
    dice_mean = {k: (v / n_dice if n_dice else float("nan")) for k, v in dice_acc.items()}
    dice_mean["mean"] = (
        float(np.mean(list(dice_mean.values()))) if n_dice and all(math.isfinite(v) for v in dice_mean.values())
        else float("nan")
    )

    preds_path = out_dir / "predictions.csv"
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label", "prob_recurrence", "pred",
                                          "dice_WT", "dice_TC", "dice_ET"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"classification": cls_metrics, "dice": dice_mean,
                  "n_cases": len(rows), "n_with_seg": n_dice}, f, indent=2)

    logger.info("wrote %d predictions to %s", len(rows), preds_path)
    logger.info("metrics: %s", cls_metrics)


if __name__ == "__main__":
    main()
