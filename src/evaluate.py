"""Evaluation entry point. Computes the metrics declared in the config and
optionally dumps the prediction visualisations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from .data import build_dataloaders
from .metrics import classification_metrics, confusion_matrix_dict, dice_score
from .models import build_model
from .utils import get_logger, load_checkpoint, load_config, set_seed
from .visualization.predictions import plot_calibration, plot_confusion_matrix, plot_roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--save_plots", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("eval")
    set_seed(cfg["project"]["seed"])

    loaders = build_dataloaders(
        manifest_path=args.manifest,
        modalities=cfg["data"]["modalities"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        batch_size=1,
        num_workers=cfg["train"]["num_workers"],
        split=cfg["data"]["train_val_test_split"],
        seed=cfg["project"]["seed"],
        use_weighted_sampler=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    cls_probs: List[np.ndarray] = []
    cls_labels: List[int] = []
    dice_records: List[float] = []
    case_ids: List[str] = []

    for batch in loaders[args.split]:
        image = batch["image"].to(device)
        out = model(image, return_aux=False)
        cls_probs.append(out["cls"].softmax(dim=1).cpu().numpy()[0])
        cls_labels.append(int(batch["label"].item()))
        case_ids.append(batch["case_id"][0])
        if "seg" in batch:
            pred = out["seg"].argmax(dim=1).cpu().numpy()[0]
            scores = dice_score(pred, batch["seg"].numpy()[0])
            dice_records.append(scores["dice_mean"])

    probs = np.stack(cls_probs)
    labels = np.array(cls_labels)
    metrics = classification_metrics(probs, labels)
    if dice_records:
        metrics["dice_mean"] = float(np.mean(dice_records))
    metrics["confusion_matrix"] = confusion_matrix_dict(probs, labels)

    metrics_path = out_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics written to %s", metrics_path)
    logger.info(json.dumps(metrics, indent=2))

    preds_path = out_dir / f"predictions_{args.split}.csv"
    with open(preds_path, "w") as f:
        f.write("case_id,label,prob_recurrence,prediction\n")
        for cid, lbl, p in zip(case_ids, labels, probs):
            pred = int(p[1] >= 0.5)
            f.write(f"{cid},{int(lbl)},{p[1]:.6f},{pred}\n")

    if args.save_plots:
        plots_dir = out_dir / f"plots_{args.split}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_roc_curve(probs, labels, out_path=str(plots_dir / "roc.png"))
        plot_confusion_matrix(metrics["confusion_matrix"], out_path=str(plots_dir / "confusion.png"))
        plot_calibration(probs, labels, out_path=str(plots_dir / "calibration.png"))
        logger.info("plots written to %s", plots_dir)


if __name__ == "__main__":
    evaluate()
