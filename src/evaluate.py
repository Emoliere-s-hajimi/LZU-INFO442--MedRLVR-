"""Evaluation entry point.

Loads a checkpoint, runs validation-style inference over the requested
split, and writes:

    <out_dir>/
      ├── metrics_<split>.json        # AUC, sensitivity, specificity, Dice
      ├── predictions_<split>.csv     # per-case prediction
      └── plots_<split>/              # ROC, confusion matrix, calibration

Usage::

    python -m src.evaluate --checkpoint outputs/best_auc.pt --split test --save_plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .data import build_dataloaders
from .metrics import classification_metrics, confusion_matrix_dict
from .models import build_model
from .utils import get_logger, load_checkpoint, load_config, set_seed
from .visualization.predictions import plot_calibration, plot_confusion_matrix, plot_roc_curve


@torch.no_grad()
def _dice_per_channel(seg_logits: torch.Tensor, seg_target: torch.Tensor,
                      threshold: float = 0.5) -> Dict[str, float]:
    pred = (torch.sigmoid(seg_logits) >= threshold).float()
    target = seg_target.float()
    names = ("WT", "TC", "ET")
    out = {}
    for c, name in enumerate(names):
        p = pred[:, c]; t = target[:, c]
        inter = (p * t).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        out[f"dice_{name}"] = float(((2.0 * inter + 1e-6) / (union + 1e-6)).mean())
    out["dice_mean"] = float(np.mean(list(out.values())))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.json")
    parser.add_argument("--aux_csv", type=str,
                        default="visualization/morphology/morphology_features.csv")
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

    # Evaluate with batch_size 1 to record per-case Dice
    eval_cfg = {**cfg}
    eval_cfg["train"] = {**cfg["train"], "batch_size": 1}
    eval_cfg["imbalance"] = {**cfg.get("imbalance", {}), "strategy": "none"}
    loaders = build_dataloaders(
        manifest_path=args.manifest, config=eval_cfg, aux_features_csv=args.aux_csv,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []
    case_ids: List[str] = []
    dice_records: Dict[str, List[float]] = {"dice_WT": [], "dice_TC": [], "dice_ET": [], "dice_mean": []}

    for batch in loaders[args.split]:
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        missing = batch["missing_mask"].to(device)
        out = model(image, missing_mask=missing, aux_features=aux, return_aux=False)

        probs = torch.softmax(out["cls"].float(), dim=1).cpu().numpy()[0]
        all_probs.append(probs)
        all_labels.append(int(batch["label"].item()))
        case_ids.append(batch["case_id"][0])

        if "seg" in batch:
            d = _dice_per_channel(out["seg"].float(), batch["seg"].to(device))
            for k, v in d.items():
                dice_records[k].append(v)

    probs = np.stack(all_probs)
    labels = np.array(all_labels)
    metrics = classification_metrics(probs, labels)
    metrics["confusion_matrix"] = confusion_matrix_dict(probs, labels)
    for k, v in dice_records.items():
        if v:
            metrics[k] = float(np.mean(v))
    metrics["n_cases"] = int(len(labels))

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
