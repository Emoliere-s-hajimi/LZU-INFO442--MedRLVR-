"""Training loop entry point. Run via `python -m src.train --config configs/default.yaml`."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import build_dataloaders
from .losses import DiceCELoss, FocalLoss, MultiTaskLoss
from .models import build_model
from .utils import dump_config, get_logger, load_config, log_to_file, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.json")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    return torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("train")
    log_to_file(logger, str(out_dir / "train.log"))
    dump_config(cfg, str(out_dir / "config_snapshot.yaml"))
    logger.info("output dir: %s", out_dir)

    loaders = build_dataloaders(
        manifest_path=args.manifest,
        modalities=cfg["data"]["modalities"],
        patch_size=tuple(cfg["data"]["patch_size"]),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        split=cfg["data"]["train_val_test_split"],
        seed=cfg["project"]["seed"],
        use_weighted_sampler=cfg["imbalance"]["strategy"] == "weighted_sampler",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    logger.info("device: %s", device)
    logger.info("model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    loss_cfg = cfg["loss"]
    cls_loss = FocalLoss(class_weights=loss_cfg.get("class_weights"))
    seg_loss = DiceCELoss()
    criterion = MultiTaskLoss(cls_loss=cls_loss, seg_loss=seg_loss, ds_weights=loss_cfg["ds_weights"])

    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    start_epoch = 0
    if args.resume:
        from .utils import load_checkpoint

        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=str(device))
        start_epoch = int(state.get("epoch", 0)) + 1
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    log_csv = out_dir / "training_log.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_loss", "val_cls_loss", "val_seg_loss"])

    best_val = float("inf")
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            targets = {"label": batch["label"].to(device)}
            if "seg" in batch:
                targets["seg"] = batch["seg"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                outputs = model(image)
                loss_dict = criterion(outputs, targets)
            scaler.scale(loss_dict["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += float(loss_dict["total"])
            n_batches += 1
            pbar.set_postfix({k: float(v) for k, v in loss_dict.items() if torch.is_tensor(v)})

        scheduler.step()
        train_loss = running / max(n_batches, 1)
        val_stats = validate(model, loaders["val"], criterion, device, cfg)
        logger.info(
            "epoch %d | train %.4f | val %.4f | cls %.4f | seg %.4f | lr %.2e",
            epoch, train_loss, val_stats["val_loss"], val_stats["cls"], val_stats["seg"], scheduler.get_last_lr()[0],
        )

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{scheduler.get_last_lr()[0]:.6e}",
                f"{train_loss:.6f}",
                f"{val_stats['val_loss']:.6f}",
                f"{val_stats['cls']:.6f}",
                f"{val_stats['seg']:.6f}",
            ])

        save_checkpoint(str(out_dir / "last.pt"), model, optimizer, scheduler, epoch=epoch, extra={"val_loss": val_stats["val_loss"]})
        if val_stats["val_loss"] < best_val:
            best_val = val_stats["val_loss"]
            save_checkpoint(str(out_dir / "best.pt"), model, optimizer, scheduler, epoch=epoch, extra={"val_loss": best_val})
            logger.info("new best val loss: %.4f", best_val)


@torch.no_grad()
def validate(model, loader, criterion, device, cfg) -> Dict[str, float]:
    model.eval()
    total = 0.0
    cls = 0.0
    seg = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        targets = {"label": batch["label"].to(device)}
        if "seg" in batch:
            targets["seg"] = batch["seg"].to(device)
        with autocast(enabled=cfg["train"]["amp"]):
            outputs = model(image, return_aux=False)
            loss_dict = criterion(outputs, targets)
        total += float(loss_dict["total"])
        cls += float(loss_dict["cls"])
        seg += float(loss_dict["seg"])
        n += 1
    n = max(n, 1)
    return {"val_loss": total / n, "cls": cls / n, "seg": seg / n}


if __name__ == "__main__":
    main()
