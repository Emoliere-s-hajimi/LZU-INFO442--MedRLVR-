"""Training entry point for the NR subproject.

Builds two disjoint dataloaders directly from the train/val manifests
emitted by ``nr.preprocess`` (no random re-splitting). Everything else
is borrowed from ``src.train``: model, criterion, optimizer, scheduler,
EMA, validation loop, checkpoint format.

Usage::

    python -m nr.train \\
        --config nr_subproject/configs/nr.yaml \\
        --train_manifest nr_subproject/processed/train_manifest.json \\
        --val_manifest   nr_subproject/processed/val_manifest.json
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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data.dataset import LABEL_MAP, MultiModalMRIDataset, _label_to_int
from src.models import build_model
from src.train import (
    ModelEMA,
    _maybe_tb_writer,
    build_criterion,
    build_optimizer,
    build_scheduler,
    validate,
)
from src.utils import (
    dump_config,
    get_logger,
    load_config,
    log_to_file,
    save_checkpoint,
    set_seed,
)


def _load_manifest(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of records")
    return rows


def _build_loader(manifest: List[Dict], cfg: Dict, *, train: bool) -> DataLoader:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    md_p = float(cfg.get("model", {}).get("modality_dropout_p", 0.0)) if train else 0.0
    ds = MultiModalMRIDataset(
        manifest=manifest,
        patch_size=tuple(data_cfg["patch_size"]),
        augment=train,
        modality_dropout_p=md_p,
    )
    sampler = None
    shuffle = train
    if train and cfg.get("imbalance", {}).get("strategy") == "weighted_sampler":
        labels = np.array([_label_to_int(r.get("label")) for r in manifest], dtype=np.int64)
        counts = np.bincount(labels, minlength=len(LABEL_MAP)).astype(np.float64)
        class_w = 1.0 / np.clip(counts, 1.0, None)
        weights = class_w[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=train,
    )


def _write_csv_header(path: Path, header: List[str]) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="nr_subproject/configs/nr.yaml")
    ap.add_argument("--train_manifest", type=str,
                    default="nr_subproject/processed/train_manifest.json")
    ap.add_argument("--val_manifest", type=str,
                    default="nr_subproject/processed/val_manifest.json")
    ap.add_argument("--resume", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("nr-train")
    log_to_file(logger, str(out_dir / "train.log"))
    dump_config(cfg, str(out_dir / "config_snapshot.yaml"))
    logger.info("output dir: %s", out_dir)

    train_manifest = _load_manifest(args.train_manifest)
    val_manifest = _load_manifest(args.val_manifest)
    if not train_manifest:
        raise SystemExit(f"empty train manifest: {args.train_manifest}")
    if not val_manifest:
        raise SystemExit(f"empty val manifest: {args.val_manifest}")

    train_loader = _build_loader(train_manifest, cfg, train=True)
    val_loader = _build_loader(val_manifest, cfg, train=False)
    logger.info("train: %d cases · val: %d cases", len(train_manifest), len(val_manifest))

    val_label_counts = {}
    for r in val_manifest:
        val_label_counts[r.get("label")] = val_label_counts.get(r.get("label"), 0) + 1
    if len(val_label_counts) < 2:
        logger.warning("val set is single-class %s — AUC will be NaN; "
                       "use loss / sensitivity for selection",
                       val_label_counts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("device: %s · model parameters: %.2f M", device, n_params)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    ema = ModelEMA(model, decay=cfg["train"].get("ema_decay", 0.999)) \
        if cfg["train"].get("use_ema", False) else None
    scaler = GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enabled=cfg["train"]["amp"] and torch.cuda.is_available(),
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_metric = -float("inf")
    if args.resume:
        from src.utils import load_checkpoint
        state = load_checkpoint(args.resume, model, optimizer, scheduler,
                                map_location=str(device))
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_val_metric = float(state.get("best_val_metric", best_val_metric))
        logger.info("resumed from %s at epoch %d", args.resume, start_epoch)

    log_csv = out_dir / "training_log.csv"
    _write_csv_header(log_csv, [
        "epoch", "lr", "train_loss", "val_loss",
        "val_cls", "val_seg_dice", "val_auc", "val_acc", "val_sens", "val_spec",
        "dice_WT", "dice_TC", "dice_ET", "dice_mean",
    ])
    tb = _maybe_tb_writer(out_dir)
    patience = int(cfg["train"].get("early_stop_patience", 30))
    no_improve = 0

    last_epoch = start_epoch
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        last_epoch = epoch
        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            targets = {
                "label": batch["label"].to(device),
                "seg": batch["seg"].to(device),
            }
            aux = batch["aux"].to(device)
            missing = batch["missing_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                enabled=cfg["train"]["amp"] and torch.cuda.is_available(),
            ):
                outputs = model(image, missing_mask=missing, aux_features=aux)
                loss_dict = criterion(outputs, targets)
            scaler.scale(loss_dict["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running += loss_dict["total"].item()
            n_batches += 1
            pbar.set_postfix({k: f"{v.detach().item():.3f}" if torch.is_tensor(v) else f"{v:.3f}"
                              for k, v in loss_dict.items() if k in {"total", "cls", "seg_dice"}})

        scheduler.step()
        train_loss = running / max(n_batches, 1)
        val_model = ema.shadow if ema is not None else model
        val = validate(val_model, val_loader, criterion, device, cfg)
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            "epoch %d | lr %.2e | train %.4f | val %.4f | "
            "auc %.3f acc %.3f sens %.3f spec %.3f | "
            "dice mean %.3f",
            epoch, lr_now, train_loss, val["val_loss"],
            val["val_auc"], val["val_acc"], val["val_sens"], val["val_spec"],
            val["dice_mean"],
        )

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr_now:.6e}", f"{train_loss:.6f}", f"{val['val_loss']:.6f}",
                f"{val['val_cls']:.6f}", f"{val['val_seg_dice']:.6f}",
                f"{val['val_auc']:.6f}", f"{val['val_acc']:.6f}",
                f"{val['val_sens']:.6f}", f"{val['val_spec']:.6f}",
                f"{val['dice_WT']:.6f}", f"{val['dice_TC']:.6f}",
                f"{val['dice_ET']:.6f}", f"{val['dice_mean']:.6f}",
            ])

        if tb is not None:
            tb.add_scalar("train/loss", train_loss, epoch)
            tb.add_scalar("train/lr", lr_now, epoch)
            for k, v in val.items():
                if isinstance(v, float) and math.isfinite(v):
                    tb.add_scalar(f"val/{k}", v, epoch)

        ckpt_extra = {
            "val_loss": val["val_loss"], "val_auc": val["val_auc"],
            "best_val_loss": best_val_loss, "best_val_metric": best_val_metric,
        }
        if ema is not None:
            ckpt_extra["ema"] = ema.state_dict()
        save_checkpoint(str(out_dir / "last.pt"), model, optimizer, scheduler,
                        epoch=epoch, extra=ckpt_extra)

        improved = False
        if math.isfinite(val["val_loss"]) and val["val_loss"] < best_val_loss:
            best_val_loss = val["val_loss"]
            ckpt_extra["best_val_loss"] = best_val_loss
            save_checkpoint(str(out_dir / "best_loss.pt"), model, optimizer, scheduler,
                            epoch=epoch, extra=ckpt_extra)
            logger.info("new best val_loss: %.4f", best_val_loss)
            improved = True

        # AUC may be NaN on the necrosis-only val set; fall back to sensitivity.
        primary = val["val_auc"] if math.isfinite(val["val_auc"]) else val["val_sens"]
        if math.isfinite(primary) and primary > best_val_metric:
            best_val_metric = primary
            ckpt_extra["best_val_metric"] = best_val_metric
            save_checkpoint(str(out_dir / "best_metric.pt"), model, optimizer, scheduler,
                            epoch=epoch, extra=ckpt_extra)
            logger.info("new best val_metric (auc-or-sens): %.4f", best_val_metric)
            improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    if tb is not None:
        tb.close()

    summary = {
        "best_val_loss": best_val_loss,
        "best_val_metric": best_val_metric,
        "stopped_at_epoch": last_epoch,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("training summary: %s", summary)


if __name__ == "__main__":
    main()
