"""Training entry point for BrainTTNet.

Usage::

    python -m src.train --config configs/default.yaml \\
                        --manifest data/processed/manifest.json

The loop saves:
    <output_dir>/
      ├── config_snapshot.yaml      # frozen training config
      ├── train.log                 # rolling text log
      ├── training_log.csv          # one row per epoch
      ├── tb/                       # TensorBoard event files
      ├── last.pt                   # latest checkpoint (overwritten each epoch)
      ├── best_loss.pt              # lowest val loss seen
      └── best_auc.pt               # highest val AUC seen
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import copy

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from tqdm import tqdm

from .data import build_dataloaders
from .losses import (
    FocalBCEWithLogits,
    FocalLoss,
    LogVolumeWeightedDice,
    MultiLabelDice,
    MultiTaskLoss,
    NestingPenalty,
    TopologyChiRegulariser,
)
from .metrics import classification_metrics
from .models import build_model
from .utils import (
    dump_config,
    get_logger,
    load_config,
    log_to_file,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_criterion(cfg: Dict) -> MultiTaskLoss:
    loss_cfg = cfg.get("loss", {})
    cls_weights = loss_cfg.get("class_weights")

    cls_loss = FocalLoss(
        alpha=loss_cfg.get("focal_alpha", 0.25),
        gamma=loss_cfg.get("focal_gamma", 2.0),
        class_weights=cls_weights,
    )
    seg_dice = MultiLabelDice(channel_weights=loss_cfg.get("seg_channel_weights"))
    seg_focal = FocalBCEWithLogits(
        alpha=loss_cfg.get("seg_focal_alpha", 0.25),
        gamma=loss_cfg.get("seg_focal_gamma", 2.0),
    )
    log_vol = LogVolumeWeightedDice()
    nesting = NestingPenalty(weight=loss_cfg.get("nesting_weight", 0.1))
    chi = TopologyChiRegulariser(
        per_class_chi=loss_cfg.get("class_chi", {0: -24.0, 1: 4.0}),
        default_chi=loss_cfg.get("default_chi", -3.0),
        weight=loss_cfg.get("chi_weight", 0.05),
    )

    return MultiTaskLoss(
        cls_loss=cls_loss,
        seg_dice_loss=seg_dice,
        seg_focal_loss=seg_focal,
        log_volume_loss=log_vol,
        nesting_loss=nesting,
        chi_loss=chi,
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        seg_weight=loss_cfg.get("seg_weight", 1.0),
        log_volume_weight=loss_cfg.get("log_volume_weight", 0.5),
        ds_weights=loss_cfg.get("ds_weights", [1.0, 0.4, 0.3]),
    )


def build_optimizer(model: torch.nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    no_decay = []
    decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay,     "weight_decay": t["weight_decay"]},
            {"params": no_decay,  "weight_decay": 0.0},
        ],
        lr=t["lr"],
        betas=t.get("betas", (0.9, 0.999)),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, steps_per_epoch: int):
    """Linear warmup → cosine annealing on epoch boundaries."""
    t = cfg["train"]
    warmup_epochs = t.get("warmup_epochs", 5)
    epochs = t["epochs"]

    def warmup_lambda(epoch_idx: int) -> float:
        return float(epoch_idx + 1) / max(warmup_epochs, 1)

    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _maybe_tb_writer(out_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(out_dir / "tb"))
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Exponential Moving Average (EMA) of model weights
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model weights.

    EMA validation usually tracks the held-out distribution more closely
    than the raw weights on small medical-imaging cohorts where the
    training trajectory wobbles. Enabled when ``train.use_ema`` is true.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        ssd = self.shadow.state_dict()
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                ssd[k].mul_(d).add_(v, alpha=1 - d)
            else:
                ssd[k].copy_(v)

    def state_dict(self):
        return self.shadow.state_dict()

    @torch.no_grad()
    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_dice(seg_logits: torch.Tensor, seg_target: torch.Tensor,
                  threshold: float = 0.5) -> Dict[str, float]:
    """Per-channel Dice on the cleaned (3, D, H, W) binary target."""
    pred = (torch.sigmoid(seg_logits) >= threshold).float()
    target = seg_target.float()
    dice = {}
    names = ("WT", "TC", "ET")
    for c, name in enumerate(names):
        p = pred[:, c]
        t = target[:, c]
        inter = (p * t).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        d = (2.0 * inter + 1e-6) / (union + 1e-6)
        dice[f"dice_{name}"] = float(d.mean())
    dice["dice_mean"] = float(np.mean(list(dice.values())))
    return dice


@torch.no_grad()
def validate(model, loader, criterion, device, cfg) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    sub_losses: Dict[str, float] = {}
    n_batches = 0
    all_probs: List[float] = []
    all_labels: List[int] = []
    dice_acc: Dict[str, float] = {"dice_WT": 0.0, "dice_TC": 0.0, "dice_ET": 0.0, "dice_mean": 0.0}

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        targets = {
            "label": batch["label"].to(device),
            "seg": batch["seg"].to(device),
        }
        aux = batch["aux"].to(device)
        missing = batch["missing_mask"].to(device)
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                      enabled=cfg["train"]["amp"] and torch.cuda.is_available()):
            outputs = model(image, missing_mask=missing, aux_features=aux,
                            return_aux=False)
            loss_dict = criterion(outputs, targets)
        total_loss += loss_dict["total"].item()
        for k in ("cls", "seg_dice", "seg_focal", "log_vol", "nest", "chi"):
            v = loss_dict[k]
            sub_losses[k] = sub_losses.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
        n_batches += 1

        # Collect probs for ROC-AUC
        probs = torch.softmax(outputs["cls"].float(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].cpu().numpy().tolist())

        # Dice
        case_dice = _compute_dice(outputs["seg"].float(), targets["seg"])
        for k, v in case_dice.items():
            dice_acc[k] += v

    n = max(n_batches, 1)
    if n_batches == 0 or len(all_probs) == 0:
        return {
            "val_loss": float("nan"),
            "val_cls": float("nan"), "val_seg_dice": float("nan"),
            "val_seg_focal": float("nan"), "val_log_vol": float("nan"),
            "val_nest": float("nan"), "val_chi": float("nan"),
            "dice_WT": float("nan"), "dice_TC": float("nan"),
            "dice_ET": float("nan"), "dice_mean": float("nan"),
            "val_auc": float("nan"), "val_acc": float("nan"),
            "val_sens": float("nan"), "val_spec": float("nan"),
            "val_f1": float("nan"),
        }
    metrics = classification_metrics(
        probs=np.array(all_probs), labels=np.array(all_labels), threshold=0.5,
    )

    return {
        "val_loss": total_loss / n,
        **{f"val_{k}": v / n for k, v in sub_losses.items()},
        **{k: v / n for k, v in dice_acc.items()},
        "val_auc": metrics.get("auc", float("nan")),
        "val_acc": metrics.get("accuracy", float("nan")),
        "val_sens": metrics.get("sensitivity", float("nan")),
        "val_spec": metrics.get("specificity", float("nan")),
        "val_f1": metrics.get("f1", float("nan")),
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.json")
    parser.add_argument("--aux_csv", type=str,
                        default="visualization/morphology/morphology_features.csv")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def _write_csv_header(path: Path, header: List[str]) -> None:
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


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
        config=cfg,
        aux_features_csv=args.aux_csv,
    )
    logger.info(
        "loaders — train: %d  val: %d  test: %d",
        len(loaders["train"].dataset), len(loaders["val"].dataset), len(loaders["test"].dataset),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("device: %s · model parameters: %.2f M", device, n_params)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))
    ema = ModelEMA(model, decay=cfg["train"].get("ema_decay", 0.999)) \
        if cfg["train"].get("use_ema", False) else None
    scaler = GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enabled=cfg["train"]["amp"] and torch.cuda.is_available(),
    )

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_auc = -float("inf")
    if args.resume:
        from .utils import load_checkpoint

        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=str(device))
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_val_auc = float(state.get("best_val_auc", best_val_auc))
        logger.info("resumed from %s at epoch %d (best_loss=%.4f best_auc=%.4f)",
                    args.resume, start_epoch, best_val_loss, best_val_auc)

    log_csv = out_dir / "training_log.csv"
    _write_csv_header(log_csv, [
        "epoch", "lr", "train_loss", "val_loss",
        "val_cls", "val_seg_dice", "val_log_vol", "val_nest", "val_chi",
        "val_auc", "val_acc", "val_sens", "val_spec",
        "dice_WT", "dice_TC", "dice_ET", "dice_mean",
    ])

    tb = _maybe_tb_writer(out_dir)
    patience = int(cfg["train"].get("early_stop_patience", 30))
    no_improve = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        n_batches = 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device, non_blocking=True)
            targets = {
                "label": batch["label"].to(device),
                "seg": batch["seg"].to(device),
            }
            aux = batch["aux"].to(device)
            missing = batch["missing_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                      enabled=cfg["train"]["amp"] and torch.cuda.is_available()):
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
        val = validate(val_model, loaders["val"], criterion, device, cfg)
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            "epoch %d | lr %.2e | train %.4f | val %.4f | auc %.3f acc %.3f sens %.3f spec %.3f "
            "| dice WT %.3f TC %.3f ET %.3f mean %.3f",
            epoch, lr_now, train_loss, val["val_loss"],
            val["val_auc"], val["val_acc"], val["val_sens"], val["val_spec"],
            val["dice_WT"], val["dice_TC"], val["dice_ET"], val["dice_mean"],
        )

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{lr_now:.6e}", f"{train_loss:.6f}", f"{val['val_loss']:.6f}",
                f"{val['val_cls']:.6f}", f"{val['val_seg_dice']:.6f}",
                f"{val['val_log_vol']:.6f}", f"{val['val_nest']:.6f}", f"{val['val_chi']:.6f}",
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

        # Checkpoints
        ckpt_extra = {
            "val_loss": val["val_loss"], "val_auc": val["val_auc"],
            "best_val_loss": best_val_loss, "best_val_auc": best_val_auc,
        }
        if ema is not None:
            ckpt_extra["ema"] = ema.state_dict()
        save_checkpoint(str(out_dir / "last.pt"), model, optimizer, scheduler,
                        epoch=epoch, extra=ckpt_extra)
        improved = False
        if val["val_loss"] < best_val_loss:
            best_val_loss = val["val_loss"]
            ckpt_extra["best_val_loss"] = best_val_loss
            save_checkpoint(str(out_dir / "best_loss.pt"), model, optimizer, scheduler,
                            epoch=epoch, extra=ckpt_extra)
            logger.info("new best val_loss: %.4f", best_val_loss)
            improved = True
        if math.isfinite(val["val_auc"]) and val["val_auc"] > best_val_auc:
            best_val_auc = val["val_auc"]
            ckpt_extra["best_val_auc"] = best_val_auc
            save_checkpoint(str(out_dir / "best_auc.pt"), model, optimizer, scheduler,
                            epoch=epoch, extra=ckpt_extra)
            logger.info("new best val_auc: %.4f", best_val_auc)
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

    # Final summary
    summary = {
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
        "stopped_at_epoch": epoch,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("training summary: %s", summary)


if __name__ == "__main__":
    main()
