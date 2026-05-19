"""Visualizations for model predictions: GT vs prediction overlays, ROC curve,
confusion matrix, calibration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _safe_import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def gt_vs_pred_overlay(
    image_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    out_path: Optional[str] = None,
    alpha: float = 0.4,
):
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image_slice, cmap="gray"); axes[0].set_title("MRI"); axes[0].axis("off")

    axes[1].imshow(image_slice, cmap="gray")
    axes[1].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap="autumn", alpha=alpha)
    axes[1].set_title("Ground truth"); axes[1].axis("off")

    axes[2].imshow(image_slice, cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap="winter", alpha=alpha)
    axes[2].set_title("Prediction"); axes[2].axis("off")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(probs: np.ndarray, labels: np.ndarray, out_path: Optional[str] = None):
    from sklearn.metrics import auc, roc_curve

    plt = _safe_import_plt()
    pos = probs[:, 1] if probs.ndim > 1 else probs
    fpr, tpr, _ = roc_curve(labels, pos)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Recurrence vs radiation necrosis — ROC")
    ax.legend(loc="lower right")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(matrix: dict, labels: Sequence[str] = ("necrosis", "recurrence"), out_path: Optional[str] = None):
    plt = _safe_import_plt()
    cm = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(2)); ax.set_yticks(range(2))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, out_path: Optional[str] = None):
    plt = _safe_import_plt()
    pos = probs[:, 1] if probs.ndim > 1 else probs
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(pos, bins) - 1
    bin_acc = []
    bin_conf = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            bin_acc.append(float(labels[mask].mean()))
            bin_conf.append(float(pos[mask].mean()))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="grey", label="ideal")
    ax.plot(bin_conf, bin_acc, "o-", label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram")
    ax.legend()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
