"""Metric computation for the recurrence-vs-necrosis task and the auxiliary
segmentation. Kept dependency-light (numpy + scikit-learn) so it is easy to
call from any stage of the pipeline.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    positive_index: int = 1,
) -> Dict[str, float]:
    """Binary classification metrics derived from a softmax matrix."""
    if probs.ndim == 1:
        positive_probs = probs
    else:
        positive_probs = probs[:, positive_index]
    preds = (positive_probs >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
    }

    if len(np.unique(labels)) == 2:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        denom = tn + fp
        metrics["specificity"] = float(tn / denom) if denom > 0 else 0.0
        try:
            metrics["auc"] = float(roc_auc_score(labels, positive_probs))
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics


def confusion_matrix_dict(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    preds = (probs[:, 1] if probs.ndim > 1 else probs) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, preds.astype(int), labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def dice_score(pred: np.ndarray, target: np.ndarray, num_classes: Optional[int] = None) -> Dict[str, float]:
    """Per-class and mean Dice. Inputs are integer label maps of equal shape."""
    if num_classes is None:
        num_classes = int(max(pred.max(), target.max())) + 1
    out: Dict[str, float] = {}
    dices = []
    for c in range(1, num_classes):  # skip background
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        d = float(2 * inter / max(union, 1e-8))
        out[f"dice_class_{c}"] = d
        dices.append(d)
    out["dice_mean"] = float(np.mean(dices)) if dices else float("nan")
    return out


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance — optional, requires scipy."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float("nan")
    pred_b = pred > 0
    target_b = target > 0
    if not pred_b.any() or not target_b.any():
        return float("nan")
    pred_dist = distance_transform_edt(~pred_b)
    target_dist = distance_transform_edt(~target_b)
    a = pred_dist[target_b]
    b = target_dist[pred_b]
    return float(np.percentile(np.concatenate([a, b]), 95))


def aggregate_classification(
    all_probs: Sequence[np.ndarray],
    all_labels: Sequence[int],
) -> Dict[str, float]:
    probs = np.stack(all_probs)
    labels = np.asarray(all_labels)
    return classification_metrics(probs, labels)
