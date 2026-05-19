"""Multimodal MRI dataset for glioma recurrence vs radiation necrosis.

Supports both 3D NIfTI volumes (BraTS-style layout from the cleaned manifest)
and 2D slice triplets exported as PNG (the format shipped in `data_example/`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augmentation import default_train_transforms
from .preprocessing import center_on_lesion, crop_or_pad, normalize_zscore


LABEL_MAP = {"necrosis": 0, "recurrence": 1}


def _load_nifti(path: str) -> np.ndarray:
    import nibabel as nib

    return nib.load(path).get_fdata().astype(np.float32)


def _load_png_stack(paths: Sequence[str]) -> np.ndarray:
    import cv2

    slices = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
    return np.stack(slices, axis=-1).astype(np.float32)


class MultiModalMRIDataset(Dataset):
    def __init__(
        self,
        manifest: List[Dict],
        modalities: Sequence[str] = ("t1", "t1ce", "t2", "flair"),
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        load_segmentation: bool = True,
    ) -> None:
        self.manifest = manifest
        self.modalities = list(modalities)
        self.patch_size = tuple(patch_size)
        self.augment = augment
        self.load_segmentation = load_segmentation
        self.transform = default_train_transforms() if augment else None

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_case(self, record: Dict) -> Dict[str, np.ndarray]:
        volumes: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            path = record["modalities"][m]
            if isinstance(path, list):
                volumes[m] = _load_png_stack(path)
            else:
                volumes[m] = _load_nifti(path)
        if self.load_segmentation and record.get("seg"):
            seg_path = record["seg"]
            seg = _load_nifti(seg_path) if isinstance(seg_path, str) else _load_png_stack(seg_path)
            volumes["seg"] = seg.astype(np.int64)
        return volumes

    def _augment(self, vols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.transform is None:
            return vols
        return self.transform(vols)

    def __getitem__(self, idx: int):
        record = self.manifest[idx]
        vols = self._load_case(record)
        for m in self.modalities:
            vols[m] = normalize_zscore(vols[m])

        seg = vols.get("seg")
        center = center_on_lesion(seg, fallback=tuple(s // 2 for s in vols[self.modalities[0]].shape))
        ps = self.patch_size
        starts = [max(0, c - p // 2) for c, p in zip(center, ps)]
        slices = tuple(slice(s, s + p) for s, p in zip(starts, ps))
        for k in vols:
            vols[k] = crop_or_pad(vols[k][slices], ps)

        if self.augment:
            vols = self._augment(vols)

        image = np.stack([vols[m] for m in self.modalities], axis=0)
        sample = {
            "image": torch.from_numpy(image).float(),
            "case_id": record["case_id"],
            "label": torch.tensor(LABEL_MAP[record["label"]], dtype=torch.long),
        }
        if "seg" in vols:
            sample["seg"] = torch.from_numpy(vols["seg"]).long()
        return sample


def build_dataloaders(
    manifest_path: str,
    modalities: Sequence[str],
    patch_size: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    split: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 442,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    with open(manifest_path) as f:
        manifest = json.load(f)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(manifest))
    rng.shuffle(indices)
    n_train = int(len(indices) * split[0])
    n_val = int(len(indices) * split[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    splits = {
        "train": [manifest[i] for i in train_idx],
        "val": [manifest[i] for i in val_idx],
        "test": [manifest[i] for i in test_idx],
    }

    loaders: Dict[str, DataLoader] = {}
    for name, subset in splits.items():
        dataset = MultiModalMRIDataset(
            manifest=subset,
            modalities=modalities,
            patch_size=patch_size,
            augment=(name == "train"),
        )
        sampler = None
        if name == "train" and use_weighted_sampler:
            labels = np.array([LABEL_MAP[r["label"]] for r in subset])
            class_counts = np.bincount(labels, minlength=len(LABEL_MAP)).astype(np.float64)
            class_weights = 1.0 / np.clip(class_counts, 1.0, None)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None and name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(name == "train"),
        )
    return loaders
