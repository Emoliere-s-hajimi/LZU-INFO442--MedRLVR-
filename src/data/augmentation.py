"""Composable augmentation transforms operating on a `dict[str, np.ndarray]`.

Each transform has the same API: it receives a sample dict (modality name →
volume, plus optional `seg`) and returns the transformed sample. Geometric
transforms are applied identically across all modalities and the segmentation;
intensity transforms only touch image modalities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np


Sample = dict


@dataclass
class Compose:
    transforms: Sequence[Callable[[Sample], Sample]]

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample


def _image_keys(sample: Sample) -> List[str]:
    return [k for k in sample if k != "seg"]


@dataclass
class RandomFlip:
    axes: Iterable[int] = (0, 1, 2)
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        for axis in self.axes:
            if np.random.rand() < self.p:
                for k in sample:
                    sample[k] = np.flip(sample[k], axis=axis).copy()
        return sample


@dataclass
class RandomIntensityShift:
    shift: float = 0.05
    scale: float = 0.1
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() < self.p:
            for k in _image_keys(sample):
                shift = np.random.uniform(-self.shift, self.shift)
                scale = np.random.uniform(1 - self.scale, 1 + self.scale)
                sample[k] = sample[k] * scale + shift
        return sample


@dataclass
class RandomGaussianNoise:
    sigma: float = 0.02
    p: float = 0.3

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() < self.p:
            for k in _image_keys(sample):
                sample[k] = sample[k] + np.random.normal(0, self.sigma, sample[k].shape).astype(sample[k].dtype)
        return sample


@dataclass
class RandomGamma:
    gamma_range: tuple = (0.8, 1.2)
    p: float = 0.3

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() < self.p:
            for k in _image_keys(sample):
                arr = sample[k]
                lo, hi = arr.min(), arr.max()
                if hi - lo < 1e-6:
                    continue
                norm = (arr - lo) / (hi - lo + 1e-8)
                gamma = np.random.uniform(*self.gamma_range)
                sample[k] = norm ** gamma * (hi - lo) + lo
        return sample


def default_train_transforms() -> Compose:
    return Compose([
        RandomFlip(),
        RandomIntensityShift(),
        RandomGamma(),
        RandomGaussianNoise(),
    ])
