"""N4 bias-field correction wrapper around SimpleITK.

Bias-field artefacts are a major confounder when comparing T1ce signal across
patients in a post-radiation cohort, so we standardise this step in the
cleaning pipeline. SimpleITK is an optional dependency: if it is unavailable
the call is a no-op and a warning is emitted.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def n4_bias_correction(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    shrink_factor: int = 4,
    n_iters: int = 50,
    n_levels: int = 4,
) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:  # pragma: no cover
        import warnings

        warnings.warn("SimpleITK not installed — N4 bias correction skipped")
        return volume

    img = sitk.GetImageFromArray(volume.astype(np.float32))

    if mask is None:
        mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    else:
        mask_img = sitk.GetImageFromArray((mask > 0).astype(np.uint8))
        mask_img.CopyInformation(img)

    if shrink_factor > 1:
        img_s = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
        mask_s = sitk.Shrink(mask_img, [shrink_factor] * img.GetDimension())
    else:
        img_s, mask_s = img, mask_img

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iters] * n_levels)
    corrected = corrector.Execute(img_s, mask_s)

    log_bias = corrector.GetLogBiasFieldAsImage(img)
    full = img / sitk.Exp(log_bias)
    return sitk.GetArrayFromImage(full)
