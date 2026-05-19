"""Inter-modality rigid registration to a chosen reference modality (default T1ce).

Registration is essential before stacking modalities into a single tensor,
since the raw scans from the cohort are not always co-registered. Uses
SimpleITK's `ImageRegistrationMethod` with a Mattes mutual information metric
and a versor-based rigid transform.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def rigid_register(
    moving: np.ndarray,
    fixed: np.ndarray,
    spacing_moving: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    spacing_fixed: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:  # pragma: no cover
        import warnings

        warnings.warn("SimpleITK not installed — registration skipped, returning moving as-is")
        return moving

    fixed_img = sitk.GetImageFromArray(fixed.astype(np.float32))
    fixed_img.SetSpacing(spacing_fixed)
    moving_img = sitk.GetImageFromArray(moving.astype(np.float32))
    moving_img.SetSpacing(spacing_moving)

    initial_tx = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_tx, inPlace=False)

    final_tx = reg.Execute(fixed_img, moving_img)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_tx)
    out_img = resampler.Execute(moving_img)
    return sitk.GetArrayFromImage(out_img)


def register_case_to_reference(
    case_volumes: Dict[str, np.ndarray],
    reference_modality: str = "t1ce",
    spacing: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Dict[str, np.ndarray]:
    if reference_modality not in case_volumes:
        return case_volumes
    fixed = case_volumes[reference_modality]
    sp = spacing or {}
    out: Dict[str, np.ndarray] = {reference_modality: fixed}
    for name, vol in case_volumes.items():
        if name == reference_modality:
            continue
        out[name] = rigid_register(
            moving=vol,
            fixed=fixed,
            spacing_moving=sp.get(name, (1.0, 1.0, 1.0)),
            spacing_fixed=sp.get(reference_modality, (1.0, 1.0, 1.0)),
        )
    return out
