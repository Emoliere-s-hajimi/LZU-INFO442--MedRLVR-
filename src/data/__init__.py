"""Data package. Heavy submodules (`dataset`, `bias_correction`, `registration`)
are imported lazily so pure-Python helpers like `cleaning` and `preprocessing`
can be used without optional dependencies (torch, SimpleITK, nibabel)."""
from __future__ import annotations

from .cleaning import CleaningPipeline, load_manifest
from .modality_map import (
    CANONICAL_MODALITIES,
    classify_filename,
    classify_series_description,
    parse_chinese_label,
)
from .preprocessing import crop_or_pad, normalize_zscore, resample_volume
from .augmentation import (
    Compose,
    RandomFlip,
    RandomGamma,
    RandomGaussianNoise,
    RandomIntensityShift,
    default_train_transforms,
)


def __getattr__(name):
    if name in {"MultiModalMRIDataset", "build_dataloaders"}:
        from .dataset import MultiModalMRIDataset, build_dataloaders

        return {"MultiModalMRIDataset": MultiModalMRIDataset, "build_dataloaders": build_dataloaders}[name]
    if name in {
        "PreprocessConfig", "process_one_case", "process_cohort",
        "discover_nifti_cases", "discover_dicom_cases", "write_report",
        "INVALID_CASE_IDS",
    }:
        from . import pipeline as _pipe

        return getattr(_pipe, name)
    if name in {"DonorPool", "synthesize_one", "synthesize_cohort",
                "detect_missing_channels"}:
        from . import synthesis as _syn

        return getattr(_syn, name)
    if name in {"load_modality", "assemble_dicom_series", "load_nifti",
                "ModalityVolume", "find_modalities_in_case_dir",
                "discover_segmentation"}:
        from . import raw_io as _raw

        return getattr(_raw, name)
    if name == "n4_bias_correction":
        from .bias_correction import n4_bias_correction

        return n4_bias_correction
    if name in {"rigid_register", "register_case_to_reference"}:
        from .registration import rigid_register, register_case_to_reference

        return {"rigid_register": rigid_register, "register_case_to_reference": register_case_to_reference}[name]
    raise AttributeError(name)
