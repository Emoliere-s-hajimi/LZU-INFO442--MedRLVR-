from .eda import run_eda


def __getattr__(name):
    if name in {"scan_dataset", "scan_nifti_root",
                "scan_structural_dicom_root", "scan_pwi_root",
                "merge_scans", "write_scans", "CaseScan"}:
        from . import cohort_scan as _scan

        return getattr(_scan, name)
    if name in {"run_storytelling", "aggregate", "CohortStats",
                "write_markdown", "sample_intensities"}:
        from . import storytelling as _st

        return getattr(_st, name)
    if name in {"compute_features", "run_morphology_analysis",
                "MorphFeatures", "build_dataframe", "class_contrast_tests"}:
        from . import morphology as _morph

        return getattr(_morph, name)
    raise AttributeError(name)
