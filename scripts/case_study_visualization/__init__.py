"""Case-study visualisations.

Each module here takes a single cleaned ``.npz`` and produces an
intuitive figure (or short set of figures) that illustrates **one** of
the medical priors mined in M2:

    case_study/anatomy.py            — three-orthogonal-view radiology panel
    case_study/tumor_3d.py           — marching-cubes WT/TC/ET surface rendering
    case_study/topology.py           — connected components, Euler χ, cavities
    case_study/morphology.py         — sphericity proxy, principal axes,
                                       distance transform
    case_study/modality_signature.py — per-modality inside/outside histograms
    case_study/run_case_study.py     — run all of the above on a list of cases

Outputs land under ``visualization/case_study/<case_id>/``.
"""
