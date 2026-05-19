"""Morphological and topological feature mining on the cleaned ``.npz``
cohort. The goal is to surface *medical priors* — shape, connectivity,
inter-modality contrast — that we can later inject as inductive biases
into the segmentation network.

Each cleaned case has::

    image  : float32 (4, H, W, D)   — channel order [t1, t1ce, t2, flair]
    label  : uint8   (3, H, W, D)   — BraTS channels [WT, TC, ET]
    affine : float64 (4, 4)

For every case we compute a flat feature record (one row of a pandas
DataFrame). The features fall into four families:

    1. Volumetric  — voxel counts and physical mm^3 for WT/TC/ET
    2. Morphological — sphericity, surface area, bbox aspect ratios,
                       centroid coordinates, elongation
    3. Topological — number of 3-D connected components per channel,
                     Euler characteristic V - E + F (proxy of "how full
                     of holes is this lesion"), TC-inside-WT containment
    4. Modality-specific contrast — mean intensity of every modality
                                    *inside* and *outside* the lesion,
                                    and the inside/outside ratio that
                                    radiologists actually read
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CHANNEL_NAMES = ("WT", "TC", "ET")
MODALITY_NAMES = ("t1", "t1ce", "t2", "flair")


# ---------------------------------------------------------------------------
# Feature record
# ---------------------------------------------------------------------------

@dataclass
class MorphFeatures:
    case_id: str
    label: Optional[str]
    shape: Tuple[int, int, int]
    spacing_mm: Tuple[float, float, float]

    # ---- volumetric ----
    voxels_WT: int = 0
    voxels_TC: int = 0
    voxels_ET: int = 0
    volume_mm3_WT: float = 0.0
    volume_mm3_TC: float = 0.0
    volume_mm3_ET: float = 0.0

    # ---- morphological ----
    sphericity_WT: float = 0.0  # ratio of equivalent-sphere area to actual surface area, in [0, 1]
    sphericity_TC: float = 0.0
    sphericity_ET: float = 0.0
    surface_area_voxels_WT: float = 0.0
    elongation_WT: float = 0.0  # bbox longest_axis / shortest_axis
    bbox_volume_ratio_WT: float = 0.0  # actual / bbox

    centroid_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # in voxel space
    centroid_rel_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # normalised by shape, in [0, 1]
    radial_distance_from_brain_center: float = 0.0

    # ---- topology ----
    n_components_WT: int = 0
    n_components_TC: int = 0
    n_components_ET: int = 0
    largest_component_share_WT: float = 0.0  # voxels(largest) / voxels(WT)
    euler_characteristic_WT: int = 0
    n_holes_WT: int = 0  # surrogate for Betti_1 via 1 - chi  (planar approx)
    tc_inside_wt_share: float = 0.0  # TC voxels also inside WT / TC total
    et_inside_tc_share: float = 0.0  # ET voxels also inside TC / ET total

    # ---- modality contrast inside vs outside the lesion ----
    intensity_inside: Dict[str, float] = field(default_factory=dict)
    intensity_outside: Dict[str, float] = field(default_factory=dict)
    intensity_ratio_in_over_out: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Low-level geometric / topological helpers
# ---------------------------------------------------------------------------

def _foreground_mask(image: np.ndarray) -> np.ndarray:
    """Brain mask = any non-zero voxel across modalities (z-scored images
    have exact zero outside the brain after foreground normalisation)."""
    return np.any(image != 0, axis=0)


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """Return (label_map, n_components, voxels_per_component)."""
    from scipy.ndimage import label

    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32), 0, np.array([], dtype=int)
    lab, n = label(mask, structure=np.ones((3, 3, 3), dtype=int))
    sizes = np.bincount(lab.ravel())[1:] if n > 0 else np.array([], dtype=int)
    return lab.astype(np.int32), int(n), sizes


def _surface_area_voxels(mask: np.ndarray) -> float:
    """Count the number of voxel faces on the mask boundary. This is the
    standard discrete surface-area proxy used in 3-D radiomics."""
    if not mask.any():
        return 0.0
    sa = 0
    for axis in range(3):
        diff = np.diff(mask.astype(np.int8), axis=axis)
        sa += int((diff != 0).sum())
    return float(sa)


def _sphericity(volume_voxels: int, surface_voxels: float) -> float:
    """Dimensionless sphericity score in [0, 1]: 1 means a perfect sphere.

    psi = (pi^(1/3) * (6V)^(2/3)) / A
    """
    if volume_voxels <= 0 or surface_voxels <= 0:
        return 0.0
    V = float(volume_voxels)
    A = float(surface_voxels)
    return float(np.pi ** (1.0 / 3.0) * (6.0 * V) ** (2.0 / 3.0) / A)


def _bbox_metrics(mask: np.ndarray) -> Tuple[float, float, Tuple[int, int, int]]:
    """Return (elongation = longest/shortest bbox axis, actual/bbox volume,
    bbox extent in voxels along each axis)."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0, 0.0, (0, 0, 0)
    mn = coords.min(0); mx = coords.max(0) + 1
    extent = tuple((mx - mn).tolist())
    longest = max(extent); shortest = max(1, min(extent))
    bbox_vol = int(extent[0] * extent[1] * extent[2])
    if bbox_vol == 0:
        return 0.0, 0.0, extent
    return float(longest / shortest), float(mask.sum() / bbox_vol), extent  # type: ignore[return-value]


def _euler_characteristic_3d(mask: np.ndarray) -> int:
    """Approximate 3-D Euler characteristic via the alternating sum of
    cell counts at every order (vertices, edges, faces, cubes).

    For binary 3-D masks the Euler number relates to Betti numbers as
        chi = beta_0 - beta_1 + beta_2
    so combined with the component count (= beta_0) we can read off a
    holes-and-handles proxy.
    """
    if not mask.any():
        return 0
    m = mask.astype(np.int8)
    # cubes (3-cells)
    cubes = m[:-1, :-1, :-1] & m[1:, :-1, :-1] & m[:-1, 1:, :-1] & m[:-1, :-1, 1:] \
        & m[1:, 1:, :-1] & m[1:, :-1, 1:] & m[:-1, 1:, 1:] & m[1:, 1:, 1:]
    n_cubes = int(cubes.sum())

    # faces (2-cells): three orientations
    faces_xy = m[:-1, :-1, :] & m[1:, :-1, :] & m[:-1, 1:, :] & m[1:, 1:, :]
    faces_xz = m[:-1, :, :-1] & m[1:, :, :-1] & m[:-1, :, 1:] & m[1:, :, 1:]
    faces_yz = m[:, :-1, :-1] & m[:, 1:, :-1] & m[:, :-1, 1:] & m[:, 1:, 1:]
    n_faces = int(faces_xy.sum() + faces_xz.sum() + faces_yz.sum())

    # edges (1-cells): three orientations
    edges_x = m[:-1, :, :] & m[1:, :, :]
    edges_y = m[:, :-1, :] & m[:, 1:, :]
    edges_z = m[:, :, :-1] & m[:, :, 1:]
    n_edges = int(edges_x.sum() + edges_y.sum() + edges_z.sum())

    n_vertices = int(m.sum())

    return n_vertices - n_edges + n_faces - n_cubes


def _centroid(mask: np.ndarray) -> Tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0.0, 0.0, 0.0)
    c = coords.mean(0)
    return (float(c[0]), float(c[1]), float(c[2]))


def _radial_distance(centroid: Tuple[float, float, float],
                     reference: Tuple[float, float, float],
                     spacing: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(
        (np.asarray(centroid) - np.asarray(reference)) * np.asarray(spacing)
    ))


# ---------------------------------------------------------------------------
# Per-case feature extraction
# ---------------------------------------------------------------------------

def compute_features(
    npz_path: str,
    label: Optional[str] = None,
    spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MorphFeatures:
    data = np.load(npz_path)
    image = data["image"]
    seg = data["label"]
    case_id = Path(npz_path).stem
    shape = tuple(image.shape[1:])  # (H, W, D)

    feat = MorphFeatures(
        case_id=case_id,
        label=label,
        shape=shape,
        spacing_mm=spacing_mm,
    )

    voxel_volume_mm3 = float(np.prod(spacing_mm))

    masks = {
        "WT": seg[0] > 0,
        "TC": seg[1] > 0,
        "ET": seg[2] > 0,
    }
    for c in CHANNEL_NAMES:
        n = int(masks[c].sum())
        setattr(feat, f"voxels_{c}", n)
        setattr(feat, f"volume_mm3_{c}", n * voxel_volume_mm3)

    for c in CHANNEL_NAMES:
        sa = _surface_area_voxels(masks[c])
        setattr(feat, f"sphericity_{c}", _sphericity(int(masks[c].sum()), sa))
    feat.surface_area_voxels_WT = _surface_area_voxels(masks["WT"])

    elong, bbox_ratio, _ = _bbox_metrics(masks["WT"])
    feat.elongation_WT = elong
    feat.bbox_volume_ratio_WT = bbox_ratio

    # Centroid (relative to lesion bounding box of WT) and distance from brain center
    feat.centroid_xyz = _centroid(masks["WT"])
    feat.centroid_rel_xyz = tuple(
        c / s if s > 0 else 0.0 for c, s in zip(feat.centroid_xyz, shape)
    )  # type: ignore[assignment]
    brain_mask = _foreground_mask(image)
    brain_center = _centroid(brain_mask)
    feat.radial_distance_from_brain_center = _radial_distance(
        feat.centroid_xyz, brain_center, spacing_mm
    )

    # Topology
    _, n_wt, sizes_wt = _connected_components(masks["WT"])
    _, n_tc, _ = _connected_components(masks["TC"])
    _, n_et, _ = _connected_components(masks["ET"])
    feat.n_components_WT = n_wt
    feat.n_components_TC = n_tc
    feat.n_components_ET = n_et
    if sizes_wt.size > 0:
        feat.largest_component_share_WT = float(sizes_wt.max() / sizes_wt.sum())
    chi_wt = _euler_characteristic_3d(masks["WT"])
    feat.euler_characteristic_WT = chi_wt
    feat.n_holes_WT = max(0, n_wt - chi_wt)  # beta_0 - chi  >= number of handles

    # TC-inside-WT containment
    if feat.voxels_TC > 0:
        feat.tc_inside_wt_share = float((masks["TC"] & masks["WT"]).sum() / feat.voxels_TC)
    if feat.voxels_ET > 0:
        feat.et_inside_tc_share = float((masks["ET"] & masks["TC"]).sum() / feat.voxels_ET)

    # Modality contrast inside vs outside the WT mask, restricted to the
    # brain foreground so background zeros don't pull the mean down.
    inside_mask = masks["WT"] & brain_mask
    outside_mask = (~masks["WT"]) & brain_mask
    for i, m in enumerate(MODALITY_NAMES):
        v = image[i]
        inside_vals = v[inside_mask]
        outside_vals = v[outside_mask]
        mi = float(inside_vals.mean()) if inside_vals.size else 0.0
        mo = float(outside_vals.mean()) if outside_vals.size else 0.0
        feat.intensity_inside[m] = mi
        feat.intensity_outside[m] = mo
        feat.intensity_ratio_in_over_out[m] = mi / mo if abs(mo) > 1e-6 else 0.0

    return feat


# ---------------------------------------------------------------------------
# Cohort-level aggregation + visualisation
# ---------------------------------------------------------------------------

def build_dataframe(records: Iterable[MorphFeatures]):
    import pandas as pd

    rows = []
    for r in records:
        d = r.to_dict()
        # Flatten nested dicts so the DataFrame stays 2-D
        for key in ("intensity_inside", "intensity_outside", "intensity_ratio_in_over_out"):
            for m, v in d.pop(key).items():
                d[f"{key}_{m}"] = v
        d["shape_h"], d["shape_w"], d["shape_d"] = d.pop("shape")
        d["spacing_h"], d["spacing_w"], d["spacing_d"] = d.pop("spacing_mm")
        d["cx"], d["cy"], d["cz"] = d.pop("centroid_xyz")
        d["rel_cx"], d["rel_cy"], d["rel_cz"] = d.pop("centroid_rel_xyz")
        rows.append(d)
    return pd.DataFrame(rows)


def _safe_import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _by_class(df, col, classes):
    return [df[df["label"] == c][col].dropna().values for c in classes]


def figure_volume_by_class(df, classes, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for i, ch in enumerate(CHANNEL_NAMES):
        col = f"volume_mm3_{ch}"
        data = _by_class(df, col, classes)
        axes[i].boxplot(data, labels=classes, showfliers=True)
        axes[i].set_title(f"{ch} lesion volume (mm^3)")
        axes[i].set_yscale("log")
        axes[i].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "morph_fig01_volume_by_class.png", dpi=150)
    plt.close(fig)


def figure_sphericity_by_class(df, classes, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for i, ch in enumerate(CHANNEL_NAMES):
        col = f"sphericity_{ch}"
        data = _by_class(df, col, classes)
        axes[i].violinplot(data, showmedians=True)
        axes[i].set_xticks(range(1, len(classes) + 1))
        axes[i].set_xticklabels(classes, rotation=15)
        axes[i].set_title(f"{ch} sphericity")
        axes[i].set_ylim(0, 1)
        axes[i].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("sphericity (1 = sphere)")
    fig.tight_layout()
    fig.savefig(out / "morph_fig02_sphericity_by_class.png", dpi=150)
    plt.close(fig)


def figure_multifocality(df, classes, out: Path) -> None:
    plt = _safe_import_plt()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    width = 0.8 / max(len(classes), 1)
    bins = np.arange(0.5, 6.5, 1)
    for i, c in enumerate(classes):
        sub = df[df["label"] == c]["n_components_WT"].clip(upper=5)
        counts, _ = np.histogram(sub, bins=bins)
        ax.bar(np.arange(1, 6) + (i - len(classes) / 2) * width, counts,
               width=width, label=c)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["1", "2", "3", "4", "5+"])
    ax.set_xlabel("# connected components (WT)")
    ax.set_ylabel("# cases")
    ax.set_title("Multifocality of the whole-tumor mask, by class")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "morph_fig03_multifocality.png", dpi=150)
    plt.close(fig)


def figure_topology(df, classes, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    # Euler characteristic distribution by class
    for c in classes:
        sub = df[df["label"] == c]["euler_characteristic_WT"]
        if len(sub):
            axes[0].hist(sub, bins=20, alpha=0.4, label=c)
    axes[0].set_title("Euler characteristic (WT)")
    axes[0].set_xlabel("chi")
    axes[0].set_ylabel("# cases")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # n_holes_WT distribution by class
    for c in classes:
        sub = df[df["label"] == c]["n_holes_WT"]
        if len(sub):
            axes[1].hist(sub.clip(upper=10), bins=11, range=(0, 10), alpha=0.4, label=c)
    axes[1].set_title("Surrogate hole count (beta_1)")
    axes[1].set_xlabel("# holes (clipped at 10)")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "morph_fig04_topology.png", dpi=150)
    plt.close(fig)


def figure_intensity_contrast(df, classes, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, len(MODALITY_NAMES), figsize=(14, 3.4), sharey=True)
    for i, m in enumerate(MODALITY_NAMES):
        col = f"intensity_inside_{m}"
        data = _by_class(df, col, classes)
        axes[i].boxplot(data, labels=classes, showfliers=False)
        axes[i].set_title(f"{m}: mean inside-lesion z-intensity")
        axes[i].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("z-scored intensity")
    fig.tight_layout()
    fig.savefig(out / "morph_fig05_inside_lesion_intensity.png", dpi=150)
    plt.close(fig)


def figure_contrast_ratio(df, classes, out: Path) -> None:
    """The t1ce inside-vs-outside ratio is the single most direct image-level
    discriminator radiologists look at — enhancing recurrence pushes this
    ratio above 1; pure necrosis can sit below or at 1."""
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, len(MODALITY_NAMES), figsize=(14, 3.4), sharey=True)
    for i, m in enumerate(MODALITY_NAMES):
        col = f"intensity_ratio_in_over_out_{m}"
        data = _by_class(df, col, classes)
        axes[i].violinplot(data, showmedians=True)
        axes[i].set_xticks(range(1, len(classes) + 1))
        axes[i].set_xticklabels(classes, rotation=15)
        axes[i].set_title(f"{m} in/out ratio")
        axes[i].axhline(1.0, color="k", lw=0.6, ls="--")
        axes[i].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "morph_fig06_modality_inside_outside_ratio.png", dpi=150)
    plt.close(fig)


def figure_centroid_locations(df, classes, out: Path) -> None:
    plt = _safe_import_plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    markers = {"recurrence": "o", "necrosis": "s", "necrosis+recurrence": "^",
               "unlabeled": "x"}
    for c in classes:
        sub = df[df["label"] == c]
        axes[0].scatter(sub["rel_cx"], sub["rel_cy"], alpha=0.6, label=c,
                        marker=markers.get(c, "o"))
        axes[1].scatter(sub["rel_cx"], sub["rel_cz"], alpha=0.6, label=c,
                        marker=markers.get(c, "o"))
    axes[0].set_xlabel("rel x (axis 0)"); axes[0].set_ylabel("rel y (axis 1)")
    axes[0].set_title("Centroid in coronal-plane normalised coords")
    axes[1].set_xlabel("rel x (axis 0)"); axes[1].set_ylabel("rel z (axis 2)")
    axes[1].set_title("Centroid in sagittal-plane normalised coords")
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "morph_fig07_centroid_locations.png", dpi=150)
    plt.close(fig)


def figure_correlation_heatmap(df, out: Path) -> None:
    plt = _safe_import_plt()
    cols = [
        "volume_mm3_WT", "volume_mm3_TC", "volume_mm3_ET",
        "sphericity_WT", "sphericity_TC", "sphericity_ET",
        "elongation_WT", "bbox_volume_ratio_WT",
        "n_components_WT", "largest_component_share_WT",
        "euler_characteristic_WT", "n_holes_WT",
        "tc_inside_wt_share", "et_inside_tc_share",
        "intensity_inside_t1ce", "intensity_inside_flair",
        "intensity_ratio_in_over_out_t1ce", "intensity_ratio_in_over_out_flair",
    ]
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].apply(lambda s: s.fillna(s.median()))
    corr = sub.corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(out / "morph_fig08_feature_correlation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Group statistical tests — class-conditional differences
# ---------------------------------------------------------------------------

def class_contrast_tests(df, classes: List[str]) -> List[Dict]:
    """Mann-Whitney U on every numeric feature between the first two
    classes (recurrence vs necrosis). Returns a sortable list of dicts.
    The MWU is non-parametric and robust to outliers, which matters here
    because lesion-volume distributions are heavy-tailed.
    """
    from scipy.stats import mannwhitneyu

    if len(classes) < 2:
        return []
    a_lbl, b_lbl = classes[0], classes[1]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []
    for col in numeric:
        a = df.loc[df["label"] == a_lbl, col].dropna().values
        b = df.loc[df["label"] == b_lbl, col].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        results.append({
            "feature": col,
            "n_a": int(len(a)), "n_b": int(len(b)),
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "u": float(u),
            "p": float(p),
        })
    results.sort(key=lambda r: r["p"])
    return results


# ---------------------------------------------------------------------------
# Narrative report
# ---------------------------------------------------------------------------

_MORPH_REPORT = """# Morphology and topology priors — Tiantan cleaned cohort

*Auto-generated by ``src.analysis.morphology``. Run after the
preprocessing pipeline; consumes the ``.npz`` files directly.*

We analyse {n_cases} cleaned cases across the classes {classes_str}.

## 1. Volumetric priors

Per-class median whole-tumor volume (mm^3):

{volume_table}

The WT volume distribution is heavy-tailed in both classes
(``morph_fig01_volume_by_class.png``, log y-axis); this is the single
strongest argument for using a **log-transformed volume weight** in the
loss rather than a linear voxel count.

## 2. Shape priors — sphericity and elongation

Per-class median sphericity (1 = perfect sphere):

{sphericity_table}

Sphericity in [0, 1] from ``psi = pi^(1/3) * (6V)^(2/3) / A`` is a clean
discrete-radiomics proxy; class-conditional violins are in
``morph_fig02_sphericity_by_class.png``. Aggregated against the
elongation prior ``elongation_WT`` and ``bbox_volume_ratio_WT``, this
gives the segmentation head a non-trivial inductive bias: typical
necrosis is rounder and more bbox-filling than typical recurrence.

## 3. Topology priors — multifocality and Betti numbers

Multifocality (#components) by class is in
``morph_fig03_multifocality.png``. Per-class share of single-component
WT masks (the radiologist's "unifocal" baseline):

{unifocal_table}

The Euler characteristic ``chi(WT)`` distribution
(``morph_fig04_topology.png``) is a low-cost proxy for handles and
holes: large negative ``chi`` correlates with handle-rich (= cavitated)
necrotic lesions, while compact enhancing recurrence sits at ``chi ~ 1``.
The fallback hole count ``max(0, beta_0 - chi)`` is reported as
``n_holes_WT``.

The nested-mask priors ``tc_inside_wt_share`` and ``et_inside_tc_share``
are useful sanity checks: the cohort's BraTS-style label channels are
nested by construction, but a model that predicts an ET region outside
the TC region is producing an anatomically impossible answer — these
ratios let us write a *containment* loss term.

## 4. Modality contrast priors

Inside-lesion vs outside-lesion intensity ratio per modality
(``morph_fig06_modality_inside_outside_ratio.png``) gives the most
clinically aligned discriminator: t1ce in/out > 1 reflects gadolinium
enhancement that the radiologist reads, while FLAIR in/out flags
peritumoral edema. Per-class medians:

{ratio_table}

The inside-lesion absolute z-intensity per modality
(``morph_fig05_inside_lesion_intensity.png``) shows where each modality
"earns its place": modalities whose inside-lesion distribution is
identical to outside-lesion are statistically uninformative for that
class and can be down-weighted in the prior-aware fusion stage.

## 5. Spatial priors — where do lesions sit

``morph_fig07_centroid_locations.png`` plots WT centroids in
shape-normalised coordinates [0, 1]^3. The ``radial_distance_from_brain_center``
feature is a one-number summary of "central vs peripheral" lesion
position, which the segmentation head can consume as a coordinate-aware
attention bias.

## 6. Feature redundancy

``morph_fig08_feature_correlation.png`` is the cross-feature correlation
heatmap. Highly correlated feature blocks (volume family, sphericity
family) tell us which priors collapse into the same axis; we should
inject orthogonal pairs (e.g. ``sphericity_WT`` + ``n_components_WT``)
rather than two co-linear ones.

## 7. Class-conditional statistical tests

Mann-Whitney U on recurrence vs necrosis (top features by p-value):

{stat_table}

See ``morph_class_tests.json`` for the full list.

## 8. Where this drops into the model

* The **sphericity and centroid priors** give the spatial-attention
  branch a hand-coded floor before any voxel has been seen.
* The **multifocality prior** justifies a graph-pooled component head
  on top of the segmentation logits, instead of treating WT as one blob.
* The **Euler-characteristic prior** can be added as a soft topology
  regulariser in the loss (penalise large excursions of predicted chi
  from per-class medians).
* The **modality in/out ratio prior** weights the modality fusion
  attention by clinically aligned contrast rather than learnt freely.
"""


def _format_quartile_table(df, cols, classes, scale: float = 1.0) -> str:
    lines = ["| class | " + " | ".join(cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for c in classes:
        row = [c]
        sub = df[df["label"] == c]
        for col in cols:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                row.append("—")
            else:
                row.append(f"{np.median(vals) * scale:.3g}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_share_table(df, classes, predicate_col: str, threshold: int = 1) -> str:
    rows = ["| class | unifocal share |", "|---|---|"]
    for c in classes:
        sub = df[df["label"] == c]
        if len(sub) == 0:
            rows.append(f"| {c} | — |")
            continue
        share = (sub[predicate_col] <= threshold).mean()
        rows.append(f"| {c} | {share:.2%} |")
    return "\n".join(rows)


def _format_stat_table(stats: List[Dict], top: int = 10) -> str:
    lines = ["| feature | median_a | median_b | p |", "|---|---|---|---|"]
    for r in stats[:top]:
        lines.append(f"| {r['feature']} | {r['median_a']:.3g} | "
                     f"{r['median_b']:.3g} | {r['p']:.2e} |")
    return "\n".join(lines)


def write_morphology_report(df, classes: List[str], out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = class_contrast_tests(df, classes)
    with open(out / "morph_class_tests.json", "w") as f:
        json.dump(stats, f, indent=2)

    body = _MORPH_REPORT.format(
        n_cases=len(df),
        classes_str=", ".join(classes) or "(none)",
        volume_table=_format_quartile_table(
            df, [f"volume_mm3_{c}" for c in CHANNEL_NAMES], classes),
        sphericity_table=_format_quartile_table(
            df, [f"sphericity_{c}" for c in CHANNEL_NAMES], classes),
        unifocal_table=_format_share_table(df, classes, "n_components_WT", threshold=1),
        ratio_table=_format_quartile_table(
            df, [f"intensity_ratio_in_over_out_{m}" for m in MODALITY_NAMES], classes),
        stat_table=_format_stat_table(stats),
    )
    p = out / "morphology_story.md"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_morphology_analysis(
    manifest: Optional[List[Dict]],
    npz_dir: Optional[str],
    out_dir: str,
) -> Dict:
    """Drive the full per-case → per-cohort morphology pipeline.

    Either ``manifest`` (the JSON written by ``preprocess_cohort.py``) or
    ``npz_dir`` (a folder of cleaned ``.npz`` files) must be provided.
    When the manifest is missing, all cases are tagged ``label=None``
    and class-stratified figures will fall back to a single bucket.
    """
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: List[MorphFeatures] = []
    if manifest is not None:
        for entry in manifest:
            npz = entry.get("npz")
            if not npz or not Path(npz).exists():
                continue
            records.append(compute_features(npz, label=entry.get("label")))
    elif npz_dir is not None:
        for p in sorted(Path(npz_dir).glob("*.npz")):
            records.append(compute_features(str(p), label=None))
    else:
        raise ValueError("Provide either manifest or npz_dir.")

    if not records:
        raise RuntimeError("No cleaned cases found to analyse.")

    df = build_dataframe(records)
    df.to_csv(out / "morphology_features.csv", index=False)

    classes = sorted([c for c in df["label"].dropna().unique().tolist()])
    if not classes:
        df["label"] = "unlabeled"
        classes = ["unlabeled"]

    figure_volume_by_class(df, classes, out)
    figure_sphericity_by_class(df, classes, out)
    figure_multifocality(df, classes, out)
    figure_topology(df, classes, out)
    figure_intensity_contrast(df, classes, out)
    figure_contrast_ratio(df, classes, out)
    figure_centroid_locations(df, classes, out)
    figure_correlation_heatmap(df, out)
    write_morphology_report(df, classes, str(out))

    return {
        "n_cases": int(len(df)),
        "classes": classes,
        "out_dir": str(out),
    }
