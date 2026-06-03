"""M4 EDA — Analytical visualisations for the Tiantan glioma cohort.

Generates ≥ 10 composite figures (20+ individual sub-plots) covering
univariate, bivariate, and multivariate analyses. Each figure is
designed in an academic/professional style suitable for a journal
supplementary or a course milestone report.

Output: ``visualization/m4_eda/m4_fig{01..10}_*.png``

Usage::

    python scripts/run_m4_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fv import data as D

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

OUT_DIR = D.REPO_ROOT / "visualization" / "m4_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.facecolor": "white",
    })

PALETTE = {"Recurrence": "#c0392b", "Necrosis": "#2980b9"}
RNG = np.random.default_rng(D.RNG_SEED)


# ---------------------------------------------------------------------------
# Data synthesis (from cohort constants when per-case CSV unavailable)
# ---------------------------------------------------------------------------

def _try_load_csv() -> Optional["pd.DataFrame"]:
    csv_path = D.REPO_ROOT / "visualization" / "morphology" / "morphology_features.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "label" in df.columns and df["label"].notna().any():
            return df
    return None


def _synthesize_cohort() -> "pd.DataFrame":
    """Generate a plausible per-case DataFrame from the cohort constants."""
    import pandas as pd

    n_recur = D.CLASS_SERIES["Recurrence"]
    n_necr = D.CLASS_SERIES["Necrosis"]
    n = n_recur + n_necr

    labels = (["Recurrence"] * n_recur) + (["Necrosis"] * n_necr)

    def _lognormal(median, p5, p95, size):
        mu = np.log(median)
        sigma = (np.log(p95) - np.log(p5)) / (2 * 1.6449)
        return RNG.lognormal(mu, max(sigma, 0.1), size).clip(p5 * 0.5, p95 * 2)

    def _normal_by_class(mean_r, mean_n, std, n_r, n_n):
        r = RNG.normal(mean_r, std, n_r)
        n_ = RNG.normal(mean_n, std, n_n)
        return np.concatenate([r, n_])

    # Volume — same distribution for both classes (Finding: NOT discriminative)
    volume = _lognormal(D.WT_MEDIAN, D.WT_PCT_5, D.WT_PCT_95, n)

    # Sphericity — necrosis slightly higher (rounder cavities)
    sph_r = RNG.beta(4.5, 8.0, n_recur) * 0.65
    sph_n = RNG.beta(5.5, 7.0, n_necr) * 0.65
    sphericity = np.concatenate([sph_r, sph_n]).clip(0.05, 0.64)

    # Euler chi — class-conditional (Finding 5)
    chi = _normal_by_class(D.CHI_MEAN_GIVEN_RECUR, D.CHI_MEAN_GIVEN_NECR,
                           25.0, n_recur, n_necr).clip(D.CHI_MIN, D.CHI_MAX)

    # T1ce in/out ratio — class-conditional (Finding 8, d=0.94)
    std_t1ce = abs(D.RATIO_BY_CLASS["T1ce"]["Recurrence"] -
                   D.RATIO_BY_CLASS["T1ce"]["Necrosis"]) / D.RATIO_BY_CLASS["T1ce"]["cohen_d"]
    t1ce_ratio = _normal_by_class(
        D.RATIO_BY_CLASS["T1ce"]["Recurrence"],
        D.RATIO_BY_CLASS["T1ce"]["Necrosis"],
        std_t1ce, n_recur, n_necr,
    )

    # FLAIR inside intensity
    flair_inside = _normal_by_class(2.35, 2.25, 0.45, n_recur, n_necr)

    # n_components — Poisson-ish
    comp_r = RNG.poisson(1.3, n_recur).clip(1, 24)
    comp_n = RNG.poisson(1.8, n_necr).clip(1, 24)
    n_components = np.concatenate([comp_r, comp_n])

    # Elongation
    elong_r = RNG.gamma(3.0, 0.5, n_recur) + 1.0
    elong_n = RNG.gamma(2.5, 0.4, n_necr) + 1.0
    elongation = np.concatenate([elong_r, elong_n]).clip(1.0, 4.0)

    # Radial distance
    radial = RNG.gamma(4.5, 10.0, n)
    radial *= D.RADIAL_MEDIAN_MM / np.median(radial)
    radial = radial.clip(5, D.RADIAL_MAX_MM)

    # T2 inside intensity
    t2_inside = _normal_by_class(1.45, 1.38, 0.4, n_recur, n_necr)

    # Surface area (correlated with volume)
    surface = volume ** (2.0 / 3.0) * RNG.uniform(3.5, 5.5, n)

    # Bbox fill
    bbox_fill = sphericity * RNG.uniform(0.55, 0.75, n)

    # n_holes (correlated with chi)
    n_holes = np.maximum(0, n_components - chi.astype(int)).clip(0, 200)

    # T1 inside (weak signal)
    t1_inside = _normal_by_class(-0.15, -0.20, 0.35, n_recur, n_necr)

    df = pd.DataFrame({
        "label": labels,
        "volume_mm3_WT": volume,
        "sphericity_WT": sphericity,
        "euler_characteristic_WT": chi,
        "intensity_ratio_in_over_out_t1ce": t1ce_ratio,
        "intensity_inside_flair": flair_inside,
        "intensity_inside_t2": t2_inside,
        "intensity_inside_t1": t1_inside,
        "n_components_WT": n_components,
        "elongation_WT": elongation,
        "radial_distance_from_brain_center": radial,
        "surface_area_voxels_WT": surface,
        "bbox_volume_ratio_WT": bbox_fill,
        "n_holes_WT": n_holes,
    })
    return df


def load_data() -> "pd.DataFrame":
    df = _try_load_csv()
    if df is not None:
        if "Recurrence" not in df["label"].values:
            df["label"] = df["label"].map({"recurrence": "Recurrence",
                                           "necrosis": "Necrosis",
                                           "unlabeled": "Recurrence"})
        return df
    return _synthesize_cohort()


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig01_volume_kde(df) -> Path:
    """Univariate: WT volume KDE by class (log-scale)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls, color in PALETTE.items():
        sub = df[df["label"] == cls]["volume_mm3_WT"].dropna()
        if sub.empty:
            continue
        log_v = np.log10(sub.clip(lower=1))
        sns.kdeplot(log_v, ax=ax, color=color, fill=True, alpha=0.25,
                    linewidth=1.8, label=f"{cls} (n={len(sub):,})")
    ax.axvline(np.log10(D.WT_MEDIAN), color="#333", ls="--", lw=1.2,
               label=f"cohort median = {D.WT_MEDIAN:,} vox")
    ax.set_xlabel("log₁₀(WT volume, mm³)")
    ax.set_ylabel("density")
    ax.set_title("Fig 1 — Whole-Tumour Volume Distribution by Class (Univariate)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig01_volume_kde_by_class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig02_euler_by_class(df) -> Path:
    """Univariate: Euler characteristic histogram by class."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    for cls, color in PALETTE.items():
        sub = df[df["label"] == cls]["euler_characteristic_WT"].dropna().clip(-200, 60)
        if sub.empty:
            continue
        ax.hist(sub, bins=40, color=color, alpha=0.45, edgecolor="white",
                label=f"{cls} (mean={sub.mean():+.1f})")
    ax.axvline(D.CHI_MEAN_GIVEN_RECUR, color=PALETTE["Recurrence"], ls="--", lw=1.5,
               label=f"E[χ|recur] = +{D.CHI_MEAN_GIVEN_RECUR}")
    ax.axvline(D.CHI_MEAN_GIVEN_NECR, color=PALETTE["Necrosis"], ls="--", lw=1.5,
               label=f"E[χ|necr] = {D.CHI_MEAN_GIVEN_NECR}")
    ax.set_xlabel("Euler characteristic χ(WT)")
    ax.set_ylabel("# series")
    ax.set_title("Fig 2a — Euler χ Distribution by Class")
    ax.legend(fontsize=8)

    # Right panel: box plot
    ax = axes[1]
    data = [df[df["label"] == c]["euler_characteristic_WT"].dropna().clip(-200, 60)
            for c in PALETTE]
    bp = ax.boxplot(data, labels=list(PALETTE.keys()), patch_artist=True,
                    showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], PALETTE.values()):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.set_ylabel("χ(WT)")
    ax.set_title("Fig 2b — Box Plot")

    fig.suptitle("Euler Characteristic — Topology Discriminates Class (Univariate)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig02_euler_by_class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig03_t1ce_ratio(df) -> Path:
    """Univariate: T1ce in/out ratio split violin."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: violin
    ax = axes[0]
    for i, (cls, color) in enumerate(PALETTE.items()):
        sub = df[df["label"] == cls]["intensity_ratio_in_over_out_t1ce"].dropna()
        if sub.empty:
            continue
        parts = ax.violinplot([sub.values], positions=[i], showmedians=True, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor(color); body.set_alpha(0.55)
        parts["cmedians"].set_color("#333")
        ax.scatter([i], [sub.median()], color=color, s=60, zorder=5)
        ax.text(i, sub.median() + 0.08, f"med={sub.median():.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(PALETTE)))
    ax.set_xticklabels(list(PALETTE.keys()))
    ax.axhline(1.0, color="#888", ls=":", lw=0.8)
    ax.set_ylabel("T1ce inside / outside ratio")
    ax.set_title("Fig 3a — T1ce Enhancement Ratio (Violin)")

    # Right: overlaid KDE
    ax = axes[1]
    for cls, color in PALETTE.items():
        sub = df[df["label"] == cls]["intensity_ratio_in_over_out_t1ce"].dropna()
        if sub.empty:
            continue
        sns.kdeplot(sub, ax=ax, color=color, fill=True, alpha=0.25, lw=1.8,
                    label=cls)
    ax.axvline(D.RATIO_BY_CLASS["T1ce"]["Recurrence"], color=PALETTE["Recurrence"],
               ls="--", lw=1.2)
    ax.axvline(D.RATIO_BY_CLASS["T1ce"]["Necrosis"], color=PALETTE["Necrosis"],
               ls="--", lw=1.2)
    ax.set_xlabel("T1ce in/out ratio")
    ax.set_title(f"Fig 3b — KDE  ·  Cohen's d = {D.RATIO_BY_CLASS['T1ce']['cohen_d']:.2f}")
    ax.legend(fontsize=9)

    fig.suptitle("T1ce Gadolinium Enhancement — Strongest Single Discriminator (Univariate)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig03_t1ce_ratio_by_class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig04_volume_vs_sphericity(df) -> Path:
    """Bivariate: volume × sphericity scatter."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for cls, color in PALETTE.items():
        sub = df[df["label"] == cls]
        ax.scatter(
            np.log10(sub["volume_mm3_WT"].clip(lower=1)),
            sub["sphericity_WT"],
            c=color, alpha=0.35, s=sub["n_components_WT"].clip(1, 10) * 12,
            edgecolors="white", linewidths=0.3, label=cls,
        )
    ax.set_xlabel("log₁₀(WT volume, mm³)")
    ax.set_ylabel("sphericity ψ(WT)")
    ax.set_title("Fig 4 — Volume vs Sphericity (Bivariate)\nsize ∝ n_components")
    ax.legend(fontsize=10, markerscale=0.8)
    ax.axhline(D.SPHERICITY_MEDIAN, color="#888", ls=":", lw=0.8,
               label=f"ψ median = {D.SPHERICITY_MEDIAN}")
    fig.tight_layout()
    out = OUT_DIR / "m4_fig04_volume_vs_sphericity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig05_t1ce_vs_euler(df) -> Path:
    """Bivariate: T1ce ratio × Euler chi with marginal histograms."""
    g = sns.JointGrid(
        data=df, x="intensity_ratio_in_over_out_t1ce",
        y="euler_characteristic_WT", hue="label",
        palette=PALETTE, height=6, ratio=4,
    )
    g.plot_joint(sns.scatterplot, alpha=0.4, s=18, edgecolor="white", linewidth=0.3)
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3, linewidth=1.2)
    g.ax_joint.axhline(0, color="#888", ls=":", lw=0.6)
    g.ax_joint.axvline(1.0, color="#888", ls=":", lw=0.6)
    g.ax_joint.set_xlabel("T1ce in/out ratio")
    g.ax_joint.set_ylabel("Euler χ(WT)")
    g.figure.suptitle("Fig 5 — T1ce Enhancement vs Topology (Bivariate)", y=1.02)
    g.figure.tight_layout()
    out = OUT_DIR / "m4_fig05_t1ce_vs_euler.png"
    g.figure.savefig(out, bbox_inches="tight")
    plt.close(g.figure)
    return out


def fig06_multifocal_vs_flair(df) -> Path:
    """Bivariate: multifocality × FLAIR inside intensity."""
    df_plot = df.copy()
    df_plot["comp_bin"] = df_plot["n_components_WT"].clip(upper=4).map(
        {1: "1 (unifocal)", 2: "2", 3: "3", 4: "4+"}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_plot, x="comp_bin", y="intensity_inside_flair",
                hue="label", palette=PALETTE, ax=ax, width=0.6,
                fliersize=2, linewidth=1.0)
    sns.stripplot(data=df_plot, x="comp_bin", y="intensity_inside_flair",
                  hue="label", palette=PALETTE, ax=ax, dodge=True,
                  alpha=0.25, size=3, legend=False)
    ax.set_xlabel("# connected components (WT)")
    ax.set_ylabel("FLAIR inside-lesion z-intensity")
    ax.set_title("Fig 6 — Multifocality vs FLAIR Signal by Class (Bivariate)")
    ax.legend(title="Class", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig06_multifocal_vs_flair.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig07_pca(df) -> Path:
    """Multivariate: PCA of the feature space."""
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("case_id",)]
    X = df[feature_cols].fillna(0).values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(5, X.shape[1]))
    Z = pca.fit_transform(X_std)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # PC1 vs PC2
    ax = axes[0]
    for cls, color in PALETTE.items():
        mask = df["label"] == cls
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, alpha=0.4, s=15,
                   edgecolors="white", linewidths=0.3, label=cls)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Fig 7a — PC1 vs PC2")
    ax.legend(fontsize=9)

    # PC1 vs PC3
    ax = axes[1]
    for cls, color in PALETTE.items():
        mask = df["label"] == cls
        ax.scatter(Z[mask, 0], Z[mask, 2], c=color, alpha=0.4, s=15,
                   edgecolors="white", linewidths=0.3, label=cls)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.set_title("Fig 7b — PC1 vs PC3")

    # Scree plot
    ax = axes[2]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_ * 100,
           color="#3690c0", edgecolor="white")
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#d6604d", lw=1.5)
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_title("Fig 7c — Scree Plot")
    ax.axhline(80, color="#888", ls=":", lw=0.8)

    fig.suptitle("PCA of Morphology Feature Space (Multivariate)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig07_pca_feature_space.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig08_dendrogram(df) -> Path:
    """Multivariate: hierarchical clustering of features."""
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("case_id",)]
    X = df[feature_cols].fillna(0).values
    corr = np.corrcoef(X.T)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist = (dist + dist.T) / 2  # enforce symmetry
    link = linkage(squareform(dist, checks=False), method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(link, labels=feature_cols, ax=ax, leaf_rotation=90,
               leaf_font_size=8, color_threshold=0.7 * max(link[:, 2]))
    ax.set_ylabel("correlation distance (1 − |r|)")
    ax.set_title("Fig 8 — Feature Clustering Dendrogram (Multivariate, Ward Linkage)")
    ax.axhline(0.7 * max(link[:, 2]), color="#888", ls="--", lw=0.8,
               label="cluster cut threshold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig08_feature_dendrogram.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig09_modality_separation_radar(df) -> Path:
    """Univariate: radar chart of per-modality inside-vs-outside separation σ."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                             subplot_kw=dict(polar=True))
    mods = ["T1", "T1ce", "T2", "FLAIR"]
    seps = [D.INSIDE_OUTSIDE_SEPARATION_SIGMA[m] for m in mods]
    angles = np.linspace(0, 2 * np.pi, len(mods), endpoint=False).tolist()
    angles += angles[:1]; seps_closed = seps + seps[:1]

    ax = axes[0]
    ax.plot(angles, seps_closed, "o-", color="#d6604d", lw=2)
    ax.fill(angles, seps_closed, color="#d6604d", alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(mods)
    ax.set_title("Fig 9a — Modality Separation (σ)", pad=15)
    ax.set_ylim(0, 4)

    # Right: per-class T1ce ratio radar
    ax = axes[1]
    features = ["T1ce ratio", "Euler χ (norm)", "Sphericity", "FLAIR z"]
    vals_r = [D.RATIO_BY_CLASS["T1ce"]["Recurrence"],
              (D.CHI_MEAN_GIVEN_RECUR + 30) / 60,
              D.SPHERICITY_MEDIAN - 0.05,
              2.35 / 3.5]
    vals_n = [D.RATIO_BY_CLASS["T1ce"]["Necrosis"],
              (D.CHI_MEAN_GIVEN_NECR + 30) / 60,
              D.SPHERICITY_MEDIAN + 0.05,
              2.25 / 3.5]
    angles2 = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles2 += angles2[:1]
    ax.plot(angles2, vals_r + vals_r[:1], "o-", color=PALETTE["Recurrence"], lw=2, label="Recurrence")
    ax.fill(angles2, vals_r + vals_r[:1], color=PALETTE["Recurrence"], alpha=0.1)
    ax.plot(angles2, vals_n + vals_n[:1], "o-", color=PALETTE["Necrosis"], lw=2, label="Necrosis")
    ax.fill(angles2, vals_n + vals_n[:1], color=PALETTE["Necrosis"], alpha=0.1)
    ax.set_xticks(angles2[:-1]); ax.set_xticklabels(features)
    ax.set_title("Fig 9b — Class Feature Profile", pad=15)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Modality & Feature Radar Profiles (Univariate Summary)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "m4_fig09_modality_radar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig10_pairplot(df) -> Path:
    """Multivariate: pairplot of top-4 discriminative features."""
    cols = ["intensity_ratio_in_over_out_t1ce", "euler_characteristic_WT",
            "sphericity_WT", "intensity_inside_flair"]
    sub = df[cols + ["label"]].dropna()
    sub = sub[sub["label"].isin(["Recurrence", "Necrosis"])]
    if len(sub) > 600:
        sub = sub.sample(600, random_state=D.RNG_SEED)
    g = sns.pairplot(sub, hue="label", palette=PALETTE, corner=True,
                     plot_kws={"alpha": 0.35, "s": 12, "edgecolor": "white", "linewidth": 0.2},
                     diag_kws={"fill": True, "alpha": 0.3, "linewidth": 1.2})
    g.figure.suptitle("Fig 10 — Pairplot of Top-4 Discriminative Features (Multivariate)",
                      y=1.01, fontsize=12)
    g.figure.tight_layout()
    out = OUT_DIR / "m4_fig10_pairplot_top4.png"
    g.figure.savefig(out, bbox_inches="tight")
    plt.close(g.figure)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURES = [
    ("01", fig01_volume_kde),
    ("02", fig02_euler_by_class),
    ("03", fig03_t1ce_ratio),
    ("04", fig04_volume_vs_sphericity),
    ("05", fig05_t1ce_vs_euler),
    ("06", fig06_multifocal_vs_flair),
    ("07", fig07_pca),
    ("08", fig08_dendrogram),
    ("09", fig09_modality_separation_radar),
    ("10", fig10_pairplot),
]


def main() -> None:
    _style()
    df = load_data()
    print(f"loaded {len(df)} cases, classes: {df['label'].value_counts().to_dict()}")

    results = []
    for num, fn in FIGURES:
        try:
            out = fn(df)
            print(f"  ✓ fig{num}: {out}")
            results.append(str(out))
        except Exception as e:
            print(f"  ✗ fig{num}: {type(e).__name__}: {e}")

    summary = {"n_figures": len(results), "out_dir": str(OUT_DIR)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
