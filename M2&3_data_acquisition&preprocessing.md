# M2/M3 — Data Acquisition & Preprocessing

INFO 442 · Team 14 · Lanzhou University × ISCAS × Beijing Tiantan Hospital

**Task.** Post-operative discrimination and joint segmentation of **glioma recurrence** vs **radiation necrosis** on follow-up multimodal MRI. Cohort: **234 patients tracked 2012-01 – 2022-12 · 2,537 raw MRI series**.

---

## Deliverable checklist (M2 & M3 rubric)

| # | Required item | Location in this submission |
|---|---|---|
| 1 | **Raw dataset (or reproducible acquisition script)** | §1 — private Tiantan cohort via formal collaboration; raw root at `data/uncleaned_examples/` (de-identified Tiantan samples committed for reproducibility) + `data/数据集/SourceData/` (full private DICOM tree); acquisition script `scripts/preprocess_cohort.py` consumes either layout |
| 2 | **Preprocessing log — every transformation with justification** | [`docs/preprocessing_log.md`](docs/preprocessing_log.md) — 8-stage transformation log with one paragraph per stage |
| 3 | **Cleaned, analysis-ready dataset** | `data/processed/*.npz` (2,396 cases, 15.8 GB) + `data/processed/manifest.json` (machine-readable index) + `data/processed/preprocess_report.json` (kept/dropped audit) |
| 4 | **Data quality summary — null rates, class balance, outlier treatment, schema** | §3 of this document, with figures from `visualization/data_insights/` |

---

## 1. Data acquisition

### 1.1 Source and access

The cohort reaches the team through a **horizontal industry–academia research project** between Associate Professor **Zhongfeng Kang** (Lanzhou University, on-site advisor) and the **Institute of Software, Chinese Academy of Sciences (ISCAS)**, where Professor **Zhulin An** leads the algorithmic side. The clinical origin is **Beijing Tiantan Hospital of Capital Medical University**, where the IRB approved the underlying retrospective study. **All 234 patients in this cohort are Tiantan Hospital patients** (post-radiotherapy follow-up MRI, 2012-01 – 2022-12); the cohort is *not* derived from any public dataset.

Raw imaging stays inside the partner-controlled compute environment; the public-facing artefacts in this repository carry only de-identified data and aggregate statistics. The Tiantan partners chose a **BraTS-2021-compatible file layout** (T1 / T1ce / T2 / FLAIR per case, optionally with a per-voxel segmentation) so the pipeline can reuse mature reading/preprocessing recipes — the *layout* is BraTS-compatible, but the *data* is entirely Tiantan.

Hand-off correspondence is archived in `data_source_comment/` (4 redacted screenshots).

### 1.2 Reproducible acquisition script

Because the raw cohort is private, we ship the acquisition step as a **pipeline that consumes either layout** the partners deliver:

```bash
# Option A — Tiantan NIfTI per-case folders (data/uncleaned_examples/),
# laid out in a BraTS-2021-compatible filename convention
python scripts/preprocess_cohort.py \
    --nifti_root data/uncleaned_examples \
    --out_dir   data/processed

# Option B — Tiantan raw DICOM tree (data/数据集/SourceData/4个常规结构像/)
python scripts/preprocess_cohort.py \
    --structural_root data/数据集/SourceData/4个常规结构像 \
    --out_dir         data/processed

# Defaults: --nifti_root data/uncleaned_examples, --out_dir data/processed
python scripts/preprocess_cohort.py
```

A small number of de-identified Tiantan cases is committed under `data/uncleaned_examples/` so reviewers can re-run the entire pipeline end-to-end. These are **Tiantan patients in BraTS-compatible filenames**, not data drawn from the public BraTS challenge.

---

## 2. Reproducing the pipeline

```bash
# 0. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Cleaning: raw NIfTI/DICOM → cleaned .npz + manifest.json (item 3)
python scripts/preprocess_cohort.py

# 2. Storytelling EDA → visualization/eda/{fig01-09.png, cohort_story.md}
python scripts/eda_storytelling.py

# 3. Missing-modality synthesis → data/processed_synth/
python scripts/synthesize_modalities.py

# 4. Morphology & topology prior mining → visualization/morphology/
python scripts/run_morphology_analysis.py
```

**Loading the cleaned dataset from any downstream notebook / script:**

```python
import json, numpy as np

manifest = json.load(open("data/processed/manifest.json"))           # 2,396 entries
data = np.load("data/processed/001.npz")
image  = data["image"]   # float32 (4, H, W, D) — [t1, t1ce, t2, flair], z-scored
label  = data["label"]   # uint8   (3, H, W, D) — [WT, TC, ET], nested binary
affine = data["affine"]  # float64 (4, 4) — world-coordinate transform
```

---

## 3. Data quality summary

### 3.1 Schema — cleaned, analysis-ready format

Every cleaned case is a single `.npz` archive:

| Array | dtype | shape | semantics |
|---|---|---|---|
| `image` | float32 | `(4, H, W, D)` | channel order **[t1, t1ce, t2, flair]**, foreground z-scored, cropped to union foreground bbox |
| `label` | uint8 | `(3, H, W, D)` | channel order **[WT, TC, ET]** — nested binary masks (WT ⊇ TC ⊇ ET) |
| `affine` | float64 | `(4, 4)` | world-coordinate transform equivalent to the source NIfTI affine |

Per-series storage: **6.6 MB**. Cohort total: **15.8 GB** for 2,396 series. Median crop **(141, 174, 138)** voxels (3.18 × 10⁶ voxels per case). Voxel spacing resampled to **1.0 mm isotropic**.

The manifest entries follow:

```json
{
  "case_id": "001",
  "npz": "data/processed/001.npz",
  "label": "recurrence" | "necrosis" | "necrosis+recurrence",
  "available_modalities": ["t1", "t1ce", "t2", "flair"],
  "missing_modalities":   [],
  "shape": [H, W, D],
  "has_seg": true
}
```

![Cohort dashboard](visualization/data_insights/fig_dashboard.png)

### 3.2 Class balance — structurally 3.5 : 1 imbalanced

![Class distribution](visualization/data_insights/fig_class_distribution.png)

| Class | Patients | Series | Series share |
|---|---|---|---|
| Glioma recurrence | 165 | **1,782** | 74.4% |
| Radiation necrosis | 47 | **508** | 21.2% |
| Recurrence + necrosis (border) | 9 | 106 | 4.4% |

Imbalance ratio recurrence : necrosis = **3.5 : 1**. Addressed downstream via (a) `WeightedRandomSampler` with weights `[1.00, 3.51, 16.81]`, (b) `FocalLoss(α = 0.25, γ = 2.0)`, (c) minority-class PR-AUC + sensitivity / specificity at fixed operating point. **χ²(missing_modality × class) = p = 0.003** → missingness correlates with class (necrosis patients are 1.6× more likely to lack T1ce).

### 3.3 Null rate (missing-modality null rate)

![Modality coverage](visualization/data_insights/fig_modality_coverage.png)

| Coverage class | Series | Share |
|---|---|---|
| All 4 modalities present | **1,869** | **78.0%** |
| Exactly 1 modality missing | 411 | 17.2% |
| Exactly 2 modalities missing | 96 | 4.0% |
| Exactly 3 modalities missing | 20 | 0.8% |
| **≥ 1 modality missing** | **527** | **22.0%** |

| Modality | Series present | Series missing | Null rate |
|---|---|---|---|
| T1 | 2,396 | **0** | **0.0%** |
| T1ce | 2,228 | **168** | **7.0%** |
| T2 | 2,132 | **264** | **11.0%** |
| FLAIR | 2,109 | **287** | **12.0%** |

Other coverage rates:
- **Per-voxel segmentation**: 2,396 / 2,396 = **100.0%**
- **PWI (perfusion-weighted imaging)**: 510 / 2,396 = **21.3%**

### 3.4 Outlier treatment

| Outlier category | Volume | Treatment |
|---|---|---|
| **Invalid patient IDs** (never received radiotherapy) | 13 patients, 141 series | **Hard exclusion** at the first check in `process_one_case`; reason logged as `invalid_id_not_irradiated`; never enter any downstream stage. The list: 047, 107, 214, 225, 311, 350, 354, 358, 374, 462, 463, 475, 481 |
| **Missing-modality cases** | 527 series (22.0%) | **Kept**, with absent channels zero-filled. Downstream code reads `(image[c] == 0).all()` as a hard modality-dropout flag, distinguishing it from a real-background zero (which carries a negative z-value after foreground z-scoring). Training uses modality-dropout augmentation on top of this so the model is robust to inference-time absence. |
| **Shape outliers** (raw shape spans 5 configurations: 240×240×155, 256×256×{20,28}, 512×512×{19,22,24}, 320×260×200, anisotropic 3-D) | 2,396 series | **Resampled** to 1.0 mm isotropic, then **cropped** to the union foreground bbox per case. Output shape varies per case (median 141×174×138). |
| **Intensity outliers** (cross-vendor scanner gain, 94 distinct `SeriesDescription` strings) | 2,396 series | **z-scored using foreground statistics** (voxels > 0) applied to the whole crop — background voxels keep the negative z-value rather than being reset to zero, preserving the "background present" signal as distinct from "modality missing". |
| **Label outliers / non-nested predictions** | 0 cases observed | The cleaning pipeline guarantees `tc_inside_wt_share = et_inside_tc_share = 1.000` in 2,396 / 2,396 series. Predictions are constrained by `NestingPenalty` in training; no post-hoc clipping required. |
| **Topology outliers** (Euler χ extreme tails) | χ ∈ [−487, +52], n_holes ≤ 412 | **Kept** — extreme negative χ is a *signal*, not noise: 314 series with χ ≤ −20 carry the cavitated radiation-necrosis morphology, 528 with χ ≥ +5 carry the compact-recurrence morphology. These outliers feed the `TopologyChiRegulariser` loss. |
| **Volume outliers** (WT spans 1,247 → 1.84 × 10⁶ voxels at 5–95 pct, max 3.84 × 10⁶) | 2,396 series | **Kept**, with the loss reweighted by `LogVolumeWeightedDice` so small lesions (< 1 cm³) are not optimised away. |

### 3.5 Knowledge mined — morphology, topology, modality contrast

![Morphology panel](visualization/data_insights/fig_morphology.png)

| Axis | Number | Interpretation |
|---|---|---|
| WT volume median | **67,420 voxels (67.4 cm³)** | Long-tailed across 4 orders of magnitude. |
| WT sphericity median | **0.362** | **0 of 2,396 series ≥ 0.65** — no tumour is spherical. |
| Single-component WT | **1,485 (62.0%)** | 38.0% (911) multifocal, max 24 components. |
| Euler χ(WT) median | **−3**, range **[−487, +52]** | χ ≤ −20 → 314 cases (cavitated necrosis); χ ≥ +5 → 528 cases (compact recurrence). |
| Label nesting | **100.0%** | TC ⊆ WT and ET ⊆ TC in every single case. |

![Modality contrast](visualization/data_insights/fig_modality_contrast.png)

Modality information density (inside-vs-outside z-intensity separation):
**FLAIR (3.33 σ)** > T2 (2.16 σ) > T1ce (1.39 σ) > T1 (0.23 σ).

The **T1ce in/out ratio** discriminates classes cleanly:
- Recurrence median: **+1.42**
- Necrosis median: **+0.88**
- **Cohen's d = 0.94** (large effect)

Cross-modality synthesis weights match radiology physics: T1ce ← T1 weight 0.72 (RMSE 0.61 z-units); T2 ↔ FLAIR mutually predictive (RMSE 0.68 – 0.78).

### 3.6 Scanner heterogeneity

![Scanner heterogeneity](visualization/data_insights/fig_scanner_heterogeneity.png)

94 distinct SeriesDescription strings, four vendors: **Siemens 42.7% · GE 28.1% · Philips 19.3% · UIH 9.9%**. 89/94 strings recognised by the shipped alias table; 5 require a one-time alias review.

---

## 4. From data quality to model priors

The data findings translate into three architectural priors plus four data-driven loss terms — the M3 modelling layer is fully wired and runs end-to-end (validated on `data/some_cleaned_examples/`).

### 4.1 Three prior modules — `src/models/priors.py`

| Module (plug-in stage) | Insights it encodes | Concrete operations |
|---|---|---|
| **`ModalityCouplingPrior`** (front stem) | Modality coverage 78% / 22% · synthesis weights · FLAIR > T2 > T1ce > T1 ranking | Per-modality CNN stems → clinical coupling matrix initialised T1 ↔ T1ce strong + T2 ↔ FLAIR strong → fusion attention initialised `[0.10, 0.20, 0.30, 0.40]` · honours `missing_mask` so absent channels get zero fusion weight |
| **`TopologyShapePrior`** (bottleneck) | Euler χ class signature · multifocality | Multi-scale morphological gradient (3/5/7) + topology spatial attention + scalar `chi_pred` head used by `TopologyChiRegulariser` |
| **`AnatomySpatialPrior`** (bottleneck) | Centroid lives mid-brain · lesions are non-spherical · future longitudinal extension | Centre-biased spatial mask (×1.5 in central [0.3, 0.7]³, hard-zero in 5-mm skull shell) + one reaction-diffusion step with case-specific (ρ, D) |

### 4.2 Four data-driven loss terms — `src/losses/losses.py`

| Loss | Source insight | Form |
|---|---|---|
| `LogVolumeWeightedDice` | WT volume spans 4 orders of magnitude | `Σᵢ log(1 + Vᵢ) · Dice_i` |
| `NestingPenalty` | TC ⊆ WT, ET ⊆ TC in 100% of cases | `0.1 · (max(0, p_TC − p_WT) + max(0, p_ET − p_TC))` |
| `TopologyChiRegulariser` | E[χ\|recur] = +4, E[χ\|necr] = −24 | `0.05 · smooth_L1(chi_pred / 100, target / 100)` |
| `FocalLoss` | 3.5 : 1 imbalance | `α = 0.25, γ = 2.0` |

### 4.3 Data-pipeline priors

- **Modality dropout augmentation** (Bernoulli p = 0.15, T1 anchored) matches the 22% real-world missingness rate at inference.
- **Stratified-by-(class, n_missing) split by patient ID** prevents multi-timepoint leakage.
- **Anisotropic decoder kernels (3×3×1 + 1×1×3)** replace isotropic 3×3×3 because no tumour is spherical.
- **T1ce in/out ratio + WT volume + sphericity + n_components** fed as a 4-D auxiliary scalar to the classification head — the orthogonal-cluster representatives.

---

## 5. Case studies — every prior, on one patient

The cohort statistics in §3 cover 2,396 series; the model priors in §4 are average effects across that cohort. To make the same priors **visible on a single real patient**, we ship a per-case **5-figure visualisation deck** generated by the [`case_study_visulization/`](case_study_visulization/) package — a top-level repo folder peer to `src/` and `scripts/`.

```bash
# 5 figures × N cases, in <8 s per case on CPU
python -m case_study_visulization.run_case_study --in_dir data/some_cleaned_examples

# Single module on a single case
python -m case_study_visulization.topology --npz data/processed/171.npz
```

Outputs land at `visualization/case_study/<case_id>/{01..05}_*.png`. Below we document each figure: drawing approach, processing steps, and the data-driven insight it surfaces. Figures shown are from **case 001**.

### 5.1 Anatomy panel — three orthogonal views × four modalities

![Case 001 — orthogonal anatomy](visualization/case_study/001/01_anatomy_orthogonal.png)

**Code.** `case_study_visulization/anatomy.py::render`.

**Drawing approach.** 5 × 3 grid: rows = T1 / T1ce / T2 / FLAIR / "T1ce + WT/TC/ET overlay"; columns = axial / coronal / sagittal. Slice indices are taken through the WT centroid. Per-slice contrast is set by the **brain-interior 2nd / 98th percentile** so cross-vendor intensity outliers do not wash out the panel.

**Detailed processing steps.**

1. Load `image` (4, H, W, D) and `label` (3, H, W, D) from the cleaned `.npz`.
2. Compute the brain mask as `image.any(axis=0) != 0`.
3. Find the lesion centroid: argwhere on the WT channel → mean → integer cast (fallback to volume centre if WT is empty).
4. For each (axis, idx) view, slice each modality + the brain mask, then compute `vmin, vmax = np.percentile(slice[brain_slice], [2, 98])`.
5. Detect modality-missing cases by `(image[c] == 0).all()` and stamp a "modality missing" overlay text rather than a noisy near-zero slice.
6. Build the seg overlay as an RGBA stack of `(WT=red, TC=green, ET=blue)` with α = 0.55, painted in nested order so inner channels visually win.

**Insight.** Lets the reader sanity-check the §1 schema (4 image channels, 3 nested label channels, foreground z-scored crop) on a real case. On case 001 you can directly see the T1ce gadolinium enhancement at the lesion rim and the bright FLAIR halo of peritumoral edema — the two clinical cues that drive Findings 7 + 8.

### 5.2 3-D nested-surface rendering — WT ⊇ TC ⊇ ET

![Case 001 — 3-D nested tumour surfaces](visualization/case_study/001/02_tumor_3d_nesting.png)

**Code.** `case_study_visulization/tumor_3d.py::render` (uses `skimage.measure.marching_cubes`).

**Drawing approach.** Top row: WT (translucent red) + TC (green) + ET (blue) all rendered together from 4 viewing angles. Bottom row: each label channel alone, plus a numeric inset showing per-case TC ⊆ WT and ET ⊆ TC shares. Inner channels are drawn last and at higher α so they pop through the outer shell.

**Detailed processing steps.**

1. Optional downsample (default factor 2) — marching cubes scales O(N³).
2. For each label channel ch ∈ {WT, TC, ET}: extract the binary mask; if `mask.sum() < 4`, skip; else call `marching_cubes(mask, level=0.5)` to get `(verts, faces)`.
3. Re-scale verts by the downsample factor so the 4 angles share a common world coordinate.
4. Render each surface as a `Poly3DCollection` with channel-specific colour and α (WT 0.12, TC 0.40, ET 0.85).
5. Auto-fit each 3-D axis to the concatenated bounding box of all surfaces.
6. Compute the nesting shares as `(WT & TC).sum() / TC.sum()` and `(TC & ET).sum() / ET.sum()`.

**Insight.** This is the only figure where the **nesting prior (Finding 6)** is visible *as geometry* rather than as a statistic. Case 001 has WT 57,305 voxels, TC 44,469 (77.6% of WT), ET 32,731 (57.1% of WT), and both nesting shares = 1.000 — i.e. the cleaning pipeline produced a perfectly nested label, ready for `NestingPenalty(weight=0.1)` to be a zero-cost hard constraint.

### 5.3 Topology study — connected components, Euler χ, cavities

![Case 001 — topology study](visualization/case_study/001/03_topology.png)

**Code.** `case_study_visulization/topology.py::render`.

**Drawing approach.** 2 × 3 grid:
**(a)** axial slice with WT components colour-coded by `tab20`;
**(b)** axial slice with internal cavities (β₁ holes) highlighted in `autumn` colormap;
**(c)** per-component voxel histogram on log y-axis;
**(d)** 3-D scatter sample (≤ 4,000 voxels) coloured by component label;
**(e)** numeric panel against the cohort signature χ ≤ −20 (necrosis) vs χ ≥ +5 (recurrence);
**(f)** distance-to-boundary 1-D profile along each axis through the centroid.

**Detailed processing steps.**

1. Connected components: `scipy.ndimage.label(wt, structure=np.ones((3,3,3)))`. Voxel-count per component via `np.bincount(lab.ravel())[1:]`.
2. Euler characteristic: count V (voxels), E (axis-aligned edges), F (axis-aligned faces), C (cubes) on the binary mask; `χ = V − E + F − C`.
3. Surrogate β₁ = `max(0, β₀ − χ)` (β₀ = number of components from step 1).
4. Internal cavities: `binary_fill_holes(wt) & ~wt` — voxels that are background but strictly inside the closed lesion.
5. Distance-to-boundary: `scipy.ndimage.distance_transform_edt(wt)` evaluated along three 1-D lines through the centroid.

**Insight.** Case 001 has **2 components, χ = −1, surrogate β₁ ≈ 1, 0 cavity voxels** — sits very close to the cohort median (`χ_median = −3, n_holes_median = 5`). The 3-D scatter shows the small second component is a satellite lesion in the same lobe rather than a contralateral metastasis. This is the case-level realisation of the cohort's bimodal χ distribution that drives `TopologyChiRegulariser` in §4.2.

### 5.4 Morphology study — sphericity, PCA axes, radial fingerprint

![Case 001 — morphology study](visualization/case_study/001/04_morphology.png)

**Code.** `case_study_visulization/morphology.py::render`.

**Drawing approach.** 2 × 3 grid:
**(a)** WT axial slice with the **equivalent-volume sphere outline** drawn on top — the gap between the dashed circle and the actual mask is the visual translation of `1 − ψ`;
**(b)** sagittal slice with the WT bbox in green (bbox-fill ratio);
**(c)** PCA principal axes as 3 coloured eigen-vectors on a sub-sampled 3-D scatter;
**(d)** surface roughness map on the axial slice — morph gradient evaluated on surface voxels only;
**(e)** numeric panel vs cohort medians;
**(f)** axial **radial-extent polar fingerprint** — max reach of the mask along 24 angles, drawn as a polar plot.

**Detailed processing steps.**

1. Volume `V` = `wt.sum()`. Surface area `A` = sum of axis-wise `np.diff(wt) != 0` edges (a clean discrete proxy).
2. Sphericity ψ = `π^(1/3) · (6V)^(2/3) / A`.
3. Bbox metrics: `argwhere(wt).min(0)` and `.max(0)+1`; `elongation = longest_axis / shortest_axis`; `bbox_fill = V / np.prod(extent)`.
4. PCA axes: centred `argwhere(wt)`, `cov = np.cov(centred.T)`, eigendecompose, pick eigenvalues largest-first.
5. Surface roughness: `grey_dilation(wt, 3) − grey_erosion(wt, 3)` evaluated only on surface voxels (i.e. `wt & ~binary_erosion(wt)`).
6. Polar fingerprint: project all WT voxels onto the axial plane; for 24 wedges of width `π/24` around the centroid, take the max radial distance of any voxel in the wedge.

**Insight.** Case 001's ψ = 0.490 is well below the cohort 95th percentile 0.541 and miles below the spherical threshold 0.65 that **0 of 2,396 cases reach**. The polar fingerprint shows the lesion is elongated along the 270° direction (anterior-posterior in this orientation) and pinched at 90° — exactly the anisotropy that motivates the decoder's **3 × 3 × 1 + 1 × 1 × 3 factorised kernel** in §4.

### 5.5 Modality signature — Findings 7 + 8 on one patient

![Case 001 — modality signature](visualization/case_study/001/05_modality_signature.png)

**Code.** `case_study_visulization/modality_signature.py::render`.

**Drawing approach.** Top row: per-modality axial slice with the WT contour traced in yellow. Bottom row: per-modality intensity histogram — **inside-WT** voxels in red, **outside-WT-inside-brain** voxels in grey, with the **Bhattacharyya distance** Bh = −log Σ √(p·q) printed for each modality. The figure footer prints the case's Bhattacharyya-distance modality ranking, compares the **T1ce in/out ratio** to the cohort recurrence (+1.42) vs necrosis (+0.88) priors, and calls out the closer class.

**Detailed processing steps.**

1. For each modality channel m: extract `inside = image[m][wt & brain]`, `outside = image[m][(~wt) & brain]`.
2. Compute `Δmean = inside.mean() − outside.mean()` and `ratio = inside.mean() / outside.mean()`.
3. Histogram both populations on the same `np.linspace(min, max, 60)` bin grid.
4. Bhattacharyya distance: normalise histograms to probability distributions and compute `−log Σ √(p·q)` (higher = more separated).
5. Footer: rank modalities by Bh descending, find the cohort prior the case's T1ce ratio is closer to (absolute difference).

**Insight.** Case 001's Bhattacharyya ranking is `T1ce(0.61) ≈ T2(0.61) > FLAIR(0.81) > T1(0.44)` and its T1ce in/out ratio is **+1.16**. Cohort priors say **recurrence median = +1.42** and **necrosis median = +0.88**, so the case is closer to **necrosis** by 0.16 z-units. This is the exact computation the auxiliary 4-D scalar input to the classification head encodes in §4.

### 5.6 Why ship five separate figures rather than one big dashboard

Each figure isolates one prior so a clinical reviewer can audit it independently. The 3-D surface figure proves *nesting*; the topology figure proves *χ* and *multifocality*; the morphology figure proves *anisotropy*; the modality figure proves *contrast ranking*. Bundled together they form a single-patient verification deck that mirrors §3 cohort-level statistics and §4 model priors one-to-one — i.e. the priors are not hidden in the loss code but visible on every patient.

---

## 6. Files delivered

```
data/
├── uncleaned_examples/                          # de-identified Tiantan samples committed to repo
├── processed/                                   # cleaned dataset (item 3)
│   ├── <case>.npz                               #   image / label / affine
│   ├── manifest.json                            #   machine-readable index
│   └── preprocess_report.json                   #   kept/dropped audit log
└── processed_synth/                             # synthesis-completed cohort (optional)

docs/
├── preprocessing_log.md                         # item 2 — 8-stage transformation log
└── data_pipeline_report.md                      # extended runbook (v1.1, narrative)

visualization/
├── data_insights/{fig_dashboard, ...}.png       # figures embedded above
├── eda/                                         # storytelling EDA outputs
├── morphology/                                  # morphology / topology features
├── synthesis/                                   # before/after synthesis panels
└── case_study/<case_id>/{01..05}_*.png          # §5 per-case deck (5 figures × case)

scripts/
├── preprocess_cohort.py                         # cleaning pipeline (item 1 + 3)
├── eda_storytelling.py                          # EDA + cohort_story.md
├── synthesize_modalities.py                     # donor-driven synthesis
├── run_morphology_analysis.py                   # morphology / topology mining
└── build_demo_manifest.py                       # build manifest from any .npz folder

src/
├── data/{cleaning, pipeline, raw_io,            # cleaning logic
│        synthesis, modality_map, dataset}.py
├── analysis/{cohort_scan, storytelling,         # EDA + morphology
│            morphology}.py
├── models/{backbone, priors, network}.py        # M3 — 3-D U-Net + 3 priors
├── losses/losses.py                             # M3 — data-driven losses
├── train.py · evaluate.py · inference.py        # M3 — training / eval / infer
└── ...

case_study_visulization/                         # §5 figure pack — peer of src/ and scripts/
├── anatomy.py                                   #   01 — orthogonal anatomy panel
├── tumor_3d.py                                  #   02 — marching-cubes nested surfaces
├── topology.py                                  #   03 — components, χ, cavities
├── morphology.py                                #   04 — sphericity, PCA, polar fingerprint
├── modality_signature.py                        #   05 — per-modality inside/outside histograms
├── helpers.py                                   #   shared utilities
└── run_case_study.py                            #   batch driver

tests/                                           # pytest — 13 passing tests
```

---

## 7. Verification status

- **Pipeline end-to-end**: validated on `data/uncleaned_examples/` (8 de-identified Tiantan cases). All 4 stage scripts run with no CLI arguments and produce the expected outputs.
- **Cleaned output format**: matches `data/some_cleaned_examples/` reference (same `image / label / affine` schema, same z-scoring convention with foreground statistics applied whole-volume).
- **Model + training loop**: validated on `data/some_cleaned_examples/` (2 raw cases inflated to 12 manifest entries). 3-epoch run shows monotonic train + val loss decrease (1.242 → 1.178 train, 1.204 → 1.166 val) and mean Dice improvement 0.198 → 0.264.
- **Unit tests**: `pytest tests/ -q` → **13 passed**.

---
