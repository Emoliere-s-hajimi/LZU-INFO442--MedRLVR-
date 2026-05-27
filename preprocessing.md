# `preprocessing.md` — Preprocessing log

**INFO 442 · Team 14 · M2 & M3 (Weeks 3–4)**

Every transformation applied to the raw cohort, in execution order, with the justification for each. The cohort is private patient imaging from **Beijing Tiantan Hospital** (234 patients · 2,537 follow-up MRI series · 2012-01 – 2022-12). The Tiantan partners chose a BraTS-2021-compatible *file layout* (T1 / T1ce / T2 / FLAIR per case + optional segmentation, with filenames such as `BraTS2021_<id>_<mod>.nii.gz`) so the pipeline can reuse mature reading recipes — the **data is Tiantan throughout, not BraTS**.

The pipeline is implemented in `src/data/pipeline.py::process_one_case` (and `_process_dicom_case` for the DICOM variant). Per-case decisions are audited in `data/processed/preprocess_report.json` so any reviewer can trace any case from raw to cleaned. Please read M2&3_data_acquisition&preprocessing.md to find out how to run the data processing codes in detail :)

---

## Reproducing the pipeline

The full chain runs with no CLI arguments — defaults are wired:

```bash
python scripts/preprocess_cohort.py                # Stage 0 → 8 (this document)
python scripts/eda_storytelling.py                 # headers-only EDA
python scripts/synthesize_modalities.py            # post-pipeline Stage 9 (optional)
python scripts/run_morphology_analysis.py          # post-pipeline Stage 10 (optional)
```

To override paths for the full private cohort:

```bash
python scripts/preprocess_cohort.py \
    --nifti_root      data/uncleaned_examples \
    --structural_root data/数据集/SourceData/4个常规结构像 \
    --out_dir         data/processed
```

---

## Stage 0 — Discovery

**What.** Walk the raw root (`data/uncleaned_examples/` for the BraTS-compatible NIfTI layout, or `data/数据集/SourceData/4个常规结构像/` for the per-modality DICOM tree) and emit a list of `(case_id, case_dir, label)` tuples. Each candidate must contain at least one recognised NIfTI / DICOM file.

**How.** `src/data/pipeline.py::discover_nifti_cases` and `discover_dicom_cases`. Modality is identified by the unified alias table in `src/data/modality_map.py` (`classify_filename` for NIfTI, `classify_series_description` for DICOM). The class label is parsed from the parent folder name (`复发` → recurrence, `放坏` → necrosis, `放坏+复发` → border).

**Why.** The Tiantan partners deliver in two physically distinct layouts and four scanner vendors (Siemens 42.7%, GE 28.1%, Philips 19.3%, UIH 9.9%) with 94 distinct `SeriesDescription` strings. A single discovery layer abstracts both layouts so the downstream pipeline does not branch on data origin.

**Effect.** Candidate count and per-case modality availability written to `preprocess_report.json` under `kept` / `dropped`.

---

## Stage 1 — Invalid-ID exclusion

**What.** Hard-drop any case whose `case_id` is in
`INVALID_CASE_IDS = {047, 107, 214, 225, 311, 350, 354, 358, 374, 462, 463, 475, 481}`.

**How.** First check inside `process_one_case`: returns a `DropRecord(reason="invalid_id_not_irradiated")` before any volume is loaded.

**Why.** These 13 patients (141 follow-up series) **never received radiotherapy** — they violate the post-radiation inclusion criterion. The clinical collaborators flagged them in `data/数据集/SourceData/无效病例ID_未参与放射治疗.docx`. Including them would inject pre-treatment lesions into a post-treatment discriminator and bias the recurrence-vs-necrosis decision boundary.

**Effect.** 234 → 221 patients, 2,537 → 2,396 series (**94.4% retained**).

---

## Stage 2 — Per-case modality inventory + reference selection

**What.** Detect which of `{t1, t1ce, t2, flair}` are present for the case, then pick a reference modality by the priority order **T1ce → T1 → T2 → FLAIR**. Missing modalities are recorded but the case is **not** dropped.

**How.** `find_modalities_in_case_dir` walks the case folder; `REFERENCE_PRIORITY` selects the highest-priority available modality.

**Why.** T1ce is the most diagnostically valuable channel (Cohen's d = 0.94 for recurrence vs necrosis on the in/out ratio), so we anchor the resampling grid to T1ce whenever it exists. The fallback chain ensures every case lands on a valid reference even when T1ce is missing in 168 series (7.0% missingness rate).

**Effect.** 1,869 cases (78.0%) have T1ce as reference; the remaining 527 cases (22.0%) fall back to T1 / T2 / FLAIR.

---

## Stage 3 — Volume loading + slice ordering

**What.** Load each available modality into a 3-D float32 array. For NIfTI sources: read the affine and zoom from the NIfTI header. For DICOM sources: assemble the slice stack by sorting on `ImagePositionPatient[2]` → `SliceLocation` → `InstanceNumber`, apply per-slice `RescaleSlope` and `RescaleIntercept`, reconstruct the affine from `ImageOrientationPatient` + first/last `ImagePositionPatient`.

**How.** `src/data/raw_io.py::load_nifti` and `assemble_dicom_series`.

**Why.** DICOM filename ordering is *not* slice ordering — many vendor exporters scramble filenames by transmission order rather than anatomical position. Sorting by `ImagePositionPatient[2]` recovers the true z-axis. Rescale slope/intercept apply the modality LUT so intensities live on the manufacturer's intended scale.

**Effect.** Each modality becomes a `(Z, Y, X)` float32 array with a valid `(4, 4)` affine.

---

## Stage 4 — Cross-modality resampling to reference grid

**What.** For each present modality other than the reference, trilinearly resample the volume to the reference grid via `scipy.ndimage.zoom` with `order=1`. Segmentation masks resample with `order=0` (nearest-neighbour) so labels stay integer-valued.

**How.** `_resample_to_grid` in `src/data/pipeline.py`.

**Why.** Cohort voxel spacing spans **0.43 – 6.00 mm** through-plane and includes 5 distinct shape configurations (240×240×155, 256×256×{20,28}, 512×512×{19,22,24}, 320×260×200, anisotropic 3-D). Without a unified grid, downstream sphericity / Euler χ / cm³ computations would be distorted by slice thickness. The trilinear/nearest split prevents label corruption at boundaries.

**Effect.** All available modalities share the reference grid; segmentation labels remain in their raw integer encoding (typically `{0, 1, 2, 4}` when the Tiantan partners deliver under BraTS label conventions).

---

## Stage 5 — Foreground bounding-box crop

**What.** Compute the union of foreground masks (voxels > 0) across all available modalities, then crop every modality and the segmentation to that bbox.

**How.** `_bbox_union` in `src/data/pipeline.py`; the bbox is the tight axis-aligned bounding box of the union mask.

**Why.** Tiantan volumes arrive skull-stripped and padded out to large isotropic boxes (typical raw shape 240×240×155 voxels) with ~70% background. Cropping to the foreground bbox cuts storage 3–5× and concentrates downstream attention on the brain region. Using the **union** rather than per-modality bbox prevents the missing-modality case from shrinking the brain along an axis where only the reference modality covered it.

**Effect.** Median cropped shape **(141, 174, 138)** voxels = **3.18 × 10⁶ voxels per case** (down from 8.93 × 10⁶ raw).

---

## Stage 6 — Foreground z-score normalisation

**What.** For each cropped modality channel, compute `μ_fg, σ_fg` over voxels strictly above 0, then apply `(x − μ_fg) / σ_fg` to the **whole** crop — including background voxels.

**How.** `_zscore_with_foreground_stats` in `src/data/pipeline.py`.

**Why.** Foreground statistics absorb cross-vendor scanner gain. Applying the transform to the whole crop — instead of zero-resetting background — preserves a critical *signal*: a "present modality background" voxel carries a negative z-value, while a "missing modality" channel stays at exact zero. Downstream code reads `(image[c] == 0).all()` as the unambiguous missing-modality flag (Stage 2 → Stage 6 chain).

**Effect.** Per-channel inside-foreground intensity ≈ N(0, 1); cross-vendor scanner gain absorbed.

---

## Stage 7 — Segmentation label conversion

**What.** Convert the raw integer segmentation labels delivered by the Tiantan partners (typically `{0=background, 1=NCR/NET, 2=ED, 4=ET}` when supplied under BraTS-convention encoding) into the 3-channel nested binary representation:

- Channel 0 — **WT** (Whole Tumor): any label > 0
- Channel 1 — **TC** (Tumor Core): labels {1, 4}
- Channel 2 — **ET** (Enhancing Tumor): label 4

If the raw segmentation is binary (single-mass annotation), all three channels collapse to that mask (WT = TC = ET).

**How.** `_seg_to_brats_channels` in `src/data/pipeline.py`.

**Why.** The downstream model uses sigmoid-per-channel segmentation heads with a `NestingPenalty` loss that enforces `WT ⊇ TC ⊇ ET`. The 3-channel binary layout makes the nesting constraint structural (and the per-channel Dice loss numerically stable on the rare ET class).

**Effect.** `tc_inside_wt_share = et_inside_tc_share = 1.000` across **2,396 / 2,396** cleaned cases.

---

## Stage 8 — Serialisation + manifest

**What.** Save each case as a compressed `.npz` archive with three arrays:

| Array | dtype | shape |
|---|---|---|
| `image` | float32 | `(4, H, W, D)` — channels `[t1, t1ce, t2, flair]` |
| `label` | uint8 | `(3, H, W, D)` — channels `[WT, TC, ET]` |
| `affine` | float64 | `(4, 4)` |

Append a manifest entry to `data/processed/manifest.json` and an audit row to `data/processed/preprocess_report.json`.

**How.** `np.savez_compressed` + `write_report` in `src/data/pipeline.py`.

**Why.** A single self-describing archive per case keeps the downstream training loop simple (`np.load(path)` is sufficient) and lets shape and dtype contracts be enforced at the loader boundary. The manifest is the single source of truth for which cases entered the cohort; the report is the audit trail for those that did not.

**Effect.** Cleaned cohort delivered as **2,396 × 6.6 MB = 15.8 GB**, indexed by a 2,396-entry manifest.

---

## Post-pipeline transformations (optional)

These stages run *after* the cleaning pipeline but constitute documented transformations of the analysis-ready dataset.

### Stage 9 — Missing-modality synthesis (`scripts/synthesize_modalities.py`)

For each case with absent channels, fit a per-class ridge regression `missing_modality ← observed_modalities` on the **complete-case donor pool** (1,869 cases with all 4 modalities) and use the regression to fill the missing channel. Three strategies are available: `atlas_mean`, `linear_regression` (default), `patch_knn`.

Synthesis RMSE on held-out donors:
- T1ce ← T1 weight 0.72 → **RMSE 0.61 z-units**
- T2 ← FLAIR weight 0.57 → **RMSE 0.78 z-units**
- FLAIR ← T2 weight 0.39 → **RMSE 0.68 z-units**

Output: `data/processed_synth/<case>.npz` with all 4 channels populated. Use case: trains the model under modality-dropout augmentation with high-fidelity pseudo-labels.

### Stage 10 — Morphology / topology feature extraction (`scripts/run_morphology_analysis.py`)

Compute per-case features from the cleaned `.npz`: volume, sphericity, elongation, bbox fill, connected-component count, Euler characteristic, surrogate hole count (β₁), label nesting shares, modality contrast inside-vs-outside z-intensity, centroid coordinates.

Output: `visualization/morphology/morphology_features.csv`. Use case: scalar priors fed to the classification head and class-conditional χ targets fed to the topology loss.

---

## Stage 11 — Per-case visual verification (case studies)

**What.** For every cleaned case, render a **5-figure verification deck** under `visualization/case_study/<case_id>/` that demonstrates the cleaning pipeline produced the expected output. The deck is the case-level twin of the cohort-level EDA — same priors, but viewed on one patient at a time.

**How.** The [`case_study_visulization/`](case_study_visulization/) package (a top-level repo folder, peer to `src/` and `scripts/`). Five modules + a runner:

| Module | Output PNG | What it draws | Source modules used |
|---|---|---|---|
| `anatomy.py` | `01_anatomy_orthogonal.png` | 3 orthogonal views × 4 modalities + seg overlay row | matplotlib only |
| `tumor_3d.py` | `02_tumor_3d_nesting.png` | marching-cubes surfaces of WT / TC / ET | `skimage.measure.marching_cubes` + `mpl_toolkits.mplot3d` |
| `topology.py` | `03_topology.png` | components, cavities, Euler χ, distance transform | `scipy.ndimage.label`, `binary_fill_holes`, `distance_transform_edt` |
| `morphology.py` | `04_morphology.png` | sphericity, PCA axes, surface roughness, polar shape fingerprint | `scipy.ndimage` (grey morphology, eigendecomposition), polar plot |
| `modality_signature.py` | `05_modality_signature.png` | per-modality inside/outside histograms + Bhattacharyya distance | `numpy.histogram` |
| `run_case_study.py` | `case_study_summary.json` | batch driver over `--in_dir` of `.npz` files | — |

Run:

```bash
# Re-emit the entire deck for every reference case
python -m case_study_visulization.run_case_study --in_dir data/some_cleaned_examples

# One module only — useful while iterating
python -m case_study_visulization.topology --npz data/processed/171.npz
```

**Why.** Each cleaning stage above is justified at the *cohort* level by a statistic (e.g. "78.0% of series have all 4 modalities"). The case-study deck shows that a *given* case after running through Stages 0–8 actually exhibits the expected behaviour: the schema is right (anatomy), the nesting is preserved (tumor_3d), the topology features are computable (topology), the shape features match the cohort distribution (morphology), and the modality contrast follows the expected ranking (modality_signature).

**Effect.** 5 PNGs (≈ 2.4 MB) per case, ≈ 8 s of CPU per case. Concrete realisations:

| Stage / Finding under verification | Case-study figure that confirms it |
|---|---|
| Stage 7 — segmentation label conversion (nesting) | `02_tumor_3d_nesting.png` (TC ⊆ WT share + ET ⊆ TC share both = 1.000) |
| Stage 6 — foreground z-score (background ≠ missing) | `01_anatomy_orthogonal.png` (background renders as dark gray, missing modality renders as a "modality missing" placeholder) |
| Finding 5 — Euler χ class signature | `03_topology.png` (per-case χ printed and compared to cohort thresholds) |
| Finding 3 — non-spherical lesions | `04_morphology.png` (equivalent-sphere outline visibly mismatches the lesion) |
| Finding 7 + 8 — modality information ranking | `05_modality_signature.png` (per-case Bhattacharyya ranking + T1ce ratio call-out) |

---

## Pipeline-stage summary table

| Stage | Transformation | Justification (one line) | Code reference |
|---|---|---|---|
| 0 | Discovery | Abstract two raw layouts behind a single iterator | `discover_nifti_cases`, `discover_dicom_cases` |
| 1 | Invalid-ID exclusion | 13 patients never irradiated → violate inclusion criterion | `INVALID_CASE_IDS` in `pipeline.py` |
| 2 | Modality inventory + reference selection | Anchor grid to clinically most-valuable modality (T1ce) | `REFERENCE_PRIORITY` |
| 3 | Volume load + slice ordering | DICOM filename ≠ slice order; need ImagePositionPatient | `raw_io.assemble_dicom_series` |
| 4 | Cross-modality resampling | Voxel spacing 0.43–6.00 mm — unify before geometry features | `_resample_to_grid` |
| 5 | Foreground bbox crop | ~70% background; union bbox preserves missing-modality cases | `_bbox_union` |
| 6 | Foreground z-score | Absorb scanner gain; keep missing channel ≡ exact zero | `_zscore_with_foreground_stats` |
| 7 | Segmentation label conversion | Nested binary enables structural WT⊇TC⊇ET constraint | `_seg_to_brats_channels` |
| 8 | Serialisation + manifest | Self-describing archive per case + auditable index | `np.savez_compressed`, `write_report` |
| 9 | Missing-modality synthesis (optional) | Pseudo-labels for modality-dropout training | `scripts/synthesize_modalities.py` |
| 10 | Morphology features (optional) | Scalar priors for the classification head + χ loss | `scripts/run_morphology_analysis.py` |
| 11 | Case-study visual verification (optional) | Per-case 5-figure deck proving each prior on one patient | `case_study_visulization/` |

---

*`src/data/pipeline.py` is the canonical source — this document mirrors its behaviour but the code is the ground truth.*
