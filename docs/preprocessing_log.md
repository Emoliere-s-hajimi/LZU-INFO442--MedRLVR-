# Preprocessing Log — BrainTT M2 & M3

Every transformation applied to the raw **Tiantan Hospital cohort** (234 patients · 2,537 follow-up MRI series · 2012-01 – 2022-12), listed in execution order, with the justification for each. The data is private hospital imaging; the file layout is BraTS-2021-compatible (a convention adopted by the partners), but the imaging itself is **not** drawn from any public BraTS challenge.

The cleaning pipeline lives in `src/data/pipeline.py::process_one_case` (and `_process_dicom_case` for the DICOM variant); per-case decisions are audited in `data/processed/preprocess_report.json`.

---

## Stage 0 — Discovery

**What.** Walk the raw root (`data/uncleaned_examples/` or `data/数据集/SourceData/4个常规结构像/`) and emit a list of `(case_id, case_dir, label)` tuples. Each candidate must contain at least one recognised NIfTI / DICOM file.

**How.** `src/data/pipeline.py::discover_nifti_cases` and `discover_dicom_cases`. Modality is identified by the unified alias table in `src/data/modality_map.py` (`classify_filename` for NIfTI, `classify_series_description` for DICOM); the class label is parsed from the parent folder name (`复发` / `放坏` / `放坏+复发` → `recurrence` / `necrosis` / `necrosis+recurrence`).

**Why.** The cohort arrives in two physically distinct layouts and four scanner vendors (Siemens 42.7%, GE 28.1%, Philips 19.3%, UIH 9.9%) with 94 distinct `SeriesDescription` strings. A single discovery layer abstracts both layouts so the downstream pipeline does not branch on data origin.

**Output.** Candidate count and per-case modality availability written to `preprocess_report.json` under `kept` / `dropped`.

---

## Stage 1 — Invalid-ID exclusion (outlier treatment)

**What.** Hard-drop any case whose `case_id` is in `INVALID_CASE_IDS = {047, 107, 214, 225, 311, 350, 354, 358, 374, 462, 463, 475, 481}`.

**How.** First check inside `process_one_case`: returns a `DropRecord(reason="invalid_id_not_irradiated")` before any volume is loaded.

**Why.** These 13 patients (141 follow-up series) **never received radiotherapy** — they violate the study's post-radiation inclusion criterion, as flagged by the clinical collaborators in `data/数据集/SourceData/无效病例ID_未参与放射治疗.docx`. Including them would inject pre-treatment lesions into a post-treatment discriminator and bias the recurrence-vs-necrosis decision boundary.

**Effect.** 234 → 221 patients, 2,537 → 2,396 series (94.4% retained).

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

**Why.** DICOM filename ordering is not slice ordering — many vendor exporters scramble file names by transmission order rather than by anatomical position. Sorting by `ImagePositionPatient[2]` recovers the true z-axis. Rescale slope/intercept apply the modality LUT so intensities live on the manufacturer's intended scale.

**Effect.** Each modality becomes a `(Z, Y, X)` float32 array with a valid `(4, 4)` affine.

---

## Stage 4 — Cross-modality resampling to reference grid

**What.** For each present modality other than the reference, trilinearly resample the volume to the reference grid via `scipy.ndimage.zoom` with `order=1`. Segmentation masks resample with `order=0` (nearest-neighbour) so labels stay integer-valued.

**How.** `_resample_to_grid` in `src/data/pipeline.py`.

**Why.** The cohort's voxel spacing spans **0.43 – 6.00 mm** through-plane and includes 5 distinct shape configurations (240×240×155, 256×256×{20,28}, 512×512×{19,22,24}, 320×260×200, anisotropic 3-D). Without a unified grid, sphericity / Euler χ / cm³ computations downstream would be distorted by slice thickness. The trilinear/nearest split prevents label corruption at boundaries.

**Effect.** All available modalities share the reference grid; segmentation labels remain in their raw integer encoding (e.g. `{0, 1, 2, 4}` when the Tiantan partners deliver the segmentation under BraTS label conventions).

---

## Stage 5 — Foreground bounding-box crop

**What.** Compute the union of foreground masks (voxels > 0) across all available modalities, then crop every modality and the segmentation to that bbox.

**How.** `_bbox_union` in `src/data/pipeline.py`; the bbox is the tight axis-aligned bounding box of the union mask.

**Why.** Tiantan volumes arrive skull-stripped and padded out to large isotropic boxes (typical raw shape 240×240×155 voxels) with ~70% background. Cropping to the foreground bbox cuts storage 3–5× and concentrates downstream attention on the brain region. Using the **union** rather than per-modality bbox prevents the missing-modality case from shrinking the brain along an axis where only the reference modality covered it.

**Effect.** Median cropped shape **(141, 174, 138)** voxels = 3.18 × 10⁶ voxels per case (vs 8.93 × 10⁶ raw).

---

## Stage 6 — Foreground z-score normalisation

**What.** For each cropped modality channel, compute `μ_fg, σ_fg` over voxels strictly above 0, then apply `(x − μ_fg) / σ_fg` to the **whole** crop — including background voxels.

**How.** `_zscore_with_foreground_stats` in `src/data/pipeline.py`.

**Why.** Foreground statistics give a stable per-case scale that absorbs cross-vendor scanner gain (the cohort spans Siemens, GE, Philips, UIH). Applying the transform to the whole crop — instead of zero-resetting background — preserves a critical *signal*: a "present modality background" voxel carries a negative z-value, while a "missing modality" channel stays at exact zero. Downstream code reads `(image[c] == 0).all()` as the unambiguous missing-modality flag (Stage 2 → Stage 6 chain).

**Effect.** Per-channel inside-foreground intensity ≈ N(0, 1); cross-vendor scanner gain absorbed.

---

## Stage 7 — Segmentation label conversion

**What.** Convert the raw integer segmentation labels delivered by the Tiantan partners (`{0=background, 1=NCR/NET, 2=ED, 4=ET}` when supplied under BraTS-convention encoding) into the 3-channel nested binary representation:

- Channel 0 — **WT** (Whole Tumor): any label > 0
- Channel 1 — **TC** (Tumor Core): labels {1, 4}
- Channel 2 — **ET** (Enhancing Tumor): label 4

If the raw segmentation is binary (single mass annotation), all three channels collapse to that mask (so WT = TC = ET).

**How.** `_seg_to_brats_channels` in `src/data/pipeline.py`.

**Why.** The downstream model uses sigmoid-per-channel segmentation heads with a `NestingPenalty` loss that enforces `WT ⊇ TC ⊇ ET`. The 3-channel binary layout makes the nesting constraint structural (and the per-channel Dice loss numerically stable on the rare ET class).

**Effect.** `tc_inside_wt_share = et_inside_tc_share = 1.000` across **2,396 / 2,396** cleaned cases.

---

## Stage 8 — Serialisation + manifest

**What.** Save the case as a compressed `.npz` with three arrays:

| Array | dtype | shape |
|---|---|---|
| `image` | float32 | `(4, H, W, D)` — channel order `[t1, t1ce, t2, flair]` |
| `label` | uint8 | `(3, H, W, D)` — channel order `[WT, TC, ET]` |
| `affine` | float64 | `(4, 4)` |

Append a manifest entry to `data/processed/manifest.json` and a kept-cases / dropped-cases audit row to `data/processed/preprocess_report.json`.

**How.** `np.savez_compressed` + `write_report` in `src/data/pipeline.py`.

**Why.** A single self-describing archive per case keeps the downstream training loop simple (`np.load(path)` is sufficient) and lets shape and dtype contracts be enforced at the loader boundary. The manifest is the single source of truth for which cases entered the cohort; the report is the audit trail for those that did not.

**Effect.** Cohort delivered as **2,396 × 6.6 MB = 15.8 GB**, indexed by a 2,396-entry manifest.

---

## Post-pipeline transformations (optional)

These stages are downstream of the cleaning pipeline but constitute documented transformations of the analysis-ready dataset.

### Stage 9 — Missing-modality synthesis (`scripts/synthesize_modalities.py`)

For each case with absent channels, fit a per-class ridge regression `missing_modality ← observed_modalities` on the **complete-case donor pool** (1,869 cases with all 4 modalities) and use the regression to fill the missing channel. Three strategies: `atlas_mean` (donor-pool average), `linear_regression` (default), `patch_knn`. Output: `data/processed_synth/<case>.npz` with all 4 channels populated. Use case: trains the model under modality-dropout augmentation with high-fidelity pseudo-labels.

### Stage 10 — Morphology / topology feature extraction (`scripts/run_morphology_analysis.py`)

Compute per-case features from the cleaned `.npz`: volume, sphericity, elongation, bbox fill, connected-component count, Euler characteristic, surrogate hole count (β₁), label nesting shares, modality contrast inside-vs-outside z-intensity, centroid coordinates. Output: `visualization/morphology/morphology_features.csv`. Use case: scalar priors fed to the classification head and class-conditional χ targets fed to the topology loss.

---
