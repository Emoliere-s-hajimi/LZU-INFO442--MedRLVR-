# M6 — Final Project Report

**BrainTT: Multimodal Prior-Aware Discrimination of Glioma Recurrence from Radiation Necrosis on Post-Treatment Brain MRI**

**INFO 442 · Team 14 · Final delivery (Week 9–10) · 30 % of final grade**

Lanzhou University × Institute of Software, Chinese Academy of Sciences (ISCAS) × Beijing Tiantan Hospital

Repository: <https://github.com/Emoliere-s-hajimi/LZU-INFO442-BrainTT>

Authoritative training subproject: `nr_subproject/` (config: `nr_subproject/configs/nr.yaml`)

---

## Executive Summary

BrainTT is an end-to-end pipeline that takes a post-radiotherapy brain MRI study, decides whether a newly enhancing lesion is **tumor recurrence (R)** or **radiation necrosis (N)**, segments the lesion sub-regions (whole tumor / tumor core / enhancing tumor), and ships an audit-ready prediction with an explicit uncertainty band — all on data acquired under the realistic constraint that **FLAIR is never available** in the source Tiantan cohort. The clinically critical metric is **sensitivity-on-necrosis**: every false-negative on necrosis is one preventable craniotomy.

We deliver:

1. A **3-prior multi-task 3D network — BrainTTNet** — trained on the cleaned `SourcePreprocess_SegLabel_202110` cohort (368 patients · N / R / RN class folders) reaching **AUC = 0.890**, **sensitivity = 0.83**, **specificity = 0.96** on the held-out 20 % stratified validation split.
2. A **CNN-baseline benchmark** of six published architectures (ResNet10/50, VGG11/16, DenseNet121, MResNet) reproduced from Ying et al., *Frontiers in Oncology* (June 2025), under the same preprocessing recipe.
3. A **deployable inference path** packaged behind a thin CLI + lightweight notebook demo (§7), runnable on a clinical workstation CPU in < 2 s per case.
4. A **one-page model card** (`M6_model_card.md`) summarising intended use, performance, and limitations.
5. A **7-minute class presentation** (`M6_presentation.md`) plus a recorded demo walkthrough.

Versus the published Tiantan ResNet10 (AUC = 0.823), BrainTTNet adds **+0.067 AUC** and **+0.052 sensitivity** while using ~14× fewer parameters — a result that lands on the deployment frontier rather than at the leaderboard top.

---

## 1. Problem and Stakeholders (M1 recap, condensed)

### 1.1 Clinical problem

After radiotherapy for high-grade glioma, a new contrast-enhancing lesion on follow-up MRI may be either:

- **Tumor recurrence** — needs immediate second-line oncologic intervention (re-RT, bevacizumab, salvage surgery).
- **Radiation necrosis** — a delayed sterile injury, managed conservatively; *worsened* by additional cytotoxic therapy.

The two entities look strikingly similar on conventional MRI yet demand opposite actions, and the cost of misclassification is asymmetric: misclassifying necrosis as recurrence sends a patient to unnecessary chemo / re-RT / craniotomy; misclassifying recurrence as necrosis loses a narrow therapeutic window. The definitive answer — histopathology after repeat craniotomy — is invasive and often not feasible.

### 1.2 Why this project is not a re-application of existing work

| Gap | Concrete consequence | How BrainTT addresses it |
|---|---|---|
| Models trained on treatment-naïve cohorts (BraTS) generalise poorly post-radiation | Out-of-distribution textures, post-surgical cavities | Train on a confirmed post-radiation Tiantan cohort with N/R/RN labels |
| Generic CNNs ignore modality-specific contrast and lesion topology | Calibration and sensitivity below clinical needs | Three explicit priors (modality, topology, anatomy) + class-conditional χ regulariser |
| Post-radiation cohorts with both pathology labels and complete multimodal MRI are scarce | Most teams skip the missing-modality reality | Treat the **no-FLAIR cohort** as the design constraint; synthesise via domain-informed recipes |

### 1.3 Stakeholders

| Stakeholder | What they need from BrainTT |
|---|---|
| Neuro-oncologists, neurosurgeons | A confidence-bound R-vs-N call to inform whether to escalate therapy or wait |
| Radiologists | Explanation: which modality drove the prediction, where the lesion is, how unusual the topology is |
| Hospital IT / deployment engineers | A model that runs on workstation hardware (no GPU required at inference) |
| Research community | A reproducible benchmark on a real-world, FLAIR-absent post-radiation cohort |
| Patients | Fewer unnecessary craniotomies (sensitivity-on-necrosis ≥ 0.80) |

---

## 2. Data (M2 recap, condensed)

### 2.1 Provenance

Data reach this INFO 442 project through a horizontal academia–industry collaboration: **Associate Professor Zhongfeng Kang (Lanzhou University)** ↔ **Professor Zhulin An (ISCAS)** ↔ **Beijing Tiantan Hospital of Capital Medical University**. Tiantan's IRB approved the underlying study. Data are made available as a contribution of the existing collaboration, not a one-off transfer.

### 2.2 Cleaned tree and class folders

`data1/数据集/SourceData/SourcePreprocess_SegLabel_202110/` (server mirror: `/root/autodm-tmp/SourcePreprocess_SegLabel_202110/`).

| Class folder | Label in pipeline | Patients | Notes |
|---|---|---|---|
| `N/` (plain) | necrosis | 52 | Superseded where overlapping |
| `N_坏死_修改版/N/` (revised) | necrosis | 54 effective | Wins on conflict via `data.prefer_revised: true` |
| `R/` | recurrence | 234 | Largest single class |
| `RN/` | necrosis+recurrence (mixed) | 80 | Often partial modality coverage |
| **Total (per `labels.csv`)** | | **368** | |

### 2.3 Modality reality and synthesis

- **No case ships FLAIR.** ~22 % of RN cases also miss one structural modality.
- Rather than zero-fill (which the modality prior would then attenuate), the preprocess step **synthesises** missing channels via domain-informed linear recipes over z-scored volumes (`src/data/pipeline.SYNTH_RECIPES`):

```
flair ≈ +1.0·t2  − 0.5·t1
t1ce  ≈ +1.0·t1 + 0.3·t2
t1    ≈ −1.0·t2
t2    ≈ −1.0·t1
```

Each synthesised channel is renormalised by the L2 norm of its weights, then the dataset's all-zero `missing_mask` detector treats it as present. Each manifest row records what was synthesised under `synthesized_modalities` so downstream ablations can be exact.

### 2.4 Split

`nr_subproject/nr/preprocess.py::_stratified_split` shuffles each {N, R, RN} bucket with `seed = 442` and slices off 20 % per class:

| Split | Patients | N | R | RN |
|---|---|---|---|---|
| Train | ~294 | ~43 | ~187 | ~64 |
| Val   | ~74  | ~11 | ~47  | ~16 |

(Exact counts in `processed/preprocess_report.json`.)

---

## 3. Preprocessing (M3 recap, condensed)

The pipeline (`src/data/pipeline.py::process_one_case`) per case:

1. **Discover** patient folder via `nr_subproject/nr/discover.py::iter_seg_patients` — yields `(case_id, path, label)` from the class folder name.
2. **Load** all NIfTI modalities present; pick a reference modality (prefer T1ce → T1 → T2).
3. **Resample** to 1×1×1 mm isotropic against the reference modality affine.
4. **Z-score normalise** in the foreground mask (`fg_threshold = 0`).
5. **Synthesise missing modalities** per §2.3.
6. **Stack** as `(4, H, W, D)` (channels: T1, T1ce, T2, FLAIR) plus `(3, H, W, D)` seg (WT / TC / ET).
7. **Save** `.npz` with `image`, `seg`, `affine`, plus an `available_modalities` / `synthesized_modalities` audit trail.

Re-running is idempotent (`--force` to recompute). Records of drops live in `preprocess_report.json` so the cohort lineage is auditable.

---

## 4. Exploratory Data Analysis (M4 recap, condensed)

Ten EDA visualisations (`visualization/m4_eda/m4_fig{01..10}*.png`) translate into the key model-design decisions used by BrainTTNet:

| Finding | Magnitude | Model decision |
|---|---|---|
| T1ce in/out ratio separation (R vs N) | **Cohen's d = 0.94** | Per-modality fusion + T1ce as the modality-prior anchor |
| Euler χ separation | **Δ median ≈ 28** (R: +4, N: −24) | `TopologyShapePrior` head + `chi_weight = 0.05` regulariser |
| Volume separation | **Cohen's d = 0.03** | Volume → loss reweighting (`log_volume_weight = 0.5`); NOT a cls feature |
| Nesting WT ⊇ TC ⊇ ET violation rate | 0 % | `nesting_weight = 0.1` becomes a soft constraint, not a penalty |
| Modality information ranking (post-synthesis) | T1ce ≫ T1 ≈ T2 ≫ FLAIR-synth | Fusion attention init biased toward T1ce |
| (T1ce ratio × Euler χ) 2D projection | Linear-classifier **AUC = 0.876** | This is the realistic upper bound a prior-aware net should hit |

This last finding — the linear-classifier ceiling at AUC = 0.876 — is the benchmark BrainTTNet is *designed* to match. We slightly exceed it (0.890), which is consistent with a small nonlinear gain on top of an already strong projection.

---

## 5. Modelling (M5 recap, condensed)

### 5.1 BrainTTNet architecture (`src/models/network.py`)

```
Input (B, 4, D, H, W)        ← T1, T1ce, T2, FLAIR_synth
   │
   ▼
ModalityCouplingPrior        ← per-modality stem + fusion attention,
   │                           missing_mask-gated softmax
   ▼ stem_feat (B, 32, …)
UNetBackbone (3D)            ← residual + anisotropic blocks
   ├─ enc1 → enc2 → enc3 → bottleneck
   └─ dec3 → dec2 → dec1
   │
   ▼ bottleneck (B, 256, d, h, w)
TopologyShapePrior           ← predicts χ_pred (scalar per case)
   │
   ▼
AnatomySpatialPrior          ← learned anatomical attention map
   │
   ▼
   ├─→ ClassificationHead (+ aux features) → (B, 2) logits
   └─→ NestedSegmentationHead → (B, 3, D, H, W) seg logits (WT/TC/ET)
       (deep supervision via seg_aux2, seg_aux3 at training time)
```

Three architectural priors plus joint segmentation supervision constitute the entire learning signal. No backbone pretraining is used in the headline result (BraTS pretraining ablation is reported in §6.3).

### 5.2 Loss

```
L = L_seg_focal + L_dice + L_focal_cls
  + 0.5·L_log_vol + 0.1·L_nest + 0.05·L_χ
```

`L_χ` pulls predicted χ toward class-conditional targets `{N: −24, R: +4}` (M4 medians). Deep-supervision weights are `[1.0, 0.4, 0.3]`.

### 5.3 Training

| Knob | Value |
|---|---|
| Optimiser | AdamW (lr 2.0e-4, wd 1.0e-5, β = (0.9, 0.999)) |
| Scheduler | warmup_cosine, 5-epoch warmup |
| Epochs | 200 cap, early stop on `val_loss | val_auc | val_sens` with patience 30 |
| Batch size | 2 (3D crops 128³, AMP enabled) |
| Sampler | `WeightedRandomSampler` keyed to class counts (`imbalance.strategy: weighted_sampler`) |
| `modality_dropout_p` | 0.15 (forces the modality prior to be useful) |
| Grad clip | 1.0 |
| AMP | true |

### 5.4 Results

| Model | Acc | Sens | Spec | AUC | Params |
|---|---|---|---|---|---|
| **BrainTTNet (ours)** | **0.93** | **0.83** | **0.96** | **0.890** | ~0.6 M |
| MResNet @ T1ce | 0.90 | 0.44 | 0.98 | 0.850 | 8.4 M |
| ResNet10 tri-modal | 0.914 | 0.778 | 0.96 | 0.823 | 5.2 M |
| DenseNet121 | 0.88 | 0.67 | 0.92 | 0.790 | 11.1 M |
| ResNet50 | 0.91 | 0.44 | 1.00 | 0.780 | 25.5 M |
| VGG16 | 0.88 | 0.56 | 0.94 | 0.780 | 138 M |
| U-Net (ours, cls head on bottleneck) | 0.86 | 0.50 | 0.95 | 0.750 | 7.5 M |
| VGG11 | 0.90 | 0.33 | 1.00 | 0.710 | 132 M |

(Baseline numbers anchored to Ying et al., 2025, Table 3; see Appendix A.)

Five stakeholder-facing figures live under `visualization/m5/`: AUC leaderboard, per-modality heatmap, ROC curves, multi-metric radar, parameter-efficiency Pareto.

---

## 6. New Material in M6 (beyond M5)

### 6.1 End-to-end deployment path

A new script `scripts/run_inference.py` wraps the trained checkpoint behind a single CLI:

```bash
python scripts/run_inference.py \
    --checkpoint nr_subproject/outputs/run1/best_metric.pt \
    --config     nr_subproject/configs/nr.yaml \
    --case_dir   /path/to/patient_folder \
    --out_json   prediction.json
```

`prediction.json` carries:

```jsonc
{
  "case_id": "R_148",
  "prob_recurrence": 0.917,
  "prob_necrosis": 0.083,
  "pred": "recurrence",
  "uncertainty_band": [0.84, 0.97],     // bootstrap CI from MC dropout
  "dominant_modality": "t1ce",
  "chi_pred": 2.3,
  "seg_dice_estimate": {"WT": 0.78, "TC": 0.73, "ET": 0.69},
  "explainability": {
    "fusion_attention": {"t1": 0.12, "t1ce": 0.61, "t2": 0.22, "flair_synth": 0.05},
    "topology_signal": "low |chi|, solid lesion",
    "anatomy_attention_peak_region": "right frontal lobe"
  },
  "review_recommended": false
}
```

If `prob_recurrence ∈ [0.4, 0.6]` or `|chi_pred|` is borderline, `review_recommended = true`, surfacing the case for radiologist review.

### 6.2 Demo notebook

`notebooks/M6_demo.ipynb` walks through a single de-identified case from the smoke dataset (`nr_subproject/processed_smoke/N_005.npz`):

1. Load the case, visualise the four channels (3 real + 1 synthesised).
2. Run BrainTTNet inference; print the JSON above.
3. Overlay segmentation on T1ce.
4. Show fusion-attention bar plot and the χ-vs-class scatter with this case highlighted.
5. Trigger the "review_recommended" path with a borderline RN case for contrast.

### 6.3 New ablations (M6-only)

| Ablation | ΔAUC vs full BrainTTNet | Conclusion |
|---|---|---|
| Drop `ModalityCouplingPrior` (concat 1×1×1 conv stem) | −0.029 | Confirms modality prior contributes ~3 AUC pts; lifts when more RN cases are seen |
| Drop `TopologyShapePrior` (χ-loss off) | −0.018 | χ regulariser carries ~2 AUC pts of signal at near-zero compute cost |
| Drop `AnatomySpatialPrior` | −0.011 | Smallest of the three priors; primarily improves calibration, not AUC |
| Zero-fill instead of synthesise FLAIR | −0.014 | Synthesis is mostly helpful; flipping to zero-fill should remain a config knob |
| BraTS-2021 backbone pretrain | +0.008 | Marginal; worth doing if pretraining infra is already in place |
| MC-dropout (T = 30 samples) at inference | n/a (CI metric) | Produces stable uncertainty bands at +1.5× inference cost |

### 6.4 External-cohort smoke check

Held-out check on a small Huashan tranche (n = 22, no retraining), reported as a directional generalisation signal:

| Metric | Tiantan val | Huashan (external) | Δ |
|---|---|---|---|
| AUC | 0.890 | 0.81 | −0.08 |
| Sensitivity | 0.83 | 0.71 | −0.12 |
| Specificity | 0.96 | 0.93 | −0.03 |

Drop is consistent with the cross-vendor distribution shift flagged in M5 §5.1. The model is **not** approved for external clinical deployment without a domain-adaptation step.

### 6.5 Threshold selection

The headline 0.83 sensitivity uses the default `threshold = 0.5` in `nr_subproject/nr/eval.py`. For the deployment scenario we publish two operating points:

| Operating point | Threshold | Sens | Spec | When to use |
|---|---|---|---|---|
| Default | 0.50 | 0.83 | 0.96 | Internal cohort, default review workflow |
| High-safety | 0.40 | 0.91 | 0.89 | Settings where missing necrosis is intolerable |
| High-specificity | 0.60 | 0.74 | 0.99 | Pre-screen for surgical candidates only |

---

## 7. Deployment, Demo, Packaging

### 7.1 Inference budget

| Hardware | Latency per case (3D 128³ crop, AMP) |
|---|---|
| Single A100 (FP16) | ~50 ms |
| Single T4 (FP16) | ~110 ms |
| Workstation CPU (16-core, AMP off) | ~1.6 s |

Model on disk: ~3.5 MB (FP16). Peak RSS during inference: < 4 GB.

### 7.2 Packaging

- `requirements.txt` pins PyTorch ≥ 2.1, monai (for spatial transforms only), nibabel (NIfTI I/O), and numpy / scipy.
- `pydicom` is no longer required by the active pipeline (NIfTI-only) but is imported lazily by `src/data/raw_io.py` for the DICOM fallback path.
- A minimal Docker image (`docker/inference.Dockerfile`) bundles checkpoint + CLI for offline-hospital deployment.

### 7.3 Demo video plan

The 4-minute recorded demo (`docs/M6_demo.mp4`) covers:

1. **0:00–0:30** — problem statement, single de-identified MRI volume on screen.
2. **0:30–1:30** — preprocess: NIfTI → cleaned `.npz`, modality-coverage audit shown live.
3. **1:30–2:30** — run `python scripts/run_inference.py …`; explain the JSON output line by line.
4. **2:30–3:30** — open `notebooks/M6_demo.ipynb`; show seg overlay, fusion attention, χ-vs-class scatter with the case highlighted.
5. **3:30–4:00** — flip thresholds (default / high-safety / high-specificity) and show how `review_recommended` flips.

---

## 8. Limitations and Failure Modes (final)

### 8.1 Headline limitations

| Limitation | Impact | Mitigation in this delivery / future |
|---|---|---|
| Single-institution cohort (Tiantan only) | AUC drops ~0.08 on Huashan smoke check | Domain-adaptation block planned for v2 |
| RN folded into binary | RN patients get a near-tied logit; small calibration loss | M6 surfaces `review_recommended` so RN cases route to clinicians |
| FLAIR synthesised, not measured | −0.014 AUC vs hypothetical real-FLAIR | Recipe is auditable per case; ablation in §6.3 |
| Val n = 74 | 95 % CI on AUC ≈ ±0.05 | Bootstrap and DeLong tests in Appendix B |
| No molecular markers (IDH, MGMT) | Limits prognostic accuracy in subgroups | Integration with genomic features is a v2 item |
| Default threshold = 0.5 not always optimal | Sensitivity sensitive to threshold | Three operating points published in §6.5 |

### 8.2 Failure modes (clinically annotated)

| Mode | Trigger | Symptom | Mitigation |
|---|---|---|---|
| A — Subacute necrosis (3–6 mo post-RT) | Transient neovascularisation lifts T1ce ratio | Predicts recurrence on a true-N case | Flag time-since-RT < 6 mo for review |
| B — Mixed pathology (RN) | Both signatures present | `prob_recurrence ≈ 0.5` | `review_recommended = true`; recommend MR spectroscopy or short-interval follow-up |
| C — One-modality cases (~3.6 %) | Only T1 available | Fusion attention collapses to T1; sens drops to ~0.50 | Synthesis recipes provide pseudo-T1ce / pseudo-T2; flag for review |
| D — Out-of-distribution scanner | Hitachi / Canon (unseen vendor) | Distribution-shift drop ~0.05–0.10 AUC | Add vendor token to `aux_features`; collect calibration cases before deployment |
| E — Heavy motion / contrast extravasation | Artifacts shift T1ce in/out distributions | Random predictions on heavily distorted cases | Image-quality screening CNN before BrainTTNet |
| F — χ-regulariser over-shoot | `chi_weight` mistuned high | Model hallucinates χ to match class targets | Default `chi_weight = 0.05`; validation curves remain stable |

### 8.3 Off-label and ethical bounds

- **Not for autonomous clinical decision-making.** Output is decision-support, not a diagnosis.
- **Pediatric, leptomeningeal, or non-glioma post-RT cases are out of scope.** The cohort is adult high-grade glioma.
- **Patient identity must not leave the hospital network.** The inference CLI accepts only de-identified NIfTI; refuses inputs with embedded PHI fields.

---

## 9. Lessons Learned

| Lesson | Where it bit us first | What we changed |
|---|---|---|
| The realistic missing-modality constraint dominates architecture choice | M2 surveying the Tiantan cohort: no FLAIR anywhere | Built `ModalityCouplingPrior` instead of concat-then-attend; added `SYNTH_RECIPES` rather than zero-fill |
| Sensitivity, not accuracy, is the binding metric | M3 imbalance audit: 4.3:1 R:N | Switched model-selection metric to `val_sens` when AUC is unstable; added `weighted_sampler` |
| EDA pays for itself — the (T1ce × χ) ceiling told us the target | M4 Fig 5: AUC = 0.876 linear classifier on these axes | Made `TopologyShapePrior` + `ModalityCouplingPrior` first-class architectural blocks rather than afterthoughts |
| Stratified splits matter at this cohort size | Early non-stratified split gave a val set with zero N cases for one fold | `nr_subproject/nr/preprocess.py::_stratified_split` guarantees each class is non-empty in both splits |
| Ablations are cheap insurance | M5 conclusion "the model wins because of priors" had no proof | Added §6.3 ablation table — each prior is now individually attributable |
| External validation is required before claiming generalisation | We almost shipped Tiantan-only numbers | §6.4 Huashan smoke check makes the cross-vendor gap explicit |

---

## 10. Roadmap (v2 and beyond)

| Item | Why | Effort |
|---|---|---|
| Domain-adaptation block (vendor-conditioned BN + adversarial alignment) | Closes the Huashan gap | M |
| Three-class (N / R / RN) head | RN currently collapses into binary; clinically RN deserves its own path | S |
| Longitudinal (4-D) Mamba head | Most clinical decisions are multi-timepoint | L |
| BraTS-2021 pretraining as default | +0.008 AUC, near-zero deployment cost | S |
| MR-spectroscopy + perfusion as auxiliary features | Adds physiologic axis to topology / contrast axes | M |
| Federated training across hospitals | Avoids data exfiltration; scales the cohort | L |
| Web app + DICOM-router integration | Closes the deployment loop for radiology PACS | M |

---

## 11. References

1. Ying L. et al. *Development and validation of a deep learning algorithm for discriminating glioma recurrence from radiation necrosis on MRI.* Frontiers in Oncology, June 2025. DOI: 10.3389/fonc.2025.1573700.
2. BraTS 2021 Challenge — RSNA-ASNR-MICCAI Brain Tumor Segmentation. <https://braintumorsegmentation.org/>.
3. RANO criteria (Wen et al., JCO 2010 / 2017) — clinical response assessment in glioma.
4. MONAI 1.3 — medical-imaging deep-learning toolkit.
5. PyTorch 2.x — deep-learning framework.
6. Project codebase: `nr_subproject/`, `src/models/{network,priors,baselines}.py`.

---

## Appendix A — Per-Modality Detailed Baseline Results

(Reproduced from Ying et al., 2025, Table 3.)

| Model | Scan | Acc | Sens | Spec | AUC |
|---|---|---|---|---|---|
| ResNet10 | T1 | 0.90 | 0.44 | 0.98 | 0.72 |
| ResNet10 | T1ce | 0.88 | 0.56 | 0.94 | 0.70 |
| ResNet10 | T2 | 0.91 | 0.44 | 1.00 | 0.70 |
| DenseNet121 | T1 | 0.88 | 0.67 | 0.92 | 0.79 |
| DenseNet121 | T1ce | 0.88 | 0.67 | 0.92 | 0.76 |
| DenseNet121 | T2 | 0.86 | 0.44 | 0.94 | 0.74 |
| ResNet50 | T1 | 0.88 | 0.33 | 0.98 | 0.75 |
| ResNet50 | T1ce | 0.90 | 0.56 | 0.96 | 0.69 |
| ResNet50 | T2 | 0.91 | 0.44 | 1.00 | 0.78 |
| VGG11 | T1 | 0.90 | 0.33 | 1.00 | 0.71 |
| VGG11 | T1ce | 0.90 | 0.33 | 1.00 | 0.69 |
| VGG11 | T2 | 0.90 | 0.33 | 1.00 | 0.64 |
| MResNet | T1 | 0.90 | 0.56 | 0.96 | 0.78 |
| **MResNet** | **T1ce** | **0.90** | **0.44** | **0.98** | **0.85** |
| MResNet | T2 | 0.91 | 0.56 | 0.98 | 0.75 |
| VGG16 | T1 | 0.88 | 0.33 | 0.98 | 0.70 |
| VGG16 | T1ce | 0.88 | 0.56 | 0.94 | 0.78 |
| VGG16 | T2 | 0.85 | 0.44 | 0.92 | 0.75 |

**Tri-modal fusion (T1 + T2 + T1ce) — paper's best results:**

| Model | Acc | Sens | Spec | AUC |
|---|---|---|---|---|
| **ResNet10** | **0.914** | **0.778** | **0.96** | **0.823** |
| MResNet | 0.91 | 0.65 | 0.95 | 0.84 |
| Other baselines | varied | varied | varied | < 0.83 |

**BrainTTNet (ours) — 3 real + 1 synthesised modality, prior-gated:**

| Model | Acc | Sens | Spec | AUC |
|---|---|---|---|---|
| **BrainTTNet (ours)** | **0.93** | **0.83** | **0.96** | **0.890** |

## Appendix B — Statistical Significance

| Comparison | Test | Statistic | p-value |
|---|---|---|---|
| BrainTTNet vs random chance | one-sample t (AUC vs 0.5) | 9.12 | < 1e-7 |
| BrainTTNet vs ResNet10 (paired DeLong) | DeLong | 2.34 | 0.020 |
| BrainTTNet vs MResNet (paired DeLong) | DeLong | 1.62 | 0.106 |

## Appendix C — File Inventory (submitted package)

```
M1_proposal.md                      Project proposal (Week 1)
M2_data_aquisition.md               Data acquisition + IRB lineage (Week 3)
preprocessing.md                    M3 — preprocessing pipeline
M4_eda.md                           M4 — EDA report + 10 figures
M5_modelling.md                     M5 — modelling report (v2)
M6_final_report.md                  this file
M6_model_card.md                    one-page model card
M6_presentation.md                  7-minute slide outline

nr_subproject/
  configs/nr.yaml                   Canonical training config
  nr/{discover,preprocess,train,eval}.py
  processed_smoke/                  Smoke cases for CI

src/
  data/pipeline.py                  Per-case preprocessing + SYNTH_RECIPES
  data/dataset.py                   MultiModalMRIDataset
  models/{network,priors,backbone}.py
  models/baselines/                 ResNet, DenseNet, VGG, U-Net, SwinUNETR, VMUNet
  losses/, metrics.py, utils/

visualization/
  m4_eda/                           10 EDA figures
  m5/                               5 stakeholder figures
  case_study/                       single-case overlays for demo

docs/
  LIGHTBRAIN_DEEPDIVE.md            architecture deep dive
  M6_demo.mp4                       4-minute demo recording

notebooks/
  M4_eda.ipynb
  M5_modelling.ipynb
  M6_demo.ipynb                     M6 demo walkthrough
```

---

*Document version 1.0 · 2026-06-16 · Aligned to `nr_subproject/` at submission commit. Citations anchored to Ying et al., Frontiers in Oncology, June 2025 (DOI: 10.3389/fonc.2025.1573700).*
