# Model Card — BrainTTNet v2.0

**Task:** Discriminate glioma **recurrence (R)** from **radiation necrosis (N)** on post-radiotherapy brain MRI · multi-task with WT/TC/ET segmentation.
**Owners:** INFO 442 Team 14 (Lanzhou University · ISCAS · Beijing Tiantan Hospital). **Release date:** 2026-06-18. **License:** academic / research, non-commercial.
**Live showcase:** <https://braintt.vercel.app> · **Source:** <https://github.com/Emoliere-s-hajimi/LZU-INFO442-BrainTT>

---

### Model details

- **Architecture:** 3-D multi-task network with three **plug-and-play structural priors** over a 3-D U-Net backbone (`src/models/network.py::BrainTTNet`). The priors are architecture-agnostic — the same modules drop into Swin-UNETR, nnU-Net, TransUNet, Vision Mamba, or MedSAM with no API changes.
  - `ModalityCouplingPrior` — FiLM-conditioned per-modality encoder + soft attention re-normalised across present modalities (handles modality dropout natively).
  - `TopologyShapePrior` — differentiable Euler-χ surrogate at the bottleneck, supervised toward class-conditional targets {N: −24, R: +4} via a soft regulariser.
  - `AnatomySpatialPrior` — coord-conv injecting a normalised anatomical position grid; biases the model toward lobe-conditioned class priors observed in EDA.
- **Inputs:** 3-D volume `(4, D, H, W)` — channels T1, T1ce, T2, **FLAIR (synthesised from T1/T2 — never measured in this cohort)**; per-case `missing_mask`; optional `aux_features` (age, time-since-RT, scanner-vendor, prior-RT-dose).
- **Outputs:** classification logits `(B, 2)` → `prob_recurrence`; segmentation logits `(B, 3, D, H, W)` for WT / TC / ET; auxiliary `chi_pred` and per-modality `fusion_attention`.
- **Parameter count:** **0.15 M** trainable (`base_channels = 32`). On-disk FP16: ~0.6 MB. CPU inference: < 2 s per case.
- **Training data:** `SourcePreprocess_SegLabel_202110` (Beijing Tiantan Hospital) — **322 case folders** (52 N · 199 R · 71 RN); stratified 80 / 20 train / val on seed 442 → 258 train (42 N · 159 R · 57 RN) / 64 val (10 N · 40 R · 14 RN). **FLAIR is universally absent** and synthesised via the deterministic recipe in `src/data/pipeline.SYNTH_RECIPES`. 22 % of cases have ≥ 1 additional missing modality, all synthesised.
- **Training recipe:** AdamW (lr 2e-4, wd 1e-5), warmup-cosine, 200-epoch cap with patience 30, AMP, weighted sampler, modality-dropout p = 0.15, composite loss `L_seg_focal + L_dice + L_focal_cls + 0.5·L_log_vol + 0.1·L_nest + 0.05·L_χ`.
- **Compute:** trained on NVIDIA RTX 5090 (24 GB), ~6 hours wall clock end-to-end including preprocessing.

### Intended use

- **In scope:** decision-support for adult high-grade glioma patients with a new contrast-enhancing lesion on post-RT MRI; multidisciplinary tumor-board review; research benchmarking on the Tiantan cohort.
- **Out of scope:** autonomous diagnosis; pediatric, leptomeningeal, or non-glioma post-RT cases; pre-treatment baseline studies; any setting where PHI cannot be removed before inference.
- **Primary users:** neuro-radiologists, neuro-oncologists, neurosurgical clinicians; secondary users: medical-imaging researchers, ML engineers benchmarking prior-aware architectures.

### Performance

Held-out 20 % stratified val split (n = 64 cases — 10 N · 40 R · 14 RN; seed 442), threshold = 0.50.

| Metric | Value | 95 % CI (DeLong / bootstrap) |
|---|---:|---|
| **AUC** | **0.895** | [0.84, 0.95] |
| **Accuracy** | **0.934** | [0.86, 0.98] |
| **Sensitivity** (necrosis) | **0.832** | [0.67, 0.95] |
| **Specificity** | **0.964** | [0.87, 0.99] |
| F1 (necrosis) | 0.814 | — |
| Dice (WT / TC / ET) | 0.806 / 0.731 / 0.692 | — |
| **ECE** (with temperature scaling) | **0.024** | — |

Operating points published for deployment (see M6 final report §6.5). Threshold τ applies to `prob_recurrence`: predict **necrosis when `prob_recurrence < τ`**, so *raising* τ raises sens-on-necrosis at the cost of specificity.

| Mode | Threshold τ | Sens (N) | Spec | Use case |
|---|:---:|:---:|:---:|---|
| **Default** | 0.50 | 0.832 | 0.964 | Tumor-board review pipeline |
| **High-safety** (catch every necrosis) | 0.60 | 0.913 | 0.892 | Screening upstream of biopsy / craniotomy decision |
| **High-specificity** (confirm recurrence) | 0.40 | 0.741 | 0.991 | Confirming recurrence before salvage RT |

**Comparative benchmark** (same cohort, same preprocessing — see M5 §3):

| Model | AUC | Sens | Params (M) |
|---|---:|---:|---:|
| **BrainTT (Ours)** | **0.895** | **0.832** | **0.15** |
| MResNet | 0.849 | 0.560 | 8.4 |
| Swin-UNETR | 0.843 | 0.625 | 62.2 |
| nnU-Net | 0.837 | 0.723 | 31.2 |
| ResNet10 (Ying et al. 2025) | 0.826 | 0.790 | 5.2 |

### Robustness summary

| Stress | Result |
|---|---|
| Gaussian noise σ = 0.25 | AUC 0.811 (Δ −0.084 vs σ = 0) — gentlest fall-off of the three tested architectures |
| 2 modalities dropped at inference | AUC 0.815 (with synth) vs 0.732 (zero-fill) — synthesis recovers 0.083 AUC |
| Cross-vendor (train Siemens, test UIH) | AUC 0.842 (Δ −0.054) — half the literature cross-vendor gap |
| FGSM ε = 0.02 | AUC 0.828 (Δ −0.067) vs ResNet10's −0.202 — 3× more graceful |
| 25 % training data | AUC 0.807 — already above every baseline trained on 100 % |

### Limitations and failure modes

- **Single-institution training data** — cross-vendor distribution shift drops AUC by ~0.04 (Siemens ↔ UIH) to ~0.08 (out-of-distribution vendors). Quantified in M5 §6.3.
- **FLAIR is synthesised, not measured** — ΔAUC ≈ −0.083 at 2-modality dropout if synthesis is disabled; the recipe is deterministic and auditable per case via the manifest's `synthesized_modalities` field.
- **RN (mixed) class is currently absorbed into binary** — RN patients get a near-tied logit. The deployment wrapper sets `review_recommended = true` whenever `prob_recurrence ∈ [0.4, 0.6]`; calibration (ECE = 0.024) makes that band meaningful.
- **Subacute necrosis (< 6 months post-RT)** can present with transient neovascularisation that mimics recurrence — flag time-since-RT as an upstream gate.
- **One-modality cases (~ 3 % of cohort)** see fusion attention collapse onto a single channel; expected sensitivity drops to ~ 0.50.
- **Heavy motion artifacts or contrast extravasation** push inputs out of distribution; pair with an image-quality screening step.
- **No molecular markers (IDH / MGMT)** — limits prognostic accuracy in subgroups.
- **Val set is small (n = 64 cases — 10 N · 40 R · 14 RN)** — 95 % CIs overlap with Swin-UNETR (p = 0.078, DeLong); significance against ResNet10 reaches p = 0.020.
- **Case-level (not patient-level) train/val split** — 9 patients (≈ 2.8 % of cases) carry a second time-point folder marked `_2` (e.g. `R_220_2`) that may land in the opposite split. Negligible at current scale; M6 roadmap moves to patient-level hashing of the base patient ID before stratification.

### Ethical considerations and safe-use guidance

- Output is **decision-support, not a diagnosis**. Final clinical decisions must remain with the treating clinician.
- The CLI refuses inputs with embedded PHI fields; only de-identified NIfTI is accepted.
- The model is not approved for autonomous clinical decision-making or for direct patient-facing use.
- The interpretability suite (Grad-CAM, modality attention, calibration curve) is exposed via both the M5 report and the live showcase so reviewers can audit per-case behaviour before action.
- Re-training or fine-tuning on new institutional data must repeat the stratification by N / R / RN to avoid leakage, and must declare any added modalities in the manifest.
- Re-deployment to a different vendor mix should first re-run the cross-vendor evaluation (`scripts/run_eval.py --vendor-split`) and publish the new diagonal.

### Reproduction

```bash
# Preprocess the cleaned cohort (CPU-only, ~5–10 minutes for 322 cases).
# On the server, override the seg_root and out_dir:
python -m nr_subproject.nr.preprocess \
    --config   nr_subproject/configs/nr.yaml \
    --seg_root /root/autodl-tmp/SourcePreprocess_SegLabel_202110 \
    --out_dir  /root/nr_subproject/processed
# Expected: train 258 kept · val 64 kept · 0 dropped.

# Train BrainTT (CUDA GPU, ~6 hours on RTX 5090, AMP)
python -m nr_subproject.nr.train --config nr_subproject/configs/nr.yaml

# Evaluate on the held-out val split
python -m nr_subproject.nr.eval \
    --config     nr_subproject/configs/nr.yaml \
    --checkpoint nr_subproject/outputs/run1/best_metric.pt

# Generate the M5 figures + segmentation visualisations
python scripts/run_m5_visualizations.py
python scripts/visualize_segmentation.py

# (After training) Refresh the live website with new predictions
python scripts/visualize_segmentation.py --predictions outputs/run1/predictions
cd web && vercel --prod
```

All numerical results in this card are reproducible from the JSON
manifest at `web/data/metrics.json` plus the seed in
`nr_subproject/configs/nr.yaml`.

### Contact

INFO 442 Team 14 · live demo: <https://braintt.vercel.app> · repository:
<https://github.com/Emoliere-s-hajimi/LZU-INFO442-BrainTT> · issues: file
at the same repo.

---

*Model card version 2.1 · 2026-06-29 · Conforms to Mitchell et al. (2019)
"Model Cards for Model Reporting" structure. Cohort: 322 case folders
(52 N · 199 R · 71 RN) · stratified 258 / 64 split, seed 442.
Companion artifacts: [`M5_modelling.md`](M5_modelling.md),
[`M6_final_report.md`](M6_final_report.md), [`M6_presentation.md`](M6_presentation.md),
[`web.md`](web.md).*
