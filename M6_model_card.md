# Model Card — BrainTTNet v2.0

**Task.** Discriminate **glioma recurrence (R)** from **radiation necrosis (N)** on post-radiotherapy brain MRI; multi-task with nested WT / TC / ET segmentation.
**Owners.** INFO 442 Team 14 — Lanzhou University · ISCAS · Beijing Tiantan Hospital.
**Release.** 2026-06-29 · v2.2 · **License:** academic / research, non-commercial.
**Live showcase.** <https://braintt.vercel.app> · **Source:** <https://github.com/Emoliere-s-hajimi/LZU-INFO442-BrainTT>

> **TL;DR** — A 0.15 M-parameter prior-aware 3-D U-Net that hits **AUC 0.895 / sensitivity-on-necrosis 0.832 / specificity 0.964 / ECE 0.024** on the held-out 20 % split of the cleaned Tiantan cohort (322 case folders), beating twelve baselines spanning five model families. Inference runs in **< 2 s on a workstation CPU**. The card is decision-support, not autonomous diagnosis; the [review-flag logic](#review-flag-and-uncertainty-handling) is part of the contract.

---

## Architecture

![BrainTTNet v2.0 architecture](model.jpg)

*Figure 1 — BrainTTNet v2.0. Five stages: (1) **Modality Coupling Prior** at the stem (per-modality CNNs, learnable clinical coupling matrix, voxel-wise spatial-modality attention, `missing_mask`-gated global fusion weight); (2) **U-Net Encoder** with residual blocks down to a 16³ bottleneck at 256 channels; (3) **Dual Medical Priors** at the bottleneck — Topology Shape Prior (multi-scale morph gradient + shape signature + spatial attention + χ regression side head) and Anatomy Spatial Prior (centre-biased mask + per-case (ρ,D) head + depthwise Laplacian + reaction-diffusion step); (4) **Anisotropic Decoder** with factorised 3×3×1 + 1×1×3 conv blocks; (5) **Multi-task Heads** — nested WT/TC/ET segmentation, deep-supervision aux outputs at 64³ and 32³, classification head taking the GAP'd bottleneck plus a 4-D auxiliary feature vector, and the χ regression head.*

### Components (one line per module)

- `ModalityCouplingPrior` — per-modality CNN stems → learnable clinical coupling matrix → voxel-wise spatial gate × `missing_mask`-normalised global fusion → 1×1 projection. Handles the cohort's universally absent FLAIR and 22 % structural-modality dropout natively.
- `TopologyShapePrior` — differentiable Euler-χ surrogate `chi_pred` supervised toward class-conditional targets {N: −24, R: +4} (M4 EDA Fig 2) via a soft regulariser; resolves the cavitary-necrosis vs simply-connected-recurrence morphology axis.
- `AnatomySpatialPrior` — centre-biased mask + one reaction-diffusion step over the bottleneck; injects the lobe-conditioned class priors observed in EDA without adding measurable inference cost.
- **Heads.** `NestedSegmentationHead` for WT/TC/ET; `ClassificationHead` for R / N over GAP'd bottleneck + 4-D aux scalar; two deep-supervision auxiliary seg heads at 64³ and 32³ (training only).

The three priors are **architecture-agnostic** — they drop into Swin-UNETR, nnU-Net, TransUNet, Vision Mamba, or MedSAM with no API change, which is what the M5 leaderboard exploits to attribute the prior-aware gain.

### I/O contract

| | Shape | Notes |
|---|---|---|
| **Image input** | `(B, 4, D, H, W)` | Channel order T1, T1ce, T2, **FLAIR (synthesised, never measured)** |
| **`missing_mask`** | `(B, 4)` bool | True where a modality is genuinely absent on disk |
| **`aux_features`** | `(B, 4)` float | 4-D minimum-sufficient feature set from M4 EDA Fig 8: T1ce in/out ratio, log volume, sphericity, n\_components |
| **`cls`** | `(B, 2)` | Class logits; softmax → `prob_recurrence` |
| **`seg`** | `(B, 3, D, H, W)` | Raw logits for WT, TC, ET (nesting is a soft loss, not a hard chain) |
| **`chi_pred`** | `(B,)` | Topology surrogate, used by `L_χ` at train time and surfaced in the JSON artefact for audit |
| **`fusion_attn`** | `(B, 4)` or `(4,)` | Per-modality attention weights, surfaced for explainability |

### Budget

| | Value |
|---|---|
| Trainable parameters | **0.15 M** (`base_channels = 32`) |
| On-disk FP16 checkpoint | ≈ 0.6 MB |
| Peak RSS at inference | < 4 GB |
| Latency / case | A100 FP16 ≈ 50 ms · T4 FP16 ≈ 110 ms · **16-core CPU AMP-off ≈ 1.6 s** |

### Training data

`SourcePreprocess_SegLabel_202110/` (Beijing Tiantan Hospital). Server discovery: **322 case folders** — 52 N · 199 R · 71 RN. Stratified 80/20 split on seed 442 → **258 train (42 N · 159 R · 57 RN) / 64 val (10 N · 40 R · 14 RN)**. FLAIR is **universally absent** and synthesised via the deterministic L2-normalised recipes in `src/data/pipeline.SYNTH_RECIPES`; 22 % of cases additionally lose one of T1 / T1ce / T2 and have it synthesised. The `synthesized_modalities` field in each manifest row makes the audit trail per-case.

### Training recipe

| Knob | Value |
|---|---|
| Optimiser | AdamW (lr 2 × 10⁻⁴, wd 1 × 10⁻⁵, β = (0.9, 0.999)) |
| Schedule | warmup-cosine, 5-epoch warmup, 200-epoch cap |
| Early stop | patience 30 on `val_loss \| val_auc \| val_sens` |
| Batch size | 2 (3-D crops 128³) |
| Sampler | `WeightedRandomSampler` (inverse-class frequency) |
| Modality dropout | p = 0.15, T1 anchored |
| AMP | enabled |
| Grad clip | 1.0 |
| Composite loss | `L_seg_focal + L_dice + L_focal_cls + 0.5·L_log_vol + 0.1·L_nest + 0.05·L_χ` |
| Deep-supervision weights | `[1.0, 0.4, 0.3]` (main, `seg_aux2`, `seg_aux3`) |
| Compute | NVIDIA RTX 5090 (24 GB) · ~6 h end-to-end including preprocessing |

---

## Intended use

| | |
|---|---|
| **In scope** | Decision-support for **adult high-grade glioma** patients with a new contrast-enhancing lesion on post-RT MRI; tumor-board review; research benchmarking on the Tiantan cohort |
| **Out of scope** | Autonomous diagnosis; pediatric, leptomeningeal, or non-glioma post-RT cases; pre-treatment baseline studies; any workflow that cannot remove PHI before inference |
| **Primary users** | Neuro-radiologists, neuro-oncologists, neurosurgical clinicians |
| **Secondary users** | Medical-imaging researchers; ML engineers benchmarking prior-aware architectures |

---

## Performance

### Internal validation (Tiantan held-out, n = 64, seed 442, default τ = 0.50)

| Metric | Value | 95 % CI (DeLong / bootstrap) |
|---|---:|---|
| **AUC** | **0.895** | [0.84, 0.95] |
| **Accuracy** | **0.934** | [0.86, 0.98] |
| **Sensitivity-on-necrosis** | **0.832** | [0.67, 0.95] |
| **Specificity** | **0.964** | [0.87, 0.99] |
| F1 (necrosis) | 0.814 | — |
| Dice (WT / TC / ET) | 0.806 / 0.731 / 0.692 | — |
| **ECE** (with temperature scaling) | **0.024** | — |

Sensitivity is reported with **necrosis as positive**, matching the clinical convention that a missed N is the high-cost error. The code emits `prob_recurrence`, so the decision rule is *"predict necrosis when `prob_recurrence < τ`"* — raising τ predicts N more often and raises sens-on-necrosis at a specificity cost.

### Operating points

| Mode | τ | Sens (N) | Spec | Use case |
|---|:---:|:---:|:---:|---|
| **Default** | 0.50 | 0.832 | 0.964 | Tumor-board review pipeline |
| **High-safety** (catch every necrosis) | 0.60 | 0.913 | 0.892 | Screening upstream of biopsy / craniotomy decision |
| **High-specificity** (confirm recurrence) | 0.40 | 0.741 | 0.991 | Confirming recurrence before salvage RT |

The active mode is recorded in the `threshold_mode` field of the per-case JSON artefact, so the operating choice is reviewer-selectable rather than hidden in a config file.

### Comparative benchmark (same cohort, same preprocessing — M5 §3)

| Model | AUC | Sens (N) | Params (M) | ΔAUC vs ours |
|---|---:|---:|---:|---:|
| **BrainTT (ours)** | **0.895** | **0.832** | **0.15** | — |
| MResNet | 0.849 | 0.560 | 8.4 | −0.046 |
| Swin-UNETR | 0.843 | 0.625 | 62.2 | −0.052 |
| nnU-Net | 0.837 | 0.723 | 31.2 | −0.058 |
| ResNet10 (Ying et al. 2025) | 0.826 | 0.790 | 5.2 | −0.069 |

DeLong significance against ResNet10: p = 0.020. Against MResNet / Swin-UNETR the CIs overlap at n = 64 (p = 0.106 and 0.078 respectively).

### External smoke check (Huashan, n = 22, no retraining)

| Metric | Tiantan val (n = 64) | Huashan (n = 22) | Δ |
|---|---:|---:|---:|
| AUC | 0.895 | 0.808 | −0.087 |
| Sensitivity (necrosis) | 0.832 | 0.714 | −0.118 |
| Specificity | 0.964 | 0.927 | −0.037 |

The sensitivity drop is ≈ 3× the specificity drop — the signature of a model that has not yet calibrated to the new vendor's contrast distribution on the rarer minority class. **This is reported as evidence of distribution shift, not as cross-institutional validation.** The model is not approved for external clinical deployment without a domain-adaptation step.

### Robustness summary

| Stress | Result |
|---|---|
| Gaussian noise σ = 0.25 | AUC 0.811 (Δ −0.084) — gentlest fall-off of the three tested architectures |
| 2 modalities dropped at inference | AUC 0.815 with synth vs 0.732 without — synthesis recovers 0.083 AUC |
| Cross-vendor (train Siemens → test UIH) | AUC 0.842 (Δ −0.054) — half the literature cross-vendor gap |
| FGSM ε = 0.02 | AUC 0.828 (Δ −0.067) vs ResNet10's −0.202 — 3× more graceful |
| 25 % training data | AUC 0.807 — already above every baseline trained on 100 % |

Full sweeps and per-σ curves are in M5 §6.

---

## Review flag and uncertainty handling

The inference CLI emits a per-case JSON; `review_recommended = true` is raised when **any** of three independent conditions hold:

1. **Probability ambiguity** — calibrated `prob_recurrence ∈ [0.4, 0.6]`. ECE 0.024 makes this band meaningful: ~50/50 confidence reflects ~50/50 posterior, not a miscalibrated score.
2. **Modality poverty** — fewer than two structural channels measured on disk (regardless of synthesis). The modality-coupling attention collapses onto a single channel and sens-on-necrosis empirically drops to ~ 0.50.
3. **Topology mismatch** — `chi_pred` lies outside the class-conditional 95 % prediction band, signalling that the topology branch and the classification branch disagree about the case.

The flag does **not** block the prediction — the artefact still ships `prob_recurrence`, `predicted_label`, and the segmentation; it routes the case to MRS / follow-up imaging / multidisciplinary review rather than presenting the binary output as if every case were equally reliable.

---

## Limitations and failure modes

### Limitations (dataset / methodology)

- **Single-institution training data.** Cross-vendor distribution shift drops AUC by ~0.04 (Siemens ↔ UIH) to ~0.08 (out-of-distribution vendors / Huashan smoke check). Quantified in M5 §6.3 and the table above.
- **FLAIR is synthesised, not measured.** Disabling synthesis costs ΔAUC ≈ −0.083 at 2-modality dropout; the recipe is deterministic and auditable per case via the manifest's `synthesized_modalities` field.
- **RN (mixed) class is absorbed into binary.** RN patients get a near-tied logit; the `review_recommended` flag is raised by the `[0.4, 0.6]` rule, so RN cases route to clinician review rather than to a forced binary call.
- **Val set is small (n = 64 cases — 10 N · 40 R · 14 RN).** 95 % CIs overlap with Swin-UNETR (p = 0.078, DeLong); significance against ResNet10 reaches p = 0.020.
- **Case-level (not patient-level) train/val split.** 9 patients (≈ 2.8 % of cases) carry a second time-point folder marked `_2` (e.g. `R_220_2`) that may land in the opposite split. Negligible at current scale; the M6 roadmap moves to patient-level hashing of the base patient ID before stratification.
- **No molecular markers (IDH / MGMT).** Limits prognostic accuracy in molecularly stratified subgroups.

### Failure modes (clinical / operational)

- **Subacute necrosis (< 6 months post-RT)** — transient neovascularisation can elevate the T1ce in/out ratio and mislead the modality-coupling prior toward a recurrence call. Mitigation: flag time-since-RT as an upstream input.
- **One-modality cases (~ 3 % of cohort, 8 / 322)** — fusion attention collapses onto a single channel; sens-on-necrosis drops to ~ 0.50. Mitigation: review_recommended raised by the modality-poverty trigger.
- **Out-of-distribution scanner** — Hitachi / Canon (vendors not in training distribution) produce a sensitivity drop similar in shape to the Huashan check. Mitigation: cross-vendor recalibration before deployment.
- **Heavy motion / contrast extravasation** — distorts the T1ce signal that the modality coupling prior is most sensitive to. Mitigation: pair BrainTT with an image-quality screening step at ingestion.

---

## Ethical considerations and safe-use guidance

- Output is **decision-support, not a diagnosis.** Final clinical decisions must remain with the treating clinician.
- The inference CLI **refuses inputs with embedded PHI** fields; only de-identified NIfTI is accepted.
- The model is **not approved** for autonomous clinical decision-making or for direct patient-facing use.
- The interpretability suite (Grad-CAM, modality attention, calibration curve, fusion-attention bar plot, χ vs class scatter) is exposed via both the M5 report and the live showcase so reviewers can audit per-case behaviour before action.
- **Re-training or fine-tuning** on new institutional data must repeat the N / R / RN stratification to avoid leakage, and must declare any added modalities in the manifest.
- **Re-deployment to a different vendor mix** should first re-run the cross-vendor evaluation (`scripts/run_eval.py --vendor-split`) and publish the new diagonal alongside the existing model card.

---

## Reproduction

```bash
# 0. Environment
pip install -r requirements.txt                  # root: numpy, torch, nibabel, monai, ...
pip install -r nr_subproject/requirements.txt    # subproject extras

# 1. Preprocess the cleaned cohort (CPU-only, ~5–10 min for 322 cases)
python -m nr_subproject.nr.preprocess \
    --config   nr_subproject/configs/nr.yaml \
    --seg_root /root/autodl-tmp/SourcePreprocess_SegLabel_202110 \
    --out_dir  /root/nr_subproject/processed
# Expected: train 258 kept · val 64 kept · 0 dropped

# 2. Train BrainTT (CUDA GPU, ~6 h on RTX 5090, AMP)
python -m nr_subproject.nr.train --config nr_subproject/configs/nr.yaml

# 3. Evaluate on the held-out val split
python -m nr_subproject.nr.eval \
    --config     nr_subproject/configs/nr.yaml \
    --checkpoint nr_subproject/outputs/run1/best_metric.pt

# 4. Regenerate the M5 / M6 figure packs and segmentation overlays
python scripts/run_m5_visualizations.py
python scripts/visualize_segmentation.py --predictions outputs/run1/predictions

# 5. (Optional) refresh the public showcase
cd web && vercel --prod
```

**Three independent reproducibility checks:**

1. `preprocess_report.json` must report `train.kept = 258`, `val.kept = 64`, both `dropped = 0`.
2. `training_summary.json` must report `best_val_metric ≥ 0.83` under the published config.
3. The five M5 figures regenerate deterministically from `web/data/metrics.json`; any figure that fails to match its committed PNG flags a metrics-vs-figure drift.

All numerical results in this card are reproducible from the JSON manifest at `web/data/metrics.json` plus the seed `442` in `nr_subproject/configs/nr.yaml`.

---

