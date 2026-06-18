# Model Card — BrainTTNet v1.0

**Task:** Discriminate glioma **recurrence (R)** from **radiation necrosis (N)** on post-radiotherapy brain MRI · multi-task with WT/TC/ET segmentation.
**Owners:** INFO 442 Team 14 (Lanzhou University · ISCAS · Beijing Tiantan Hospital). **Release date:** 2026-06-16. **License:** academic / research, non-commercial.

---

### Model details

- **Architecture:** 3-D multi-task network with three explicit priors over a U-Net backbone (`src/models/network.py::BrainTTNet`).
  - `ModalityCouplingPrior` — per-modality stem + missing-aware fusion attention.
  - `TopologyShapePrior` — bottleneck head predicting Euler characteristic χ, supervised toward class-conditional targets {N: −24, R: +4}.
  - `AnatomySpatialPrior` — learned anatomical attention map.
- **Inputs:** 3-D volume `(4, D, H, W)` — channels T1, T1ce, T2, FLAIR; per-case `missing_mask`; optional `aux_features` (age, time-since-RT, scanner-vendor, prior-RT-dose).
- **Outputs:** classification logits `(B, 2)` → `prob_recurrence`; segmentation logits `(B, 3, D, H, W)` for WT / TC / ET; auxiliary `chi_pred` and per-modality `fusion_attention`.
- **Parameter count:** ~0.6 M (`base_channels=32`). On-disk FP16: ~3.5 MB.
- **Training data:** `SourcePreprocess_SegLabel_202110` — 368 patients (234 R · 54 N · 80 RN), stratified 80 / 20 train / val on seed 442. FLAIR is universally absent and synthesised via `src/data/pipeline.SYNTH_RECIPES`.
- **Training recipe:** AdamW (lr 2e-4, wd 1e-5), warmup-cosine, 200-epoch cap with patience 30, AMP, weighted sampler, modality-dropout p = 0.15, composite loss `L_seg_focal + L_dice + L_focal_cls + 0.5·L_log_vol + 0.1·L_nest + 0.05·L_χ`.

### Intended use

- **In scope:** decision-support for adult high-grade glioma patients with a new contrast-enhancing lesion on post-RT MRI; multidisciplinary tumor-board review; research benchmarking on the Tiantan cohort.
- **Out of scope:** autonomous diagnosis; pediatric, leptomeningeal, or non-glioma post-RT cases; pre-treatment baseline studies; any setting where PHI cannot be removed before inference.
- **Primary users:** neuro-radiologists, neuro-oncologists, neurosurgical clinicians; secondary users: medical-imaging researchers.

### Performance

Held-out 20 % stratified val split (n ≈ 74; seed 442), threshold = 0.5.

| Metric | Value | 95 % CI (DeLong) |
|---|---|---|
| AUC | **0.890** | [0.83, 0.94] |
| Accuracy | 0.93 | [0.85, 0.98] |
| Sensitivity (necrosis) | 0.83 | [0.66, 0.95] |
| Specificity | 0.96 | [0.86, 0.99] |
| F1 (necrosis) | 0.81 | — |
| Dice mean (WT / TC / ET) | 0.78 / 0.73 / 0.69 | — |

Operating points published for deployment (see M6 §6.5):

| Mode | Threshold | Sens | Spec |
|---|---|---|---|
| Default | 0.50 | 0.83 | 0.96 |
| High-safety | 0.40 | 0.91 | 0.89 |
| High-specificity | 0.60 | 0.74 | 0.99 |

**External smoke check** (Huashan, n = 22, no retraining): AUC 0.81 · sens 0.71 · spec 0.93 — directional only; **not** validated for cross-institutional deployment.

### Limitations and failure modes

- **Single-institution training data** — cross-vendor distribution shift drops AUC by ~0.08 on the external smoke check.
- **FLAIR is synthesised, not measured** — ΔAUC ≈ −0.014 if zero-fill is used instead (auditable per case via manifest's `synthesized_modalities`).
- **RN (mixed) class is currently absorbed into binary** — RN patients get a near-tied logit. M6 surfaces `review_recommended = true` whenever `prob_recurrence ∈ [0.4, 0.6]`.
- **Subacute necrosis (3–6 months post-RT)** can present with transient neovascularisation that mimics recurrence — flag for review.
- **One-modality cases (~3.6 % of cohort)** see fusion attention collapse onto a single channel; sensitivity drops to ~0.50.
- **Heavy motion artifacts or contrast extravasation** push inputs out of distribution; pair with an image-quality screening step.
- **No molecular markers (IDH / MGMT)** — limits prognostic accuracy in subgroups.
- **Val set is small (n ≈ 74)** — confidence intervals overlap with the next-best baseline; significance vs MResNet does not reach p < 0.05.

### Ethical considerations and safe-use guidance

- Output is **decision-support, not a diagnosis**. Final clinical decisions must remain with the treating clinician.
- The CLI refuses inputs with embedded PHI fields; only de-identified NIfTI is accepted.
- The model is not approved for autonomous clinical decision-making or for direct patient-facing use.
- Re-training or fine-tuning on new institutional data must repeat the stratification by N / R / RN to avoid leakage, and must declare any added modalities in the manifest.

### Reproduction

```bash
python -m nr_subproject.nr.preprocess --config nr_subproject/configs/nr.yaml
python -m nr_subproject.nr.train     --config nr_subproject/configs/nr.yaml
python -m nr_subproject.nr.eval      --config nr_subproject/configs/nr.yaml \
    --checkpoint nr_subproject/outputs/run1/best_metric.pt
```

### Contact

INFO 442 Team 14 · repository: <https://github.com/Emoliere-s-hajimi/LZU-INFO442-BrainTT> · issues: file at the same repo.

---

*Model card version 1.0 · 2026-06-16 · Conforms to Mitchell et al. (2019) "Model Cards for Model Reporting" structure.*
