# Distinguishing Glioma Recurrence from Radiation Necrosis on Post-Treatment Brain MRI

*INFO 442 — Team 14 · Project BrainTT*
- **Team Leader: Yutong Wang (ID:320230942461)** — Google Scholar: [https://scholar.google.com/citations?hl=en&authuser=1&user=73MjwF0AAAAJ](https://scholar.google.com/citations?hl=en&authuser=1&user=73MjwF0AAAAJ) Personal Profile: [https://emoilere.github.io/](https://emoilere.github.io/)
- **Team Members: Zijin Wu(ID:320230942571), Xiaopeng Fan(ID:320230941801), Ye Wang(ID:320230942541), Yunfei Shang(ID:320230942271)**

This is a data-science project carried out in collaboration with the **Institute of Software, Chinese Academy of Sciences (ISCAS)** and **Beijing Tiantan Hospital**, advised by **Prof. Zhulin An (ISCAS) and Prof. Zhongfeng Kang (Lanzhou University)**. **The team has been granted access to a private post-radiation brain-tumor MRI cohort that is not publicly available**, and our goal is to build a clinically useful decision-support pipeline on top of it. 

**Note——updated 5.12.2026: The dataset was collected and labbled by Tiantan Hospital, Zhulin An and Zhongfeng Kang, and it is totally  sourced ethically, with all private information about patients removed.**

**Note:——updated 5.19.2026: The dataset is too large that even uploading 3 cases is also impossible, so we upload a case with full modalities(case1) and a case with only 2 modalities(case2), and their processed versions in /some_cleaned_examples_small directory**

---

## 1 · Clinical motivation and project value

After radiotherapy for high-grade glioma, follow-up MRI frequently reveals new contrast-enhancing lesions. Two very different conditions can produce visually similar images:

- **Tumor recurrence** — the disease is back; the patient typically needs immediate second-line oncologic treatment.
- **Radiation necrosis (RN)** — a delayed, sterile injury caused by the radiation itself; the standard management is conservative, anti-inflammatory, and explicitly *not* further anti-tumor therapy.

These two outcomes look alike on conventional MRI but require *opposite* clinical actions. Misclassification leads either to unnecessary chemotherapy / re-irradiation, or to a missed window for treating an aggressive recurrence. Histopathological confirmation requires a repeat craniotomy, which is invasive and not always safe. A reliable non-invasive discriminator therefore has direct, measurable clinical value, and is exactly the gap our industrial partners have asked us to address.

This project takes us through the full data-science life-cycle required by INFO 442:

1. **Data cleaning** of a heterogeneous, real-world private cohort.
2. **Exploratory analysis** of class balance, modality coverage, and lesion morphology.
3. **Visualization** of the multi-modal MRI volumes and downstream model outputs.
4. **Modeling** with a multimodal, medical-prior-aware deep network.
5. **Evaluation** against clinically relevant metrics (sensitivity, specificity, AUC, Dice).

---

## 2 · The data

### 2.1 Source

The cohort is shared with us directly by Prof. An's group and originates from Beijing Tiantan Hospital. Below is the original message exchange documenting the hand-off (Chinese with auto-translated English; redacted for privacy).

<p align="center">
  <img src="data_source_comment/1.png" width="48%" />
  <img src="data_source_comment/2.png" width="48%" />
</p>
<p align="center">
  <img src="data_source_comment/3.png" width="48%" />
  <img src="data_source_comment/4.png" width="48%" />
</p>

Two clinical aspects from this exchange directly shape our project:

- The professor confirmed that the data format mirrors a **BraTS-2021-style multimodal MRI layout** (T1, T1ce, T2, FLAIR per case), which lets us reuse mature preprocessing recipes.
- The main known caveat is **class imbalance** between positive (recurrence) and negative (radiation-necrosis) samples — handling this is an explicit design requirement, not an afterthought.

### 2.2 Example slices

A representative slice panel from the cohort (axial T2-weighted slices for two cases, ID 148 and 149):

<p align="center">
  <img src="data_example/2dslides_examples.png" width="70%" />
</p>

Each case carries co-registered T1 / T1ce / T2 / FLAIR volumes plus a per-voxel lesion annotation provided by clinical collaborators. The on-disk layout we standardize to looks like:

```
data/processed/
├── manifest.json
└── <case_id>/
    ├── <case_id>_t1.nii.gz
    ├── <case_id>_t1ce.nii.gz
    ├── <case_id>_t2.nii.gz
    ├── <case_id>_flair.nii.gz
    └── <case_id>_seg.nii.gz
```

---

## 3 · Why this project is hard and where we add value

Off-the-shelf brain-tumor models trained on public datasets (BraTS) target newly diagnosed tumors and **do not transfer well** to the post-treatment recurrence-vs-necrosis question, because:

1. The post-radiation appearance of *both* recurrence and RN can mimic an active tumor on conventional MRI — purely image-level texture features are not enough.
2. Public cohorts are dominated by treatment-naïve cases; our population is post-radiation by construction.
3. The radiology priors that actually drive a clinician's judgement (contrast-enhancement patterns across modalities, lesion topology, plausible spatial-temporal evolution) are rarely modelled explicitly in generic segmentation networks.

Our approach therefore plans to inject **explicit medical priors** into a multi-modal deep network so that the model is *guided* by domain knowledge rather than only by raw voxel intensities. We will validate the pipeline on the private Tiantan cohort, with class-imbalance handling as a first-class concern.

The concrete novelties will be locked in during the implementation phase; for the proposal we describe the high-level direction rather than the final design.

---

## 4 · Planned pipeline

```
raw cohort  ──►  cleaning  ──►  EDA + visualization  ──►  modeling  ──►  evaluation
   (NIfTI)     (manifest.json)   (figures, stats)        (.pt)         (metrics.json)
```

What we have so far:

| Stage | Code | What it does |
|---|---|---|
| Cleaning | `src/data/cleaning.py` | Walks the raw dump, harmonises modality naming, drops cases with missing modalities or unknown labels, writes `manifest.json` and a structured drop-report. |
| Bias-field correction | `src/data/bias_correction.py` | N4 inhomogeneity correction (SimpleITK) — important for cross-patient T1ce comparison. |
| Inter-modality registration | `src/data/registration.py` | Rigid registration of each modality to T1ce (Mattes mutual information). |
| Preprocessing | `src/data/preprocessing.py` | Foreground z-score normalisation, isotropic resampling, lesion-centred crop/pad. |
| Augmentation | `src/data/augmentation.py` | Composable transforms: flips, intensity shift, gamma, Gaussian noise. |
| Dataset | `src/data/dataset.py` | Multimodal torch `Dataset` + weighted-sampler dataloader to address class imbalance. |


---

## 5 · Planned repository layout

```
.
├── configs/
│   └── default.yaml              # done, but need revision as the project proceeding
├── data_example/                 # representative MRI slices
├── data_source_comment/          # data hand-off correspondence
├── docs/                         # clinical references provided by collaborators
├── src/
|—— case1                         # done
|—— case2                         # done
|—— some_cleaned_examples         # done
│   ├── analysis/                 # done
│   ├── data/                     # done
│   ├── losses/
│   ├── models/
│   ├── visualization/
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   ├── run_clean.py
│   └── run_eda.py
├── requirements.txt
└── README.md
```

---


### Acknowledgements

We thank **Prof. Zhulin An** and his group at the **Institute of Software, Chinese Academy of Sciences**, clinical collaborators at **Beijing Tiantan Hospital**, for sharing the private post-radiation glioma cohort and for the clinical guidance that shapes this project, and **Prof. Zhongfeng Kang**, **André Catarino**, **Rui**  for carefully guidance.

---

## Data and ethics statement

The cohort used in this project is private patient data covered by a data-use agreement between our partners and the team lead. Raw imaging and any patient-identifiable information are **not** included in this repository; the example images in `data_example/` are de-identified illustrative slices, and the screenshots in `data_source_comment/` document the data hand-off only. All experiments will be carried out locally on approved compute.
