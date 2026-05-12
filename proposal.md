# M1 — Project Proposal

**BrainTT: Multimodal Prior-Aware Discrimination of Glioma Recurrence from Radiation Necrosis on Post-Treatment Brain MRI**

*INFO 442, Team 14 — Lanzhou University. Submitted on behalf of the team by the team lead.*

Repository: <https://github.com/Emoliere-s-hajimi/LZU-INFO442-BrainTT>

---

## 1. Domain and motivation

After radiotherapy for high-grade glioma, follow-up MRI commonly shows a **new contrast-enhancing lesion** that may be either *true tumor recurrence* — which demands immediate second-line oncologic intervention — or *radiation necrosis*, a delayed sterile injury that is managed conservatively and would be **worsened** by additional cytotoxic therapy. The two entities look strikingly similar on conventional MRI, yet the clinical actions they require are essentially opposite, and the cost of misclassification is asymmetric and severe: a necrosis patient mistaken for recurrence may receive unnecessary chemotherapy or re-irradiation, while a recurrence patient mistaken for necrosis may lose a narrow therapeutic window. The definitive answer — histopathology after a repeat craniotomy — is invasive and often not feasible. A reliable image-based discriminator is therefore not a marginal optimization but a clinically consequential capability.

Three concrete gaps motivate a new attempt rather than a re-application of existing methods. First, models trained on treatment-naïve cohorts such as BraTS generalise poorly to the post-radiation distribution. Second, generic CNN classifiers ignore the modality-specific enhancement patterns and lesion-topology cues that actually drive clinician reading. Third, post-radiation cohorts that carry both pathology-confirmed labels and complete multimodal MRI are very difficult to assemble; the problem is data-bound at least as much as it is model-bound. Our project is designed as a coordinated answer to all three constraints.

## 2. Dataset description

**Source and access.** The cohort reaches us through a **horizontal (industry–academia) collaborative research project** between Associate Professor **Zhongfeng Kang** (Lanzhou University) and the **Institute of Software, Chinese Academy of Sciences (ISCAS)**, where Professor **Zhulin An** leads the algorithmic side. The clinical origin of the data is **Beijing Tiantan Hospital of Capital Medical University**, one of the largest neurosurgical centres in China and the institution whose IRB approved the underlying study. The data are made available to this INFO 442 project as a contribution of the existing collaboration rather than through a one-off transfer.

**Cohort.** A retrospective consecutive series of **234 patients** admitted between January 2012 and December 2022. Every patient underwent prior glioma resection, received adjuvant radiotherapy, presented with a suspected recurrent lesion on follow-up MRI, and ultimately received a **pathology-confirmed** ground-truth label of either *recurrence* or *radiation necrosis*. The two classes are **not evenly represented** — class imbalance has been explicitly flagged by the advisors as a first-class design concern.

**Format and modalities.** Each case carries **four co-registered conventional MRI sequences**: T1, T1ce (Gd-DTPA, 0.2 ml/kg, 5 min post-injection), T2, and FLAIR. The raw 2D DICOM stacks have been assembled along the slice axis into volumetric **NIfTI** files compatible with a BraTS-style pipeline; modality registration follows the priority order T1ce → T1 → T2 → FLAIR, skull stripping is applied, and intensities are normalised across patients.

**Scale.** A point easily under-estimated from the patient count alone is the volumetric footprint: in its on-disk form the cohort occupies **31.9 GB across 2,530 files**. Among the public BraTS family, only BraTS 2021 contains substantially more cases; several recent BraTS sub-challenges, such as pediatric, meningioma-RT, brain-metastases and Sub-Saharan Africa, are at or below our scale. Critically, **no public dataset of comparable size and label quality currently exists for the recurrence-versus-necrosis question itself**, which makes this collaboration a genuinely rare resource for the problem.

**Ethics and data governance.** The cohort is governed by a data-use agreement between our partner institutions and the team. The underlying Tiantan study was IRB-approved; the retrospective nature of the data collection waived individual informed consent at the source. All raw imaging is processed locally on approved compute; **no patient-identifiable information leaves the partner-controlled environment**, and the public-facing artifacts of the project (repository, figures, metrics, report) contain only de-identified material. Representative de-identified slices and the original data hand-off note are kept in `data_example/` and `data_source_comment/`; the full raw cohort is kept off-repository. Any change of scope that would require sharing data outside this perimeter will be raised with our advisors before it is acted upon.

## 3. Scientific question

We ask **three intertwined research questions**, each framed for a distinct audience and each tied to a concrete, measurable decision about the choice between continuing conservative management and escalating to second-line oncologic intervention.

**RQ1 — For method reviewers (methodology).** *Can the explicit incorporation of medical priors enhance both the accuracy and the interpretability of recurrence-versus-necrosis discrimination on post-radiation multimodal MRI? If so, how should these priors — inter-modality coupling, lesion-topology preservation, and biologically plausible spatial dynamics — be modelled inside a 3D encoder–decoder backbone?* Measured against a generic 3D-CNN baseline on a held-out, pathology-confirmed test split using **AUC, F1, and accuracy**, supplemented by **Dice** for the auxiliary segmentation head and by qualitative interpretability artifacts.

**RQ2 — For statisticians (imbalance).** *Which combination of input-side sampling and output-side loss design best handles the extreme class imbalance of the cohort, in the sense of stabilising minority-class sensitivity without inflating false positives in a clinically meaningful operating regime?* Measured by **sensitivity and specificity at a fixed operating point**, by **PR-AUC** on the minority class, and by the stability of these metrics across cross-validation folds.

**RQ3 — For radiologists (interpretability).** *Does the prior-augmented model behave like a real radiologist rather than a black box — that is, do its spatial and per-modality attributions concentrate on the contrast-enhancement and topology cues that a clinician would actually name, more so than those of a generic 3D-CNN baseline?* Measured by side-by-side qualitative review of saliency / attention maps on a fixed case panel and by the agreement of model-highlighted regions with lesion masks.

## 4. Preliminary hypothesis

We expect that **a multimodal multi-task network whose backbone is augmented with a small number of carefully chosen medical-prior modules and trained with imbalance-aware sampling and loss will outperform a generic 3D-CNN baseline on recurrence-versus-necrosis discrimination**, primarily on **minority-class sensitivity** and **AUC**, while also producing **more clinically legible attributions** than the baseline.

The reasoning behind this hypothesis is threefold. (i) The visual ambiguity of the task is concentrated in modality-specific enhancement patterns and lesion topology — exactly the cues that generic backbones do not represent explicitly, so encoding them as inductive priors should narrow the hypothesis space toward biologically faithful solutions. (ii) The post-radiation distribution is structurally different from the BraTS treatment-naïve regime that has dominated prior work; training directly on the post-radiation cohort, and anchoring the classification head in lesion-localised features via an auxiliary segmentation signal, should reduce the distribution-shift penalty that has limited prior radiomic and CNN approaches. (iii) Joint treatment of class imbalance through weighted sampling at the input side and a focal-style loss at the output side has been repeatedly shown to stabilise minority-class learning when, as here, the imbalance is structural rather than incidental.

We intentionally do **not** commit to a fully specified architecture at the proposal stage; the exact form of each prior module and the design of the ablation study will be guided by what the data actually show in the first three weeks of work. We do not expect to deliver a clinically deployable system within the eight-week window; the final report will state what would still be required — external validation on a non-Tiantan cohort, prospective testing, integration with the radiologists' reading workflow — to move beyond a research prototype.

## 5. Roles

The project is carried out by a five-person undergraduate team at Lanzhou University. Each member owns one strand; roles overlap by design through weekly internal meetings and pairwise code reviews.

| Member | Role | GitHub | Primary responsibility |
|---|---|---|---|
| **Yutong Wang** *(team lead)* | Principal Researcher | `Emoliere` | Methodology, research direction, system integration, communication with the partner institutions, final report. [Google Scholar](https://scholar.google.com/citations?hl=en&authuser=1&user=73MjwF0AAAAJ). |
| **Zijin Wu** | Machine Learning Engineer | `ZijinWu1` | Network and prior-module implementation, training infrastructure. |
| **Xiaopeng Fan** | Data Scientist | `24baigei` | Data curation, cleaning, exploratory analysis, cohort report. |
| **Ye Wang** | Research Scientist | `ye52` | Evaluation protocol, metric pipeline, prediction-visualisation artifacts. |
| **Yunfei Shang** | Product Manager | `shangyf0528` | Project documentation, slide deck, writing editor, visual identity of the final presentation. |

**Acknowledgements.** We gratefully acknowledge **Assoc. Prof. Zhongfeng Kang** (Lanzhou University) and **Prof. Zhulin An** (ISCAS), whose horizontal collaborative research project is the channel through which this cohort reaches our team, and the clinical team at **Beijing Tiantan Hospital, Capital Medical University**, who originally curated the cohort and authorise its use here.
