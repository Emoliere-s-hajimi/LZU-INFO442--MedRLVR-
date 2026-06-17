# BrainTT · Project Showcase

A single-page, fully interactive site that tells the **BrainTT** story end-to-end:
the clinical stakes, the data, our missing-modality synthesis trick, the prior
modules, a live MRI case viewer, model performance, an ablation playground, and
the failure modes we ship with the model.

```
web/
├── index.html             — 9-section single-page app
├── style.css              — neon-dark theme, custom animations, glass cards
├── js/
│   ├── data.js            — shared data loader (cases.json + metrics.json)
│   ├── main.js            — scroll-reveal, nav, donut chart, modality bars
│   ├── hero.js            — Three.js particle "brain"
│   ├── synthesis.js       — live FLAIR-synthesis demo (canvas)
│   ├── architecture.js    — interactive prior cards + node diagram
│   ├── viewer.js          — NiiVue case viewer (NIfTI in-browser)
│   ├── metrics.js         — Plotly ROC + Pareto + leaderboard
│   └── ablation.js        — prior toggle playground + failure browser
├── data/
│   ├── cases.json         — manifest of 4 demo patients
│   ├── metrics.json       — bench numbers, priors, ROC curves, failures
│   ├── _make_synth_slices.py — pre-render T1/T2 slices for the synth demo
│   ├── synth/             — 26 grayscale PNG slices (T1 / T2) for synth canvas
│   └── cases/             — 4 patients × {t1,t1ce,t2,seg}.nii.gz (~46 MB)
│       ├── N_005/         — radiation necrosis · cavitary, Euler χ ≈ -22
│       ├── R_148/         — tumour recurrence · solid, simply-connected
│       ├── RN_003/        — mixed pathology · ambiguous confidence
│       └── RN_044/        — only T1 + T1ce on disk · synthesis showcase
└── README.md              — this file
```

## What's on the page

| # | Section | What you do there |
|---|---|---|
| 1 | **Hero** | Watch a Three.js particle brain rotate with mouse parallax. |
| 2 | **The Clinical Stakes** | Read the recurrence-vs-necrosis decision tree and why missing each direction is bad. |
| 3 | **The Data Story** | Donut chart of the 322-patient cohort + a modality coverage bar reveal. |
| 4 | **Missing-Modality Synthesis** | Drag two weight sliders and a slice slider; we **literally re-run the recipe in the browser** on real T1/T2 slices from N_005 and show you the synthesised FLAIR. |
| 5 | **Architecture** | Hover or click the nodes in the SVG diagram — each one binds to a prior card that explains what it does, with the math. |
| 6 | **Interactive Case Viewer** | Pick one of 4 real cases. Switch modalities, toggle segmentation overlay, flip between axial and multi-planar views. All powered by [NiiVue](https://niivue.github.io/niivue/), so the NIfTI volume is rendered in WebGL in your browser. |
| 7 | **Performance Dashboard** | Plotly ROC for the top-4 models, sensitivity-vs-parameters Pareto scatter, and a sortable leaderboard. |
| 8 | **Ablation Playground** | Toggle the three priors on or off. The AUC / sensitivity / parameter readout updates from real ablation rows in `metrics.json`. |
| 9 | **Failure Modes** | Four documented failure modes, each with its frequency and the mitigation we ship. |

## Run it

### Option 1 — Local (zero dependencies)

```bash
cd web
python -m http.server 8000
# open http://localhost:8000
```

That's it. No build step, no `npm install`, no bundler. All third-party libs
(Three.js, Plotly, NiiVue) load from CDN.

### Option 2 — GitHub Pages

1. Commit the `web/` folder.
2. In the GitHub repo settings → **Pages**, set the source to your branch and
   the directory to `/web` (or move the contents to `/docs` if that's easier).
3. Visit `https://<user>.github.io/<repo>/`.

### Option 3 — Any static host (Vercel / Netlify / Cloudflare Pages / Nginx)

The `web/` directory is fully static — point any HTTP server at it. **No
back-end is required**: predictions, biomarkers, and metrics are pre-computed
and shipped as JSON, the synth demo is computed live in JS, and NiiVue
streams the NIfTI files over plain `GET` requests.

```bash
# Vercel
cd web && vercel deploy --prod

# Netlify
cd web && netlify deploy --prod --dir=.

# Plain nginx
location /braintt/ { root /srv/www/; index index.html; }
```

**One caveat** — the NIfTI volumes are ~10–15 MB each (46 MB total). They're
served over plain HTTP, no CORS gymnastics needed when same-origin, but if you
embed the page in an `<iframe>` from a different origin, set
`Access-Control-Allow-Origin: *` on the static host.

## Regenerating the synth demo slices

`web/data/synth/` ships 13 axial slices each of T1 and T2 from patient N_005,
saved as 8-bit grayscale PNG. If you swap the source case or the slice
selection, re-run:

```bash
python web/data/_make_synth_slices.py
```

You need `numpy`, `nibabel`, and `pillow` — already in `requirements.txt`.

## Updating numbers

All metric numbers, ablation rows, ROC curves, failure-mode copy, and case
narratives live in `data/metrics.json` and `data/cases.json`. **No code changes
needed** when you re-train and want to push fresh results — overwrite the
relevant JSON, hard-refresh the page.

## Customising the demo cases

```bash
# Pick a new patient, copy the four files, then add an entry to cases.json:
cp data1/数据集/SourceData/SourcePreprocess_SegLabel_202110/R/200/R_200_t1.nii.gz \
   web/data/cases/R_200/t1.nii.gz
# ... t1ce / t2 / seg same way
# Then edit web/data/cases.json — copy an existing block, tweak fields.
```

The narrative + biomarker fields are free-text — they're how you tell the
clinical story for each case.

## Technology

| | |
|---|---|
| **HTML / CSS / JS** | Hand-rolled. No framework, no bundler. |
| **Three.js** | r160 from `cdn.jsdelivr.net`. The hero scene only. |
| **Plotly** | 2.30 from `cdn.plot.ly`. ROC + Pareto plots. |
| **NiiVue** | latest from `niivue.github.io`. WebGL NIfTI rendering. |
| **Fonts** | Inter + JetBrains Mono via Google Fonts. |

Total page weight (without the NIfTI volumes): **~120 KB compressed**.
CDN libs add ~600 KB, served gzipped, cached for the user's life.

## Browser support

- **Required:** WebGL2 (NiiVue needs it; ~98 % of browsers as of 2026)
- Tested on recent Chrome, Edge, Firefox, Safari
- The page is responsive down to ~360 px; the case viewer benefits from ≥ 1024 px

## License

The site code is part of the BrainTT project repository. The MRI data
shown here is from the Beijing Tiantan Hospital cohort — for research use only,
not redistributable beyond the agreed academic scope.
