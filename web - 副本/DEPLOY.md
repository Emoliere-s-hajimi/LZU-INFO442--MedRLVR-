# Deploy · BrainTT Project Showcase

The site is **pure static** — no build, no API, no database. Anything that
serves files over HTTP will work. These instructions are for **Vercel**, the
recommended host for this project.

## Pre-flight (already done in this repo)

- `vercel.json` — caching headers, CORS for the NIfTI volumes, security
  hardening.
- `.vercelignore` — keeps the data-pipeline Python scripts out of the
  deploy.
- All CDN dependencies (Plotly, Three.js, NiiVue via esm.sh,
  Google Fonts) are loaded at runtime — no bundling step needed.

Project size: **~48 MB**, fits comfortably under Vercel's 100 MB hobby tier.

---

## Option 1 — Vercel CLI (fastest, ~2 minutes)

```bash
# One-time install
npm install -g vercel

# One-time login
vercel login

# From the project root
cd web

# Preview deploy — gives you a unique URL, no production routing yet
vercel

# Production deploy — assigns your *.vercel.app domain (or custom domain)
vercel --prod
```

The first `vercel` run will ask:

| Prompt | Answer |
|---|---|
| Set up and deploy "web"? | **Y** |
| Which scope? | your personal account |
| Link to existing project? | **N** (first time) |
| What's your project's name? | `braintt` (or anything) |
| In which directory is your code located? | **`./`** (just press Enter — we're already in `web/`) |
| Want to modify settings? | **N** |

Vercel auto-detects "static site" and skips the build step. The site is live
on `https://braintt-<random>.vercel.app/` in 30–60 seconds.

## Option 2 — Vercel via GitHub auto-deploy

If you want every push to `main` to redeploy:

1. Push the repo to GitHub.
2. Go to <https://vercel.com/new>, click **Import Git Repository**.
3. Pick the repo. **Root Directory** = `web`. Everything else: defaults.
4. Click **Deploy**.

Every push to `main` re-deploys; pull requests get preview URLs.

## Option 3 — Any other static host

The `web/` directory is self-contained. Drop it on:

```bash
# Cloudflare Pages
npx wrangler pages deploy web --project-name braintt

# Netlify
cd web && netlify deploy --prod --dir=.

# GitHub Pages — copy web/ to /docs and enable Pages
cp -r web docs && git add docs && git commit -m "deploy" && git push

# Bare nginx
rsync -avz web/ user@server:/srv/www/braintt/
```

Then point nginx at `/srv/www/braintt/`:

```nginx
server {
  listen 443 ssl http2;
  server_name braintt.example.com;
  root /srv/www/braintt;
  index index.html;

  # Match the Vercel cache strategy
  location ~* \.(nii\.gz|png|jpg|jpeg)$    { expires 1y; add_header Cache-Control "public, immutable"; }
  location ~* \.json$                       { expires 5m; }
  location ~* \.(js|css)$                   { expires 5m; }

  # NIfTI files are not gzipped further (.nii.gz is already deflate)
  gzip on;
  gzip_types text/css application/javascript application/json image/svg+xml;
}
```

---

## Smoke-test the deploy

After it's live, click through this checklist on the deployed URL:

1. **Hero** — Three.js brain rotates, stat cards readable.
2. **Cohort Explorer** — 322 dots render; lasso-select a region; the
   readout (count, AUC, median χ, missing-pct) all change.
3. **Synthesis Demo** — drag the sliders, the gold canvas updates live.
4. **Case Viewer** — pick a tab; the NiiVue volume loads within ~5 seconds
   (this is the most CDN-sensitive section because it pulls
   `https://esm.sh/@niivue/niivue@0.69.0` and its dependency chain).
5. **Performance** — ROC, Pareto, parallel-coords, Sankey all render.
6. **Console** — drag the threshold; the confusion matrix swaps colours.
7. **Robustness** — noise slider moves the marker diamonds along curves.
8. **Interpretability** — Grad-CAM PNGs load; t-SNE / feature-space /
   calibration plots render.
9. **Sandbox** — switch backbones, the gold ⬥ jumps around the Pareto plot.
10. **Press `?`** — keyboard shortcut overlay opens.

## Custom domain

Vercel **Project Settings → Domains** — add your domain, point its DNS to
Vercel as described, takes 1–5 minutes to propagate. Free SSL via Let's
Encrypt.

## Updating the site

Just re-run `vercel --prod` (CLI) or push to the linked branch (GitHub
auto-deploy). The HTML / JSON have short caches; new numbers and copy show
up immediately. JS/CSS take up to 5 minutes to fully propagate; force-refresh
to see them sooner.

## Bandwidth

Vercel's hobby tier ships **100 GB / month**. The home page is ~50 KB,
the case viewer's volumes add up to ~46 MB *per first visitor*, so the
budget covers ≈ 2,000 first-time complete viewings per month before you
need to upgrade. The long-cache headers make repeat visits effectively
free.
