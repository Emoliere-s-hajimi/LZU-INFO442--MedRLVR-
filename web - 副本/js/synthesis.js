// Live FLAIR synthesis demo:
//   FLAIR_synth(x) = (w_t2 * T2(x) + w_t1 * T1(x)) / ||w||_2
// Renders to three canvases — T2, T1, and the synthesised FLAIR.

(function () {
  const NSLICES = 13;
  const wT2 = document.getElementById("wT2");
  const wT1 = document.getElementById("wT1");
  const slice = document.getElementById("synthSlice");
  const wT2val = document.getElementById("wT2val");
  const wT1val = document.getElementById("wT1val");
  const sliceVal = document.getElementById("synthSliceVal");
  const preset = document.getElementById("synthPreset");
  const formula = document.getElementById("recipeFormula");

  const cT2 = document.getElementById("synthT2");
  const cT1 = document.getElementById("synthT1");
  const cOut = document.getElementById("synthOut");
  if (!cT2 || !cT1 || !cOut) return;

  const ctxT2 = cT2.getContext("2d");
  const ctxT1 = cT1.getContext("2d");
  const ctxOut = cOut.getContext("2d");

  // Pre-load all slice ImageData up front (one fetch per file).
  const t1Slices = [];
  const t2Slices = [];
  const tmpCanvas = document.createElement("canvas");
  const tmpCtx = tmpCanvas.getContext("2d");

  let WIDTH = 0, HEIGHT = 0;
  function ensureCanvasSize(w, h) {
    [cT2, cT1, cOut].forEach((c) => {
      if (c.width !== w || c.height !== h) {
        c.width = w; c.height = h;
      }
    });
    WIDTH = w; HEIGHT = h;
  }

  function loadImg(src) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = src;
    });
  }

  async function loadAll() {
    const tasks = [];
    for (let i = 0; i < NSLICES; i++) {
      const idx = String(i).padStart(2, "0");
      tasks.push(loadImg(`data/synth/t1_${idx}.png`).then((img) => ({ kind: "t1", i, img })));
      tasks.push(loadImg(`data/synth/t2_${idx}.png`).then((img) => ({ kind: "t2", i, img })));
    }
    const loaded = await Promise.all(tasks);
    if (!loaded.length) return;
    const sample = loaded[0].img;
    tmpCanvas.width = sample.width;
    tmpCanvas.height = sample.height;
    ensureCanvasSize(sample.width, sample.height);

    loaded.forEach(({ kind, i, img }) => {
      tmpCtx.clearRect(0, 0, tmpCanvas.width, tmpCanvas.height);
      tmpCtx.drawImage(img, 0, 0);
      const data = tmpCtx.getImageData(0, 0, tmpCanvas.width, tmpCanvas.height);
      (kind === "t1" ? t1Slices : t2Slices)[i] = data;
    });
    render();
  }

  // z-score over foreground (pixels > 8 / 255 ≈ 0.03), then linear combine.
  function synth(t2Data, t1Data, wT2v, wT1v) {
    const len = t2Data.data.length / 4;
    const t2 = new Float32Array(len);
    const t1 = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      t2[i] = t2Data.data[i * 4] / 255;
      t1[i] = t1Data.data[i * 4] / 255;
    }
    // Foreground stats: use voxels >0.03 on either modality
    let sum2 = 0, sumSq2 = 0, sum1 = 0, sumSq1 = 0, n = 0;
    for (let i = 0; i < len; i++) {
      if (t2[i] > 0.03 || t1[i] > 0.03) {
        sum2 += t2[i]; sumSq2 += t2[i] * t2[i];
        sum1 += t1[i]; sumSq1 += t1[i] * t1[i];
        n++;
      }
    }
    if (n === 0) n = 1;
    const m2 = sum2 / n, s2 = Math.sqrt(Math.max(sumSq2 / n - m2 * m2, 1e-6));
    const m1 = sum1 / n, s1 = Math.sqrt(Math.max(sumSq1 / n - m1 * m1, 1e-6));

    const norm = Math.sqrt(wT2v * wT2v + wT1v * wT1v) || 1e-8;
    const out = new Float32Array(len);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < len; i++) {
      const z2 = (t2[i] - m2) / s2;
      const z1 = (t1[i] - m1) / s1;
      let v = (wT2v * z2 + wT1v * z1) / norm;
      // mask: only paint where we actually have anatomy
      if (t2[i] <= 0.03 && t1[i] <= 0.03) v = -3.0;
      out[i] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    // Map to 0..255 with foreground percentiles
    const p1 = -2.5, p99 = 3.0; // perceptually stable window
    const outImg = ctxOut.createImageData(t2Data.width, t2Data.height);
    for (let i = 0; i < len; i++) {
      let v = (out[i] - p1) / (p99 - p1);
      v = Math.max(0, Math.min(1, v));
      const b = Math.round(v * 255);
      outImg.data[i * 4]     = b;
      outImg.data[i * 4 + 1] = b;
      outImg.data[i * 4 + 2] = b;
      outImg.data[i * 4 + 3] = 255;
    }
    // Apply a subtle gold tint by boosting blue-channel less than red/green
    for (let i = 0; i < len; i++) {
      const b = outImg.data[i * 4];
      // gold → r=250, g=204, b=21 normalised
      outImg.data[i * 4]     = Math.min(255, b * 1.0);
      outImg.data[i * 4 + 1] = Math.min(255, b * 0.85);
      outImg.data[i * 4 + 2] = Math.min(255, b * 0.35);
    }
    return outImg;
  }

  function paint(ctx, imgData) {
    ctx.putImageData(imgData, 0, 0);
  }

  function pickSlice(frac) {
    const i = Math.round(frac * (NSLICES - 1));
    return Math.max(0, Math.min(NSLICES - 1, i));
  }

  function render() {
    if (!t1Slices.length || !t2Slices.length) return;
    const wT2v = parseFloat(wT2.value);
    const wT1v = parseFloat(wT1.value);
    const idx = pickSlice(parseFloat(slice.value));

    const t2img = t2Slices[idx];
    const t1img = t1Slices[idx];
    if (!t2img || !t1img) return;

    ctxT2.putImageData(t2img, 0, 0);
    ctxT1.putImageData(t1img, 0, 0);
    paint(ctxOut, synth(t2img, t1img, wT2v, wT1v));

    wT2val.textContent = (wT2v >= 0 ? "+" : "") + wT2v.toFixed(2);
    wT1val.textContent = (wT1v >= 0 ? "+" : "") + wT1v.toFixed(2);
    sliceVal.textContent = `${Math.round(parseFloat(slice.value) * 100)}%`;

    formula.innerHTML =
      `FLAIR<sub>synth</sub>(x) = ( <span class="const">${wT2v >= 0 ? "+" : ""}${wT2v.toFixed(2)}</span> · T2(x) ` +
      `<span class="const">${wT1v >= 0 ? "+" : ""}${wT1v.toFixed(2)}</span> · T1(x) ) / ||w||₂`;
  }

  [wT2, wT1, slice].forEach((el) => el.addEventListener("input", render));
  preset.addEventListener("click", () => {
    wT2.value = 1.0;
    wT1.value = -0.5;
    slice.value = 0.5;
    render();
  });

  loadAll().catch((e) => console.error("synth: failed to load slices", e));
})();
