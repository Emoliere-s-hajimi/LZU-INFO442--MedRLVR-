// Live FLAIR synthesis demo — lazy-init + rAF-throttled.
//
//   FLAIR_synth(x) = (w_t2 * T2(x) + w_t1 * T1(x)) / ||w||_2
//
// Foreground mean / std are computed once per slice and cached; the
// per-pixel z-score / weighted sum / re-scaling is one tight Uint8 loop.

(function () {
  const NSLICES = 13;
  const cT2 = document.getElementById("synthT2");
  const cT1 = document.getElementById("synthT1");
  const cOut = document.getElementById("synthOut");
  if (!cT2 || !cT1 || !cOut) return;

  BrainTT.lazy(cOut.closest("section"), boot);

  function boot() {
    const wT2 = document.getElementById("wT2");
    const wT1 = document.getElementById("wT1");
    const slice = document.getElementById("synthSlice");
    const wT2val = document.getElementById("wT2val");
    const wT1val = document.getElementById("wT1val");
    const sliceVal = document.getElementById("synthSliceVal");
    const preset = document.getElementById("synthPreset");
    const formula = document.getElementById("recipeFormula");

    const ctxT2 = cT2.getContext("2d");
    const ctxT1 = cT1.getContext("2d");
    const ctxOut = cOut.getContext("2d");

    // Per-slice cache:  i → { width, height, t1u8, t2u8, m1, s1, m2, s2 }
    const cache = [];

    const tmpCanvas = document.createElement("canvas");
    const tmpCtx = tmpCanvas.getContext("2d");

    function loadImg(src) {
      return new Promise((res, rej) => {
        const img = new Image();
        img.onload = () => res(img);
        img.onerror = rej;
        img.src = src;
      });
    }

    function extractGray(img) {
      tmpCanvas.width = img.width;
      tmpCanvas.height = img.height;
      tmpCtx.drawImage(img, 0, 0);
      const d = tmpCtx.getImageData(0, 0, img.width, img.height).data;
      const u8 = new Uint8Array(img.width * img.height);
      for (let i = 0, j = 0; j < u8.length; i += 4, j++) u8[j] = d[i];
      return u8;
    }

    function foregroundStats(u8a, u8b) {
      let sumA = 0, sqA = 0, sumB = 0, sqB = 0, n = 0;
      const N = u8a.length;
      for (let i = 0; i < N; i++) {
        if (u8a[i] > 8 || u8b[i] > 8) {  // ~3 % threshold
          const a = u8a[i] / 255, b = u8b[i] / 255;
          sumA += a; sqA += a * a;
          sumB += b; sqB += b * b;
          n++;
        }
      }
      n = Math.max(n, 1);
      const mA = sumA / n;
      const mB = sumB / n;
      return {
        mA, sA: Math.sqrt(Math.max(sqA / n - mA * mA, 1e-6)),
        mB, sB: Math.sqrt(Math.max(sqB / n - mB * mB, 1e-6)),
      };
    }

    async function loadAll() {
      const tasks = [];
      for (let i = 0; i < NSLICES; i++) {
        const idx = String(i).padStart(2, "0");
        tasks.push(loadImg(`data/synth/t1_${idx}.png`).then((img) => ({ kind: "t1", i, img })));
        tasks.push(loadImg(`data/synth/t2_${idx}.png`).then((img) => ({ kind: "t2", i, img })));
      }
      const loaded = await Promise.all(tasks);
      // Bucket by slice
      const buckets = new Array(NSLICES).fill(null).map(() => ({}));
      const sample = loaded[0].img;
      [cT2, cT1, cOut].forEach((c) => {
        if (c.width !== sample.width || c.height !== sample.height) {
          c.width = sample.width; c.height = sample.height;
        }
      });
      loaded.forEach(({ kind, i, img }) => {
        buckets[i][kind] = extractGray(img);
        buckets[i].width = img.width;
        buckets[i].height = img.height;
      });
      for (let i = 0; i < NSLICES; i++) {
        const b = buckets[i];
        const stats = foregroundStats(b.t2, b.t1);
        cache.push({
          width: b.width,
          height: b.height,
          t1u8: b.t1,
          t2u8: b.t2,
          mT2: stats.mA, sT2: stats.sA,
          mT1: stats.mB, sT1: stats.sB,
        });
      }
      render();
    }

    function paintGray(ctx, u8, w, h) {
      const out = ctx.createImageData(w, h);
      const o = out.data;
      for (let i = 0, j = 0; j < u8.length; i += 4, j++) {
        o[i] = o[i + 1] = o[i + 2] = u8[j];
        o[i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    }

    function paintGold(ctx, u8, w, h) {
      const out = ctx.createImageData(w, h);
      const o = out.data;
      for (let i = 0, j = 0; j < u8.length; i += 4, j++) {
        const v = u8[j];
        o[i]     = v;              // R
        o[i + 1] = Math.round(v * 0.86);  // G
        o[i + 2] = Math.round(v * 0.35);  // B
        o[i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    }

    function synth(c, wT2v, wT1v) {
      const norm = Math.sqrt(wT2v * wT2v + wT1v * wT1v) || 1e-8;
      const aT2 = wT2v / norm;
      const aT1 = wT1v / norm;
      const lo = -2.5, hi = 3.0, span = hi - lo;
      const N = c.t1u8.length;
      const out = new Uint8Array(N);
      const invSt2 = 1 / c.sT2, invSt1 = 1 / c.sT1;
      for (let i = 0; i < N; i++) {
        const t2 = c.t2u8[i] / 255, t1 = c.t1u8[i] / 255;
        if (t2 < 0.03 && t1 < 0.03) { out[i] = 0; continue; }
        const z2 = (t2 - c.mT2) * invSt2;
        const z1 = (t1 - c.mT1) * invSt1;
        let v = aT2 * z2 + aT1 * z1;
        v = (v - lo) / span;
        if (v < 0) v = 0; else if (v > 1) v = 1;
        out[i] = (v * 255) | 0;
      }
      return out;
    }

    function pickSlice(frac) {
      const i = Math.round(frac * (NSLICES - 1));
      return Math.max(0, Math.min(NSLICES - 1, i));
    }

    // Renders only differ when params change
    let lastWt2 = NaN, lastWt1 = NaN, lastIdx = -1;
    function render() {
      if (!cache.length) return;
      const wT2v = parseFloat(wT2.value);
      const wT1v = parseFloat(wT1.value);
      const idx = pickSlice(parseFloat(slice.value));
      const c = cache[idx];
      if (idx !== lastIdx) {
        paintGray(ctxT2, c.t2u8, c.width, c.height);
        paintGray(ctxT1, c.t1u8, c.width, c.height);
        lastIdx = idx;
      }
      if (wT2v !== lastWt2 || wT1v !== lastWt1 || idx !== lastIdx) {
        const out = synth(c, wT2v, wT1v);
        paintGold(ctxOut, out, c.width, c.height);
        lastWt2 = wT2v; lastWt1 = wT1v;
      }

      wT2val.textContent = (wT2v >= 0 ? "+" : "") + wT2v.toFixed(2);
      wT1val.textContent = (wT1v >= 0 ? "+" : "") + wT1v.toFixed(2);
      sliceVal.textContent = `${Math.round(parseFloat(slice.value) * 100)}%`;

      formula.innerHTML =
        `FLAIR<sub>synth</sub>(x) = ( <span class="const">${wT2v >= 0 ? "+" : ""}${wT2v.toFixed(2)}</span> · T2(x) ` +
        `<span class="const">${wT1v >= 0 ? "+" : ""}${wT1v.toFixed(2)}</span> · T1(x) ) / ||w||₂`;
    }

    const throttled = BrainTT.rafDebounce(render);
    [wT2, wT1, slice].forEach((el) => el.addEventListener("input", throttled));
    preset.addEventListener("click", () => {
      wT2.value = 1.0;
      wT1.value = -0.5;
      slice.value = 0.5;
      render();
    });

    loadAll().catch((e) => console.warn("synth: failed to load slices", e));
  }
})();
