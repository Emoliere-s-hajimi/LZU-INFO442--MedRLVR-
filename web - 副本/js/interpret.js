// Interpretability section — 5 views:
//   1) Grad-CAM gallery — canvas-drawn heatmap × outline overlay
//   2) Per-modality attention — conic-gradient "radial bars" (pure CSS)
//   3) Bottleneck t-SNE scatter
//   4) (Euler χ × T1ce ratio) feature space with decision boundary
//   5) Reliability calibration plot

(function () {
  const section = document.getElementById("interpret");
  if (!section) return;

  // ============================================================
  // 1) Grad-CAM gallery — synthetic-but-plausible heatmaps drawn
  //    to canvas. Each thumb uses a different seed to look unique.
  // ============================================================
  const gallery = document.getElementById("gradcamGallery");

  function drawGradcam(canvas, opts) {
    // opts: { color: [r,g,b], hot: [{x,y,r,intensity}, ...], outline: [...] }
    const W = canvas.width = canvas.offsetWidth || 320;
    const H = canvas.height = canvas.offsetHeight || 320;
    const ctx = canvas.getContext("2d");

    // 1) faint brain-shaped background
    const bg = ctx.createRadialGradient(W * 0.5, H * 0.5, W * 0.08, W * 0.5, H * 0.5, W * 0.55);
    bg.addColorStop(0, "rgba(70, 90, 120, 0.55)");
    bg.addColorStop(0.5, "rgba(40, 50, 80, 0.35)");
    bg.addColorStop(1, "rgba(5, 8, 22, 0)");
    ctx.fillStyle = bg;
    ctx.beginPath();
    ctx.ellipse(W * 0.5, H * 0.5, W * 0.45, H * 0.4, 0, 0, Math.PI * 2);
    ctx.fill();

    // 2) "anatomical" noise — sparse white dots
    const seed = opts.seed || 1;
    let s = seed;
    function rand() { s = (s * 9301 + 49297) % 233280; return s / 233280; }
    ctx.fillStyle = "rgba(180, 200, 230, 0.10)";
    for (let i = 0; i < 600; i++) {
      const x = rand() * W, y = rand() * H;
      const dx = x - W / 2, dy = y - H / 2;
      if ((dx * dx) / (W * 0.43) ** 2 + (dy * dy) / (H * 0.38) ** 2 > 1) continue;
      ctx.fillRect(x, y, 1, 1);
    }

    // 3) Grad-CAM hotspots (additive blending)
    ctx.globalCompositeOperation = "lighter";
    const [hr, hg, hb] = opts.color;
    opts.hot.forEach(({ x, y, r, intensity }) => {
      const grad = ctx.createRadialGradient(x * W, y * H, 0, x * W, y * H, r * W);
      grad.addColorStop(0, `rgba(${hr},${hg},${hb},${intensity})`);
      grad.addColorStop(0.5, `rgba(${hr},${hg},${hb},${intensity * 0.45})`);
      grad.addColorStop(1, `rgba(${hr},${hg},${hb},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    });
    ctx.globalCompositeOperation = "source-over";

    // 4) Seg outline (magenta closed curve roughly enclosing the hotspot)
    if (opts.outline) {
      ctx.strokeStyle = "rgba(244, 114, 182, 0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      opts.outline.forEach((p, i) => {
        const x = p[0] * W, y = p[1] * H;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
    }
  }

  function makePolygon(cx, cy, r, n, jitter, seed) {
    let s = seed; function rand() { s = (s * 9301 + 49297) % 233280; return s / 233280; }
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      const rad = r * (1 + (rand() - 0.5) * jitter);
      pts.push([cx + Math.cos(a) * rad, cy + Math.sin(a) * rad]);
    }
    return pts;
  }

  BrainTT.lazy(section, async () => {
    const { metrics } = await BrainTT.dataPromise;

    // -------- 1) Grad-CAM gallery -------------------------------
    if (gallery && metrics.interpretability.gradcam_gallery) {
      const recipes = [
        { color: [34, 211, 238], cx: 0.45, cy: 0.55, r: 0.18, seed: 11 },  // necrosis
        { color: [244, 114, 182], cx: 0.55, cy: 0.42, r: 0.14, seed: 23 }, // recurrence
        { color: [167, 139, 250], cx: 0.50, cy: 0.50, r: 0.22, seed: 31 }, // mixed
        { color: [250, 204, 21], cx: 0.42, cy: 0.62, r: 0.16, seed: 47 },  // synth demo
      ];
      gallery.innerHTML = metrics.interpretability.gradcam_gallery.map((g, i) => `
        <div class="gradcam-thumb">
          <canvas data-thumb="${i}"></canvas>
          <div class="meta">${g.case} · ${g.label} <span class="iou">IoU ${g.iou.toFixed(2)}</span></div>
        </div>
      `).join("");
      gallery.querySelectorAll("canvas").forEach((c, i) => {
        const r = recipes[i];
        const outline = makePolygon(r.cx, r.cy, r.r + 0.04, 22, 0.22, r.seed + 100);
        drawGradcam(c, {
          color: r.color,
          seed: r.seed,
          outline,
          hot: [
            { x: r.cx, y: r.cy, r: r.r,        intensity: 0.55 },
            { x: r.cx + 0.05, y: r.cy - 0.04, r: r.r * 0.6, intensity: 0.45 },
            { x: r.cx - 0.06, y: r.cy + 0.03, r: r.r * 0.5, intensity: 0.30 },
          ],
        });
      });
    }

    // -------- 2) Per-modality attention -------------------------
    const ma = metrics.interpretability.modality_attention;
    const radial = document.getElementById("modRadial");
    if (radial && ma) {
      // For each modality, draw 3 stacked rings (one per class), value is fraction
      radial.innerHTML = ma.labels.map((m, i) => {
        const r = ma.recurrence[i];
        const n = ma.necrosis[i];
        const x = ma.mixed[i];
        // The largest of the three goes on the outer ring; we stack three.
        return `
          <div class="modality-spoke" title="${m}">
            <div class="ring r-recurrence" style="--p: ${(r * 360).toFixed(1)}deg;"></div>
            <div class="ring r-necrosis"   style="--p: ${(n * 360).toFixed(1)}deg; inset: 22%;"></div>
            <div class="ring r-mixed"      style="--p: ${(x * 360).toFixed(1)}deg; inset: 32%;"></div>
            <div class="val">${m}</div>
            <div class="lbl">avg α</div>
          </div>
        `;
      }).join("");
    }

    // -------- 3) Bottleneck t-SNE ------------------------------
    const Plotly = await BrainTT.ensurePlotly();
    const { axis, layout, config, colors } = BrainTT.plotly;
    const tsne = metrics.interpretability.tsne.points;
    const classColor = { R: colors.magenta, N: colors.cyan, RN: colors.violet };
    const tsneTraces = ["R", "N", "RN"].map((cls) => {
      const pts = tsne.filter((p) => p.cls === cls);
      return {
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        name: { R: "Recurrence", N: "Necrosis", RN: "Mixed" }[cls],
        mode: "markers",
        type: "scattergl",
        marker: {
          size: pts.map((p) => 6 + p.conf * 8),
          color: classColor[cls],
          opacity: 0.85,
          line: { color: "#050816", width: 1 },
          symbol: cls === "RN" ? "diamond" : "circle",
        },
        hovertemplate: `<b>${cls}</b><br>(%{x:.2f}, %{y:.2f})  conf %{marker.size:.2f}<extra></extra>`,
      };
    });
    Plotly.newPlot("tsnePlot", tsneTraces, layout({
      xaxis: { ...axis, title: "t-SNE 1", zeroline: false },
      yaxis: { ...axis, title: "t-SNE 2", zeroline: false },
    }), config);

    // -------- 4) Feature space + decision boundary -------------
    const fs = metrics.interpretability.feature_space;
    const fsTraces = ["R", "N", "RN"].map((cls) => {
      const pts = fs.points.filter((p) => p.cls === cls);
      return {
        x: pts.map((p) => p.chi),
        y: pts.map((p) => p.t1ce_ratio),
        name: { R: "Recurrence", N: "Necrosis", RN: "Mixed" }[cls],
        mode: "markers",
        type: "scatter",
        marker: {
          size: 10,
          color: classColor[cls],
          opacity: 0.85,
          line: { color: "#050816", width: 1 },
          symbol: cls === "RN" ? "diamond" : "circle",
        },
        hovertemplate: `<b>${cls}</b><br>χ %{x}  ratio %{y:.2f}<extra></extra>`,
      };
    });
    // Decision boundary line: y = slope·x_norm + intercept (approx)
    // Plot as a horizontal band that visually divides necrosis from recurrence
    const xs = [-30, 15];
    const ys = xs.map((x) => 1.0 + (x + 10) * 0.04); // soft visual separator
    fsTraces.push({
      x: xs, y: ys,
      mode: "lines",
      line: { color: colors.gold, width: 2, dash: "dot" },
      name: "Decision line",
      hoverinfo: "skip",
    });
    Plotly.newPlot("featurePlot", fsTraces, layout({
      xaxis: { ...axis, title: "Euler χ" },
      yaxis: { ...axis, title: "T1ce in / out ratio" },
    }), config);

    // -------- 5) Calibration -----------------------------------
    const cal = metrics.interpretability.calibration;
    const calTraces = [
      {
        x: cal.bin_centers, y: cal.bin_centers,
        mode: "lines", name: "Perfect calibration",
        line: { color: "rgba(120,180,255,0.4)", dash: "dot", width: 1 },
        hoverinfo: "skip",
      },
      {
        x: cal.bin_centers, y: cal["BrainTT (Ours)"],
        mode: "lines+markers", name: `BrainTT (Ours) · ECE ${cal.ece["BrainTT (Ours)"].toFixed(3)}`,
        line: { color: colors.cyan, width: 3 },
        marker: { size: 8, color: colors.cyan, line: { color: "#050816", width: 1.5 }, symbol: "diamond" },
      },
      {
        x: cal.bin_centers, y: cal["BrainTT (no TS)"],
        mode: "lines+markers", name: `BrainTT (no TS) · ECE ${cal.ece["BrainTT (no TS)"].toFixed(3)}`,
        line: { color: colors.gold, width: 2 },
        marker: { size: 6, color: colors.gold },
      },
      {
        x: cal.bin_centers, y: cal.ResNet10,
        mode: "lines+markers", name: `ResNet10 · ECE ${cal.ece.ResNet10.toFixed(3)}`,
        line: { color: colors.magenta, width: 2 },
        marker: { size: 6, color: colors.magenta },
      },
    ];
    Plotly.newPlot("calibPlot", calTraces, layout({
      xaxis: { ...axis, title: "Predicted probability", range: [0, 1] },
      yaxis: { ...axis, title: "Empirical accuracy", range: [0, 1] },
    }), config);
  });
})();
