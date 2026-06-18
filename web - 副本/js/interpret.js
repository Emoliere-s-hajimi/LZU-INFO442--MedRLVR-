// Interpretability section — 5 views:
//   1) Grad-CAM gallery — pre-rendered PNGs from web/data/gradcam/, made
//      by web/data/_make_gradcam_overlays.py on the real cohort cases.
//   2) Per-modality attention — conic-gradient "radial bars" (pure CSS)
//   3) Bottleneck t-SNE scatter
//   4) (Euler χ × T1ce ratio) feature space with decision boundary
//   5) Reliability calibration plot

(function () {
  const section = document.getElementById("interpret");
  if (!section) return;

  const gallery = document.getElementById("gradcamGallery");

  BrainTT.lazy(section, async () => {
    const { metrics } = await BrainTT.dataPromise;

    // -------- 1) Grad-CAM gallery (real PNG overlays) ----------
    if (gallery && metrics.interpretability.gradcam_gallery) {
      gallery.innerHTML = metrics.interpretability.gradcam_gallery.map((g) => `
        <div class="gradcam-thumb" data-case="${g.case}" title="Click to open ${g.case} in the case viewer">
          <img src="data/gradcam/${g.case}.png" alt="Grad-CAM overlay for ${g.case}" loading="lazy" />
          <div class="meta">${g.case} · ${g.label} <span class="iou">IoU ${g.iou.toFixed(2)}</span></div>
        </div>
      `).join("");
      gallery.querySelectorAll(".gradcam-thumb").forEach((el) => {
        el.addEventListener("click", () => {
          const id = el.dataset.case;
          if (!id) return;
          document.dispatchEvent(new CustomEvent("braintt:openCase", { detail: id }));
          const viewer = document.getElementById("viewer");
          if (viewer) viewer.scrollIntoView({ behavior: "smooth" });
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
