// Robustness Lab — 5 plots:
//   1) live Gaussian-noise degradation with draggable marker
//   2) modality-dropout staircase (with vs without synthesis)
//   3) cross-vendor AUC heatmap (DOM grid, no Plotly — fast + no JS deps)
//   4) FGSM ε sweep (lin-log axes)
//   5) Sample efficiency curve (10 %–100 % training data)

(function () {
  const section = document.getElementById("robustness");
  if (!section) return;

  BrainTT.lazy(section, async () => {
    const [{ metrics }, Plotly] = await Promise.all([
      BrainTT.dataPromise,
      BrainTT.ensurePlotly(),
    ]);
    const { axis, layout, config, colors } = BrainTT.plotly;
    const R = metrics.robustness;

    // ===========================================================
    // 1) Noise degradation (interactive)
    // ===========================================================
    const slider = document.getElementById("noiseSlider");
    const sliderVal = document.getElementById("noiseSliderVal");

    const noiseColors = {
      "BrainTT (Ours)": colors.cyan,
      "Swin-UNETR":     colors.violet,
      "ResNet10":       colors.gold,
    };
    const noiseTraces = Object.entries(R.noise)
      .filter(([k]) => k !== "sigma")
      .map(([name, vals]) => ({
        x: R.noise.sigma,
        y: vals,
        name,
        mode: "lines",
        line: { color: noiseColors[name], width: name === "BrainTT (Ours)" ? 3 : 2, shape: "spline" },
        hovertemplate: "<b>%{fullData.name}</b><br>σ %{x:.2f}  |  AUC %{y:.3f}<extra></extra>",
      }));

    // Marker dot per model — updated in the slider handler
    const markerTrace = (name) => ({
      x: [0], y: [R.noise[name][0]],
      mode: "markers",
      marker: { size: 12, color: noiseColors[name], line: { color: "#050816", width: 2 }, symbol: "diamond" },
      name: `${name} · @σ`,
      showlegend: false,
      hovertemplate: `<b>${name}</b><br>σ %{x:.2f}  |  AUC %{y:.3f}<extra></extra>`,
    });
    const markers = ["BrainTT (Ours)", "Swin-UNETR", "ResNet10"].map(markerTrace);

    // vertical reference line
    const vlineShape = {
      type: "line", x0: 0, x1: 0, y0: 0.4, y1: 1,
      line: { color: "rgba(34,211,238,0.45)", width: 1, dash: "dot" },
    };

    Plotly.newPlot("noisePlot", [...noiseTraces, ...markers], layout({
      xaxis: { ...axis, title: "Gaussian noise σ", range: [0, 0.5] },
      yaxis: { ...axis, title: "AUC", range: [0.4, 0.95] },
      shapes: [vlineShape],
    }), config);

    function interp(xs, ys, x) {
      // linear interp in pre-sorted xs
      if (x <= xs[0]) return ys[0];
      if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
      for (let i = 1; i < xs.length; i++) {
        if (x <= xs[i]) {
          const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
          return ys[i - 1] * (1 - t) + ys[i] * t;
        }
      }
      return ys[ys.length - 1];
    }

    const updateMarker = BrainTT.rafDebounce(() => {
      const s = parseFloat(slider.value);
      sliderVal.textContent = s.toFixed(2);
      const updX = []; const updY = [];
      const order = ["BrainTT (Ours)", "Swin-UNETR", "ResNet10"];
      order.forEach((name) => {
        updX.push([s]);
        updY.push([interp(R.noise.sigma, R.noise[name], s)]);
      });
      // markers are traces 3, 4, 5 (after the 3 curves)
      Plotly.restyle("noisePlot", { x: updX, y: updY }, [3, 4, 5]);
      Plotly.relayout("noisePlot", { "shapes[0].x0": s, "shapes[0].x1": s });
    });
    slider.addEventListener("input", updateMarker);

    // ===========================================================
    // 2) Modality dropout staircase
    // ===========================================================
    const md = R.modality_drop;
    const dropColors = {
      "BrainTT (Ours)":      colors.cyan,
      "BrainTT w/o synth":   colors.gold,
      "Swin-UNETR":          colors.violet,
      "ResNet10":            colors.magenta,
    };
    const dropTraces = Object.entries(md)
      .filter(([k]) => k !== "n_dropped")
      .map(([name, vals]) => ({
        x: md.n_dropped, y: vals,
        type: "bar",
        name,
        marker: {
          color: dropColors[name],
          line: { color: "#050816", width: 1 },
        },
        hovertemplate: "<b>%{fullData.name}</b><br>drop %{x}  |  AUC %{y:.3f}<extra></extra>",
      }));
    Plotly.newPlot("modDropPlot", dropTraces, layout({
      barmode: "group",
      bargap: 0.18,
      bargroupgap: 0.05,
      xaxis: { ...axis, title: "Modalities dropped", tickvals: [0, 1, 2, 3] },
      yaxis: { ...axis, title: "AUC", range: [0.4, 0.92] },
    }), config);

    // ===========================================================
    // 3) Cross-vendor matrix — DOM grid (no Plotly)
    // ===========================================================
    const vendorGrid = document.getElementById("vendorMatrix");
    if (vendorGrid && R.vendor_matrix) {
      const { vendors, auc } = R.vendor_matrix;
      // Find min/max for colour mapping
      let lo = Infinity, hi = -Infinity;
      auc.flat().forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      const lerp = (a, b, t) => a + (b - a) * t;
      const colorFor = (v) => {
        const t = (v - lo) / Math.max(hi - lo, 1e-6);
        const r = Math.round(lerp(244, 34, t));
        const g = Math.round(lerp(114, 211, t));
        const b = Math.round(lerp(182, 238, t));
        return `rgba(${r},${g},${b},${0.18 + 0.55 * t})`;
      };
      let html = `<div class="vm-corner">train ↓ / test →</div>`;
      vendors.forEach((v) => { html += `<div class="vm-collabel">${v}</div>`; });
      for (let i = 0; i < vendors.length; i++) {
        html += `<div class="vm-rowlabel">${vendors[i]}</div>`;
        for (let j = 0; j < vendors.length; j++) {
          const v = auc[i][j];
          const diag = i === j ? " diag" : "";
          html += `<div class="vm-cell${diag}" style="background:${colorFor(v)}" title="train ${vendors[i]} → test ${vendors[j]}: ${v.toFixed(3)}">${v.toFixed(3)}</div>`;
        }
      }
      vendorGrid.innerHTML = html;
    }

    // ===========================================================
    // 4) Adversarial FGSM
    // ===========================================================
    const fg = R.adversarial_fgsm;
    const fgsmColors = {
      "BrainTT (Ours)":     colors.cyan,
      "BrainTT (w/ TTA)":   colors.green,
      "ResNet10 baseline":  colors.magenta,
    };
    const fgsmTraces = Object.entries(fg)
      .filter(([k]) => k !== "epsilon")
      .map(([name, vals]) => ({
        x: fg.epsilon, y: vals,
        name,
        mode: "lines+markers",
        line: { color: fgsmColors[name], width: 2.5, shape: "spline" },
        marker: { size: 6, color: fgsmColors[name], line: { color: "#050816", width: 1.5 } },
        hovertemplate: "<b>%{fullData.name}</b><br>ε %{x}  |  AUC %{y:.3f}<extra></extra>",
      }));
    Plotly.newPlot("fgsmPlot", fgsmTraces, layout({
      xaxis: { ...axis, title: "FGSM ε", type: "linear" },
      yaxis: { ...axis, title: "AUC", range: [0.1, 0.95] },
    }), config);

    // ===========================================================
    // 5) Sample efficiency
    // ===========================================================
    const se = R.sample_efficiency;
    const seColors = {
      "BrainTT (Ours)": colors.cyan,
      "ResNet10":       colors.magenta,
      "nnU-Net":        colors.green,
    };
    const seTraces = Object.entries(se)
      .filter(([k]) => k !== "train_pct")
      .map(([name, vals]) => ({
        x: se.train_pct, y: vals,
        name,
        mode: "lines+markers",
        line: { color: seColors[name], width: 2.5, shape: "spline" },
        marker: { size: 8, color: seColors[name], line: { color: "#050816", width: 1.5 }, symbol: "diamond" },
        fill: name === "BrainTT (Ours)" ? "tozeroy" : "none",
        fillcolor: name === "BrainTT (Ours)" ? "rgba(34,211,238,0.08)" : undefined,
        hovertemplate: "<b>%{fullData.name}</b><br>train %{x}%  |  AUC %{y:.3f}<extra></extra>",
      }));
    Plotly.newPlot("samplePlot", seTraces, layout({
      xaxis: { ...axis, title: "Training fraction (%)", tickvals: se.train_pct },
      yaxis: { ...axis, title: "AUC", range: [0.55, 0.92] },
    }), config);
  });
})();
