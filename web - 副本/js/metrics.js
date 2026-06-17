// Performance dashboard — ROC, Pareto, parallel-coords, Sankey, leaderboard.
// All Plotly initialisation is gated behind BrainTT.lazy so the library
// isn't even fetched until the user scrolls into the section.

(function () {
  const section = document.getElementById("performance");
  if (!section) return;

  BrainTT.lazy(section, async () => {
    const [{ metrics }, Plotly] = await Promise.all([
      BrainTT.dataPromise,
      BrainTT.ensurePlotly(),
    ]);
    const { axis, layout, config, colors } = BrainTT.plotly;

    // ---- ROC ------------------------------------------------------
    const rocColors = {
      "BrainTT (Ours)": colors.cyan,
      "Swin-UNETR":     colors.violet,
      "MResNet":        colors.magenta,
      "nnU-Net":        colors.green,
      "ResNet10":       colors.gold,
    };
    const rocTraces = Object.entries(metrics.roc).map(([name, pts]) => ({
      x: pts.fpr, y: pts.tpr,
      name,
      mode: "lines",
      line: { width: name === "BrainTT (Ours)" ? 3 : 2, color: rocColors[name] || "#95a8c8", shape: "spline" },
      hovertemplate: "<b>%{fullData.name}</b><br>FPR %{x:.2f}  |  TPR %{y:.2f}<extra></extra>",
    }));
    rocTraces.push({
      x: [0, 1], y: [0, 1], mode: "lines",
      line: { dash: "dot", width: 1, color: "rgba(120,180,255,0.3)" },
      name: "Chance", showlegend: true, hoverinfo: "skip",
    });
    Plotly.newPlot("rocPlot", rocTraces, layout({
      xaxis: { ...axis, title: "False positive rate", range: [0, 1] },
      yaxis: { ...axis, title: "True positive rate", range: [0, 1] },
    }), config);

    // ---- Pareto scatter ------------------------------------------
    const models = metrics.models;
    const pareto = [{
      x: models.map((m) => m.params),
      y: models.map((m) => m.sens),
      text: models.map((m) => m.name),
      mode: "markers+text",
      type: "scatter",
      marker: {
        size: models.map((m) => 8 + Math.sqrt(m.auc) * 22),
        color: models.map((m) => m.highlight ? colors.cyan : colors.magenta),
        opacity: models.map((m) => m.highlight ? 1 : 0.55),
        line: { color: "#050816", width: 2 },
      },
      textposition: "top center",
      textfont: { ...BrainTT.plotly.baseFont, color: colors.text, size: 9 },
      hovertemplate: "<b>%{text}</b><br>params %{x:.2f} M  |  sens %{y:.2f}<extra></extra>",
    }];
    Plotly.newPlot("paretoPlot", pareto, layout({
      showlegend: false,
      xaxis: { ...axis, title: "Parameters (M)", type: "log" },
      yaxis: { ...axis, title: "Sensitivity on necrosis", range: [0.25, 1.0] },
    }), config);

    // ---- Parallel coordinates ------------------------------------
    const pcModels = models.filter((m) => m.dice > 0); // only segmentation-capable models
    const parallel = [{
      type: "parcoords",
      line: {
        color: pcModels.map((m) => m.highlight ? 1 : 0),
        colorscale: [
          [0, colors.magenta],
          [1, colors.cyan],
        ],
        showscale: false,
        width: 2,
      },
      dimensions: [
        { range: [0.7, 0.95],  label: "AUC",       values: pcModels.map((m) => m.auc) },
        { range: [0.4, 0.9],   label: "Sens",      values: pcModels.map((m) => m.sens) },
        { range: [0.85, 1.0],  label: "Spec",      values: pcModels.map((m) => m.spec) },
        { range: [0.7, 0.85],  label: "Dice",      values: pcModels.map((m) => m.dice) },
        { range: [0, 100],     label: "Params M",  values: pcModels.map((m) => m.params) },
      ],
    }];
    Plotly.newPlot("parallelPlot", parallel, layout({
      margin: { l: 50, r: 30, t: 30, b: 40 },
      showlegend: false,
    }), config);

    // ---- Sankey: true class → predicted --------------------------
    const sk = metrics.interpretability.sankey;
    const sankey = [{
      type: "sankey",
      orientation: "h",
      arrangement: "snap",
      node: {
        pad: 20,
        thickness: 18,
        line: { color: "rgba(120,180,255,0.25)", width: 1 },
        label: sk.nodes,
        color: [colors.magenta, colors.cyan, colors.violet, colors.magenta, colors.cyan, colors.gold],
      },
      link: {
        source: sk.links.map((l) => l.source),
        target: sk.links.map((l) => l.target),
        value:  sk.links.map((l) => l.value),
        color:  sk.links.map((l) => {
          const cs = [colors.magenta, colors.cyan, colors.violet][l.source % 3];
          // Convert hex to rgba with low alpha for ribbons
          const r = parseInt(cs.slice(1, 3), 16);
          const g = parseInt(cs.slice(3, 5), 16);
          const b = parseInt(cs.slice(5, 7), 16);
          return `rgba(${r},${g},${b},0.35)`;
        }),
      },
    }];
    Plotly.newPlot("sankeyPlot", sankey, layout({
      margin: { l: 10, r: 10, t: 10, b: 10 },
      showlegend: false,
      font: { ...BrainTT.plotly.baseFont, color: colors.text, size: 12 },
    }), config);

    // ---- Leaderboard table ---------------------------------------
    const tbody = document.querySelector("#leaderboardTable tbody");
    if (tbody) {
      const famClass = (f) => {
        const s = (f || "").toLowerCase().replace(/[^a-z]/g, "-");
        return `fam-${s}`;
      };
      tbody.innerHTML = models
        .slice()
        .sort((a, b) => b.auc - a.auc)
        .map((m) => `
          <tr class="${m.highlight ? "highlight" : ""}">
            <td>${m.name}</td>
            <td class="${famClass(m.family)}">${m.family || "—"}</td>
            <td>${m.auc.toFixed(3)}</td>
            <td>${m.sens.toFixed(3)}</td>
            <td>${m.spec.toFixed(3)}</td>
            <td>${m.dice > 0 ? m.dice.toFixed(3) : "—"}</td>
            <td>${m.params}</td>
            <td>${m.flops_g}</td>
          </tr>
        `).join("");
    }
  });
})();
