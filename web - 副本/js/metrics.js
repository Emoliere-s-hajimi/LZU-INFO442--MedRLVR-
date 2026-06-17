// Plotly dashboards: ROC, Pareto (sens vs params), leaderboard table.

(async function () {
  const { metrics } = await BrainTT.dataPromise;

  // Shared layout primitives
  const baseFont = { family: "JetBrains Mono, ui-monospace, monospace", color: "#95a8c8", size: 11 };
  const axisStyle = {
    color: "#5a6c8b",
    gridcolor: "rgba(120, 180, 255, 0.07)",
    zerolinecolor: "rgba(120, 180, 255, 0.15)",
    linecolor: "rgba(120, 180, 255, 0.25)",
    tickfont: baseFont,
    titlefont: { ...baseFont, color: "#95a8c8" },
  };
  const baseLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: baseFont,
    margin: { l: 60, r: 20, t: 24, b: 50 },
    showlegend: true,
    legend: { font: baseFont, bgcolor: "rgba(0,0,0,0)", bordercolor: "rgba(120,180,255,0.15)", borderwidth: 1 },
  };

  // ---- ROC ---------------------------------------------------------
  const rocColors = {
    "BrainTT (Ours)": "#22d3ee",
    "MResNet":        "#f472b6",
    "ResNet10":       "#a78bfa",
    "DenseNet121":    "#facc15",
  };
  const rocTraces = Object.entries(metrics.roc).map(([name, pts]) => ({
    x: pts.fpr,
    y: pts.tpr,
    name,
    mode: "lines+markers",
    line: { width: name === "BrainTT (Ours)" ? 3 : 2, color: rocColors[name] || "#95a8c8", shape: "spline", smoothing: 0.5 },
    marker: { size: name === "BrainTT (Ours)" ? 7 : 5 },
    hovertemplate: "<b>%{fullData.name}</b><br>FPR %{x:.2f}  |  TPR %{y:.2f}<extra></extra>",
  }));
  // Random-chance diagonal
  rocTraces.push({
    x: [0, 1], y: [0, 1], mode: "lines",
    line: { dash: "dot", width: 1, color: "rgba(120,180,255,0.3)" },
    name: "Chance", showlegend: true, hoverinfo: "skip",
  });
  Plotly.newPlot("rocPlot", rocTraces, {
    ...baseLayout,
    xaxis: { ...axisStyle, title: "False positive rate", range: [0, 1] },
    yaxis: { ...axisStyle, title: "True positive rate", range: [0, 1] },
  }, { responsive: true, displayModeBar: false });

  // ---- Pareto scatter: sensitivity vs parameters -------------------
  const models = metrics.models;
  const pareto = [{
    x: models.map((m) => m.params),
    y: models.map((m) => m.sens),
    text: models.map((m) => m.name),
    mode: "markers+text",
    type: "scatter",
    marker: {
      size: models.map((m) => 10 + Math.sqrt(m.auc) * 24),
      color: models.map((m) => m.highlight ? "#22d3ee" : "#f472b6"),
      opacity: models.map((m) => m.highlight ? 1 : 0.55),
      line: { color: "#050816", width: 2 },
    },
    textposition: "top center",
    textfont: { ...baseFont, color: "#e6f0ff", size: 10 },
    hovertemplate: "<b>%{text}</b><br>params %{x:.2f} M  |  sens %{y:.2f}<extra></extra>",
  }];
  Plotly.newPlot("paretoPlot", pareto, {
    ...baseLayout,
    showlegend: false,
    xaxis: { ...axisStyle, title: "Parameters (M)", type: "log" },
    yaxis: { ...axisStyle, title: "Sensitivity on necrosis", range: [0.25, 1.0] },
  }, { responsive: true, displayModeBar: false });

  // ---- Leaderboard table -------------------------------------------
  const tbody = document.querySelector("#leaderboardTable tbody");
  if (tbody) {
    tbody.innerHTML = models
      .slice()
      .sort((a, b) => b.auc - a.auc)
      .map((m) => `
        <tr class="${m.highlight ? "highlight" : ""}">
          <td>${m.name}</td>
          <td>${m.auc.toFixed(3)}</td>
          <td>${m.acc.toFixed(3)}</td>
          <td>${m.sens.toFixed(3)}</td>
          <td>${m.spec.toFixed(3)}</td>
          <td>${m.params}</td>
        </tr>
      `).join("");
  }
})();
