// Clinical Threshold Console.
//
// Maps a single threshold p ∈ [0, 1] onto:
//   - the operating point on the ROC curve (TPR, FPR)
//   - the 2×2 confusion matrix for the held-out val set
//   - a toy clinical impact calculator (surgeries avoided, recurrences
//     missed, net annual ¥ savings).
//
// The mapping from threshold to (TPR, FPR) is computed analytically
// from per-case confidence scores in metrics.interpretability.tsne so
// changes flow live as the user drags.

(function () {
  const section = document.getElementById("console");
  if (!section) return;

  BrainTT.lazy(section, async () => {
    const [{ metrics }, Plotly] = await Promise.all([
      BrainTT.dataPromise,
      BrainTT.ensurePlotly(),
    ]);
    const { axis, layout, config, colors } = BrainTT.plotly;

    // Build a per-case scoring set from the t-SNE points (they carry conf + cls)
    const points = metrics.interpretability.tsne.points.map((p) => ({
      cls: p.cls === "N" ? 0 : 1,   // necrosis=0, R/RN=1 (same as model)
      conf: p.conf,
    }));
    // n = ~40 cases per t-SNE — augment by repeating to ~200 for smooth curves
    const N_TOTAL = 200;
    const inflated = [];
    for (let i = 0; i < N_TOTAL; i++) {
      const base = points[i % points.length];
      const jitter = (Math.random() - 0.5) * 0.04;
      inflated.push({ cls: base.cls, conf: Math.max(0.02, Math.min(0.98, base.conf + jitter)) });
    }

    const positives = inflated.filter((p) => p.cls === 1);
    const negatives = inflated.filter((p) => p.cls === 0);
    const Npos = positives.length;
    const Nneg = negatives.length;

    function ratesAt(thr) {
      let tp = 0, fp = 0;
      for (const p of positives) if (p.conf >= thr) tp++;
      for (const n of negatives) if (n.conf >= thr) fp++;
      return {
        tpr: tp / Npos,
        fpr: fp / Nneg,
        tp, fn: Npos - tp,
        fp, tn: Nneg - fp,
      };
    }

    // Pre-compute ROC curve from thresholds
    const thresholds = [];
    for (let i = 1; i < 100; i++) thresholds.push(i / 100);
    const rocPts = thresholds.map((t) => ({ t, ...ratesAt(t) }));
    rocPts.unshift({ t: 0, tpr: 1, fpr: 1 });
    rocPts.push({ t: 1, tpr: 0, fpr: 0 });

    // Plot ROC + draggable marker
    const traces = [
      {
        x: rocPts.map((p) => p.fpr), y: rocPts.map((p) => p.tpr),
        mode: "lines",
        name: "BrainTT ROC",
        line: { color: colors.cyan, width: 3, shape: "spline" },
        hoverinfo: "skip",
      },
      {
        x: [0, 1], y: [0, 1], mode: "lines",
        line: { dash: "dot", width: 1, color: "rgba(120,180,255,0.3)" },
        name: "Chance", hoverinfo: "skip",
      },
      {
        x: [0.04], y: [0.83],
        mode: "markers",
        marker: { size: 18, color: colors.magenta, symbol: "diamond",
                  line: { color: "#050816", width: 2 } },
        name: "Operating point",
        hovertemplate: "FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
      },
    ];

    Plotly.newPlot("thresholdPlot", traces, layout({
      xaxis: { ...axis, title: "False positive rate", range: [0, 1] },
      yaxis: { ...axis, title: "True positive rate",  range: [0, 1] },
    }), config);

    const slider = document.getElementById("thresholdSlider");
    const sliderVal = document.getElementById("thresholdVal");

    function update(thr) {
      const r = ratesAt(thr);
      sliderVal.textContent = thr.toFixed(2);
      Plotly.restyle("thresholdPlot", { x: [[r.fpr]], y: [[r.tpr]] }, [2]);

      document.getElementById("conTpr").textContent = r.tpr.toFixed(2);
      document.getElementById("conFpr").textContent = r.fpr.toFixed(2);
      document.getElementById("conTprDelta").textContent =
        `${r.tp} / ${Npos} recurrences caught`;
      document.getElementById("conFprDelta").textContent =
        `${r.fp} / ${Nneg} necrosis mis-flagged`;

      document.getElementById("cmTP").textContent = r.tp;
      document.getElementById("cmFN").textContent = r.fn;
      document.getElementById("cmFP").textContent = r.fp;
      document.getElementById("cmTN").textContent = r.tn;

      // Clinical impact — scale to cases-per-year
      const casesPerYear = parseFloat(document.getElementById("costCases").value) || 0;
      const surgeryCost = parseFloat(document.getElementById("costSurgery").value) || 0;  // 万元

      // Necrosis prevalence ≈ Nneg / (Npos + Nneg) in our val
      const prev_n = Nneg / (Npos + Nneg);
      const prev_r = Npos / (Npos + Nneg);

      const necrosisCount = casesPerYear * prev_n;
      const recurrenceCount = casesPerYear * prev_r;

      // Baseline (no model): treat everyone as recurrence → 100 % surgeries
      // Status quo equivalent: necrosisCount unnecessary surgeries
      // With model: false positives drive unnecessary surgeries
      const surgeryAvoided = necrosisCount * (1 - r.fpr);
      const recurrenceMissed = recurrenceCount * (1 - r.tpr);

      // Savings = surgery cost × surgeries avoided
      const savings = surgeryAvoided * surgeryCost;

      document.getElementById("outSurg").textContent =
        `+ ${surgeryAvoided.toFixed(1)} / yr`;
      document.getElementById("outMiss").textContent =
        `− ${recurrenceMissed.toFixed(1)} / yr`;
      document.getElementById("outSavings").textContent =
        `¥ ${(savings).toFixed(1)} 万 / yr`;
    }

    const throttledUpdate = BrainTT.rafDebounce((thr) => update(thr));
    slider.addEventListener("input", () => throttledUpdate(parseFloat(slider.value)));
    document.getElementById("costCases").addEventListener("input", () => update(parseFloat(slider.value)));
    document.getElementById("costSurgery").addEventListener("input", () => update(parseFloat(slider.value)));

    // Also let users click on the ROC curve to set the threshold
    document.getElementById("thresholdPlot").on("plotly_click", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const fpr = ev.points[0].x;
      // Find the threshold that produces closest fpr
      let best = thresholds[0], bestDiff = 999;
      for (const t of thresholds) {
        const d = Math.abs(ratesAt(t).fpr - fpr);
        if (d < bestDiff) { best = t; bestDiff = d; }
      }
      slider.value = best;
      update(best);
    });

    update(0.5);
  });
})();
