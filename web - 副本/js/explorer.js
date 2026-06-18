// Cohort Explorer — 322 cases as a lasso-able scatter.
//
// - Axes switchable via two <select>s
// - Class toggles filter by N / R / RN
// - Lasso (Plotly's "select" mode) drives the per-selection read-out
// - Hovering a dot fills the side card with that case's biomarkers
// - Click → open the Case Viewer if the id matches one of the 4 demo cases

(function () {
  const section = document.getElementById("explorer");
  if (!section) return;

  BrainTT.lazy(section, async () => {
    const [Plotly, raw] = await Promise.all([
      BrainTT.ensurePlotly(),
      fetch("data/cohort_explorer.json").then((r) => r.json()),
    ]);
    const { axis, layout, config, colors } = BrainTT.plotly;
    const points = raw.points;
    const axisMeta = raw.axes;

    const xPick = document.getElementById("xAxisPick");
    const yPick = document.getElementById("yAxisPick");
    const resetBtn = document.getElementById("explorerReset");
    const classToggles = section.querySelectorAll(".class-toggle");

    const state = {
      x: xPick.value,
      y: yPick.value,
      classes: new Set(["R", "N", "RN"]),
      selectedIds: null, // null = whole cohort
    };

    const CLASS_COLOR = { R: colors.magenta, N: colors.cyan, RN: colors.violet };
    const CLASS_FULL = { R: "Recurrence", N: "Necrosis", RN: "Mixed" };

    function pointsByClass() {
      const byCls = { R: [], N: [], RN: [] };
      for (const p of points) {
        if (!state.classes.has(p.label_short)) continue;
        byCls[p.label_short].push(p);
      }
      return byCls;
    }

    function buildTraces() {
      const byCls = pointsByClass();
      const traces = [];
      for (const cls of ["R", "N", "RN"]) {
        const pts = byCls[cls];
        traces.push({
          x: pts.map((p) => p[state.x]),
          y: pts.map((p) => p[state.y]),
          customdata: pts.map((p) => p.id),
          text: pts.map((p) => p.id),
          name: CLASS_FULL[cls],
          mode: "markers",
          type: "scattergl",
          marker: {
            size: 7,
            color: CLASS_COLOR[cls],
            opacity: 0.78,
            line: { color: "#050816", width: 0.8 },
            symbol: cls === "RN" ? "diamond" : "circle",
          },
          hovertemplate:
            `<b>%{customdata}</b><br>` +
            `${axisMeta[state.x].label} = %{x}<br>` +
            `${axisMeta[state.y].label} = %{y}<extra></extra>`,
        });
      }
      return traces;
    }

    function lay() {
      return layout({
        dragmode: "lasso",
        margin: { l: 60, r: 20, t: 22, b: 52 },
        xaxis: { ...axis, title: axisMeta[state.x].label },
        yaxis: { ...axis, title: axisMeta[state.y].label },
        legend: { ...layout({}).legend, orientation: "h", x: 0, y: 1.06, yanchor: "bottom" },
      });
    }

    Plotly.newPlot("explorerPlot", buildTraces(), lay(), {
      ...config, modeBarButtonsToAdd: ["lasso2d", "select2d"], displayModeBar: false,
    });

    function redraw() {
      Plotly.react("explorerPlot", buildTraces(), lay(), config);
      // After react, also clear any prior lasso selection visual
      state.selectedIds = null;
      updateReadout();
    }

    xPick.addEventListener("change", () => { state.x = xPick.value; redraw(); });
    yPick.addEventListener("change", () => { state.y = yPick.value; redraw(); });

    classToggles.forEach((t) => {
      t.addEventListener("click", () => {
        const cls = t.dataset.cls;
        const on = !t.classList.contains("on");
        // Always keep at least one class enabled
        const next = new Set(state.classes);
        if (on) next.add(cls); else next.delete(cls);
        if (next.size === 0) return;
        t.classList.toggle("on", on);
        state.classes = next;
        redraw();
      });
    });

    resetBtn.addEventListener("click", () => {
      state.selectedIds = null;
      Plotly.restyle("explorerPlot", { selectedpoints: [null, null, null] });
      updateReadout();
    });

    const plot = document.getElementById("explorerPlot");
    plot.on("plotly_selected", (ev) => {
      if (!ev || !ev.points || ev.points.length === 0) {
        state.selectedIds = null;
      } else {
        state.selectedIds = new Set(ev.points.map((p) => p.customdata));
      }
      updateReadout();
    });
    plot.on("plotly_deselect", () => { state.selectedIds = null; updateReadout(); });
    plot.on("plotly_hover", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const id = ev.points[0].customdata;
      const p = points.find((x) => x.id === id);
      if (p) renderHover(p);
    });
    plot.on("plotly_click", (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const id = ev.points[0].customdata;
      const demoIds = ["N_005", "R_148", "RN_003", "RN_044"];
      if (demoIds.includes(id)) {
        // Trigger Case Viewer to open this case
        document.dispatchEvent(new CustomEvent("braintt:openCase", { detail: id }));
        document.getElementById("viewer").scrollIntoView({ behavior: "smooth" });
      }
    });

    function renderHover(p) {
      const hov = document.getElementById("exHover");
      hov.innerHTML = `
        <div class="row"><span class="k">id</span><span class="v" style="color:${CLASS_COLOR[p.label_short]}">${p.id}</span></div>
        <div class="row"><span class="k">class</span><span class="v">${CLASS_FULL[p.label_short]}</span></div>
        <div class="row"><span class="k">Euler χ</span><span class="v">${p.chi}</span></div>
        <div class="row"><span class="k">T1ce ratio</span><span class="v">${p.t1ce_ratio.toFixed(2)}</span></div>
        <div class="row"><span class="k">volume</span><span class="v">${p.volume_ml.toFixed(1)} mL</span></div>
        <div class="row"><span class="k">sphericity</span><span class="v">${p.sphericity.toFixed(2)}</span></div>
        <div class="row"><span class="k">confidence</span><span class="v">${p.confidence.toFixed(2)}</span></div>
        <div class="row"><span class="k">predicted</span><span class="v" style="color:${p.correct ? 'var(--green)' : 'var(--red)'}">${p.predicted}${p.correct ? ' ✓' : ' ✗'}</span></div>
      `;
    }

    function estimateAuc(subset) {
      // Pairwise AUC: count concordant pairs across class boundaries.
      // Class 1 = recurrence; class 0 = necrosis. Mixed treated as 1 (matches model).
      const pos = []; const neg = [];
      for (const p of subset) {
        if (p.label_short === "N") neg.push(p.confidence);
        else pos.push(p.confidence); // R + RN
      }
      if (!pos.length || !neg.length) return NaN;
      let cnt = 0, ties = 0;
      for (const a of pos) {
        for (const b of neg) {
          if (a > b) cnt++;
          else if (a === b) ties++;
        }
      }
      return (cnt + 0.5 * ties) / (pos.length * neg.length);
    }

    function median(arr) {
      if (!arr.length) return NaN;
      const s = [...arr].sort((a, b) => a - b);
      const m = s.length >> 1;
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    }

    function updateReadout() {
      const subset = state.selectedIds
        ? points.filter((p) => state.classes.has(p.label_short) && state.selectedIds.has(p.id))
        : points.filter((p) => state.classes.has(p.label_short));

      const n = subset.length;
      const patients = new Set(subset.map((p) => p.patient)).size;
      document.getElementById("exCount").textContent = n;
      document.getElementById("exPatientCount").textContent = `${patients} unique patient${patients === 1 ? "" : "s"}`;

      // Class bars
      const counts = { R: 0, N: 0, RN: 0 };
      subset.forEach((p) => { counts[p.label_short]++; });
      const max = Math.max(1, ...Object.values(counts));
      const bars = ["R", "N", "RN"].map((cls) => `
        <div class="exclass-bar" style="--c: ${CLASS_COLOR[cls]}">
          <span class="lbl">${cls}</span>
          <span class="track"><span class="fill" style="transform: scaleX(${counts[cls] / max})"></span></span>
          <span class="val">${counts[cls]}</span>
        </div>
      `).join("");
      document.getElementById("exClassBars").innerHTML = bars;

      // Metrics
      const auc = estimateAuc(subset);
      document.getElementById("exAuc").textContent = isNaN(auc) ? "—" : auc.toFixed(3);
      document.getElementById("exMedChi").textContent = n ? Math.round(median(subset.map((p) => p.chi))) : "—";
      document.getElementById("exMedRatio").textContent = n ? median(subset.map((p) => p.t1ce_ratio)).toFixed(2) : "—";
      const miss = subset.filter((p) => p.missing.length > 0).length;
      document.getElementById("exMissPct").textContent = n ? `${Math.round((miss / n) * 100)}%` : "—";
    }

    updateReadout();
  });
})();
