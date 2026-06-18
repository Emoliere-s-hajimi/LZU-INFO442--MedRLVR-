// Architecture Sandbox — design-your-own model + estimate metrics.
//
// The estimator is intentionally lightweight: it interpolates from the
// real leaderboard / ablation rows in metrics.json by combining a
// backbone baseline with additive prior-on contributions and small
// scalers for loss / data / dropout choices. Marked as an estimate in
// the UI; the published rows still anchor everything.

(function () {
  const section = document.getElementById("sandbox");
  if (!section) return;

  BrainTT.lazy(section, async () => {
    const [{ metrics }, Plotly] = await Promise.all([
      BrainTT.dataPromise,
      BrainTT.ensurePlotly(),
    ]);
    const { axis, layout, config, colors } = BrainTT.plotly;

    // -------- Backbones -------------------------------------------
    // Each row: starting AUC / Sens / Params *without* the priors,
    // pulled from the leaderboard ablation row "All priors off" plus
    // adjustments per architecture (closer to the published numbers).
    const BACKBONES = {
      unet:      { name: "U-Net",       auc0: 0.750, sens0: 0.50, params0: 7.5  },
      swin:      { name: "Swin-UNETR",  auc0: 0.812, sens0: 0.60, params0: 62.2 },
      nnunet:    { name: "nnU-Net",     auc0: 0.820, sens0: 0.68, params0: 31.2 },
      transunet: { name: "TransUNet",   auc0: 0.785, sens0: 0.57, params0: 96.7 },
      mamba:     { name: "V-Mamba",     auc0: 0.778, sens0: 0.55, params0: 27.5 },
      medsam:    { name: "MedSAM (FT)", auc0: 0.768, sens0: 0.58, params0: 93.6 },
    };

    const PRIORS = {
      modality: { auc: 0.042, sens: 0.09, params: 0.02 },
      topology: { auc: 0.028, sens: 0.06, params: 0.01 },
      anatomy:  { auc: 0.019, sens: 0.02, params: 0.02 },
    };

    const LOSS_FLAGS = {
      focal:   { auc: 0.012, sens: 0.04 },
      dice:    { auc: 0.018, sens: 0.03 },
      nesting: { auc: 0.008, sens: 0.02 },
      chi:     { auc: 0.014, sens: 0.05 },
    };

    const state = {
      backbone: "unet",
      priors: { modality: true, topology: true, anatomy: true },
      flags:  { focal: true, dice: true, nesting: false, chi: false },
      dataPct: 100,
      dropP: 0.15,
    };

    function estimate() {
      const bb = BACKBONES[state.backbone];
      let auc = bb.auc0;
      let sens = bb.sens0;
      let params = bb.params0;

      // Prior contributions, with mild saturation when stacking
      let nOn = 0;
      ["modality", "topology", "anatomy"].forEach((p) => {
        if (state.priors[p]) {
          const att = 1 - 0.12 * nOn;          // mild diminishing return
          auc += PRIORS[p].auc * att;
          sens += PRIORS[p].sens * att;
          params += PRIORS[p].params;
          nOn++;
        }
      });
      // The UNet variant of BrainTT shrinks the backbone aggressively
      if (state.backbone === "unet" && nOn === 3) params = 0.15;

      // Loss flags
      Object.entries(state.flags).forEach(([k, on]) => {
        if (on) {
          auc += LOSS_FLAGS[k].auc * 0.6;
          sens += LOSS_FLAGS[k].sens * 0.6;
        }
      });
      if (state.backbone === "unet" && state.flags.focal && state.flags.dice && state.flags.nesting && state.flags.chi && nOn === 3) {
        auc = 0.890; sens = 0.83; params = 0.15;   // exact BrainTT recipe
      }

      // Training-data discount — pulled from sample_efficiency curve
      const se = metrics.robustness.sample_efficiency;
      const idx = se.train_pct.indexOf(state.dataPct);
      if (idx >= 0) {
        const dataAuc = se["BrainTT (Ours)"][idx];
        const fullAuc = se["BrainTT (Ours)"][se.train_pct.length - 1];
        auc *= dataAuc / fullAuc;
      } else {
        // linear interp
        const xs = se.train_pct, ys = se["BrainTT (Ours)"];
        let t = ys[ys.length - 1];
        for (let i = 1; i < xs.length; i++) {
          if (state.dataPct <= xs[i]) {
            const a = (state.dataPct - xs[i - 1]) / (xs[i] - xs[i - 1]);
            t = ys[i - 1] * (1 - a) + ys[i] * a;
            break;
          }
        }
        auc *= t / ys[ys.length - 1];
      }

      // Modality dropout cost (rough)
      const dropPenalty = state.dropP * 0.10 - 0.015;  // tiny bonus near 0.15
      auc -= Math.max(0, dropPenalty);

      // Clamp
      auc = Math.max(0.55, Math.min(0.94, auc));
      sens = Math.max(0.25, Math.min(0.95, sens));
      params = Math.max(0.10, params);

      return { auc, sens, params };
    }

    function renderRecipe() {
      const priors = Object.entries(state.priors).filter(([, v]) => v).map(([k]) => k).join("+");
      const flags = Object.entries(state.flags).filter(([, v]) => v).map(([k]) => k).join("+");
      const txt = `${BACKBONES[state.backbone].name} · priors[${priors || "none"}] · loss[${flags || "none"}]`;
      document.getElementById("sbRecipe").textContent = txt;
    }

    // Baseline = the published BrainTT result for delta display
    const BASELINE = { auc: 0.890, sens: 0.83, params: 0.15 };

    function renderDelta(curr, base, el, lowerIsBetter = false) {
      const d = curr - base;
      el.textContent = (d > 0 ? "▲ " : d < 0 ? "▼ " : "— ") +
        (Math.abs(d) < 1e-4 ? "ours" : Math.abs(d).toFixed(3));
      el.classList.remove("up", "down");
      if (Math.abs(d) < 1e-4) return;
      const better = lowerIsBetter ? d < 0 : d > 0;
      el.classList.add(better ? "up" : "down");
    }

    function paint(est) {
      document.getElementById("sbAuc").textContent = est.auc.toFixed(3);
      document.getElementById("sbSens").textContent = est.sens.toFixed(2);
      document.getElementById("sbParams").textContent = est.params.toFixed(2);
      renderDelta(est.auc, BASELINE.auc, document.getElementById("sbAucDelta"));
      renderDelta(est.sens, BASELINE.sens, document.getElementById("sbSensDelta"));
      renderDelta(est.params, BASELINE.params, document.getElementById("sbParamsDelta"), true);
      renderRecipe();
      drawPareto(est);
    }

    function drawPareto(est) {
      const models = metrics.models;
      const tracesBase = [{
        x: models.map((m) => m.params),
        y: models.map((m) => m.sens),
        text: models.map((m) => m.name),
        mode: "markers",
        type: "scatter",
        marker: {
          size: 11,
          color: models.map((m) => m.highlight ? colors.cyan : colors.magenta),
          opacity: models.map((m) => m.highlight ? 1 : 0.45),
          line: { color: "#050816", width: 1.5 },
        },
        name: "Published",
        hovertemplate: "<b>%{text}</b><br>params %{x:.2f} M  |  sens %{y:.2f}<extra></extra>",
      }, {
        x: [est.params], y: [est.sens],
        text: ["Your model"],
        mode: "markers+text",
        type: "scatter",
        marker: {
          size: 28,
          color: colors.gold,
          symbol: "diamond",
          opacity: 0.95,
          line: { color: "#050816", width: 2 },
        },
        textposition: "top center",
        textfont: { color: colors.gold, family: "JetBrains Mono", size: 11 },
        name: "Your sandbox model",
        hovertemplate: "<b>YOUR MODEL</b><br>params %{x:.2f} M  |  sens %{y:.2f}<extra></extra>",
      }];
      Plotly.react("sandboxPlot", tracesBase, layout({
        showlegend: false,
        margin: { l: 60, r: 20, t: 12, b: 50 },
        xaxis: { ...axis, title: "Parameters (M)", type: "log" },
        yaxis: { ...axis, title: "Sensitivity on necrosis", range: [0.25, 0.95] },
      }), config);
    }

    // Wire-up
    section.querySelectorAll(".sb-radio").forEach((r) => {
      r.addEventListener("click", () => {
        section.querySelectorAll(".sb-radio").forEach((x) => x.classList.remove("on"));
        r.classList.add("on");
        state.backbone = r.dataset.val;
        paint(estimate());
      });
    });

    section.querySelectorAll(".sb-toggle").forEach((t) => {
      t.addEventListener("click", () => {
        t.classList.toggle("on");
        const on = t.classList.contains("on");
        if (t.dataset.prior) state.priors[t.dataset.prior] = on;
        if (t.dataset.flag)  state.flags[t.dataset.flag]   = on;
        paint(estimate());
      });
    });

    const dataPct = document.getElementById("sbDataPct");
    const dataPctVal = document.getElementById("sbDataPctVal");
    dataPct.addEventListener("input", () => {
      state.dataPct = parseInt(dataPct.value, 10);
      dataPctVal.textContent = `${state.dataPct}%`;
      paint(estimate());
    });

    const dropP = document.getElementById("sbDropP");
    const dropPVal = document.getElementById("sbDropPVal");
    dropP.addEventListener("input", () => {
      state.dropP = parseFloat(dropP.value);
      dropPVal.textContent = state.dropP.toFixed(2);
      paint(estimate());
    });

    paint(estimate());
  });
})();
