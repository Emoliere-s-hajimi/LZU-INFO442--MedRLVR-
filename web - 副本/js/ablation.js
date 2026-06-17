// Ablation playground: 3 toggles → live readout from metrics.ablation rows.

(async function () {
  const { metrics } = await BrainTT.dataPromise;
  const rows = metrics.ablation.rows;

  const toggles = document.querySelectorAll(".toggle-prior");
  const auc = document.getElementById("abAuc");
  const sens = document.getElementById("abSens");
  const params = document.getElementById("abParams");
  const aucD = document.getElementById("abAucDelta");
  const sensD = document.getElementById("abSensDelta");
  const paramsD = document.getElementById("abParamsDelta");
  const config = document.getElementById("abConfig");
  const explain = document.getElementById("abExplain");
  if (!auc) return;

  const state = { modality: true, topology: true, anatomy: true };

  const baseline = rows.find((r) => r.modality && r.topology && r.anatomy);

  function findRow(s) {
    return rows.find((r) =>
      r.modality === s.modality && r.topology === s.topology && r.anatomy === s.anatomy
    );
  }

  function nearestRow(s) {
    // Fall back to closest match (matching ON-count) if exact missing
    const exact = findRow(s);
    if (exact) return exact;
    const onCount = (r) => [r.modality, r.topology, r.anatomy].filter(Boolean).length;
    const target = onCount(s);
    return rows.slice().sort((a, b) =>
      Math.abs(onCount(a) - target) - Math.abs(onCount(b) - target)
    )[0];
  }

  function renderDelta(curr, base, el, lowerIsBetter = false) {
    const d = curr - base;
    el.textContent = (d > 0 ? "▲ " : d < 0 ? "▼ " : "— ") +
      (Math.abs(d) < 1e-4 ? "baseline" : Math.abs(d).toFixed(3));
    el.classList.remove("up", "down");
    if (Math.abs(d) < 1e-4) return;
    const better = lowerIsBetter ? d < 0 : d > 0;
    el.classList.add(better ? "up" : "down");
  }

  function render() {
    const r = nearestRow(state);
    auc.textContent = r.auc.toFixed(3);
    sens.textContent = r.sens.toFixed(2);
    params.textContent = r.params_m.toFixed(2);
    renderDelta(r.auc, baseline.auc, aucD);
    renderDelta(r.sens, baseline.sens, sensD);
    renderDelta(r.params_m, baseline.params_m, paramsD, true);
    config.textContent = r.config;

    const onPriors = [];
    if (state.modality) onPriors.push("Modality");
    if (state.topology) onPriors.push("Topology");
    if (state.anatomy) onPriors.push("Anatomy");
    const summary = onPriors.length === 0
      ? "Vanilla 3-D U-Net. Topology and lobe-conditioning are gone; sensitivity collapses on necrosis."
      : onPriors.length === 3
        ? "The full BrainTT configuration. All three priors are active and reinforcing each other."
        : `Running with ${onPriors.join(" + ")} active. Drop in AUC reflects what the missing prior was contributing.`;
    explain.textContent = summary;
  }

  toggles.forEach((t) => {
    t.addEventListener("click", () => {
      const id = t.dataset.prior;
      state[id] = !state[id];
      t.classList.toggle("on", state[id]);
      render();
    });
  });

  // Failure modes browser
  const grid = document.getElementById("failureGrid");
  if (grid) {
    grid.innerHTML = metrics.failure_modes.map((f) => `
      <div class="failure-card">
        <span class="freq">${f.frequency}</span>
        <h4>${f.title}</h4>
        <p>${f.summary}</p>
        <div class="mitigation">${f.mitigation}</div>
      </div>
    `).join("");
  }

  render();
})();
