// Case viewer — lazy-initialised. NiiVue is loaded on demand.

(function () {
  const section = document.getElementById("viewer");
  const tabsEl = document.getElementById("caseTabs");
  const infoEl = document.getElementById("caseInfo");
  const canvas = document.getElementById("niivueCanvas");
  const loading = document.getElementById("nvLoading");
  const toolbar = document.querySelector(".niivue-toolbar");
  if (!section || !tabsEl || !canvas) return;

  BrainTT.lazy(section, async () => {
    const data = await BrainTT.dataPromise;
    const cases = data.cases;

    let nv = null;
    let activeCaseId = null;
    let activeMod = "t1ce";
    let segOn = false;
    let multi = false;
    let pendingLoadToken = 0;

    function buildTabs() {
      tabsEl.innerHTML = cases.map((c, i) => `
        <div class="case-tab${i === 0 ? " active" : ""}" data-case="${c.id}">
          <span class="swatch" style="background:${c.color}"></span>
          <strong>${c.id}</strong> · ${c.label_short} · ${c.label}
        </div>
      `).join("");
      tabsEl.querySelectorAll(".case-tab").forEach((t) => {
        t.addEventListener("click", () => selectCase(t.dataset.case));
      });
    }

    function renderInfo(c) {
      const verdictColor = c.correct === false ? "var(--red)"
                          : c.correct === null  ? "var(--gold)"
                          : "var(--green)";
      const verdictLbl = c.correct === false ? "Incorrect"
                        : c.correct === null  ? "Ambiguous"
                        : "Confirmed";

      const synthChips = c.synthesized.map((m) =>
        `<span class="tag gold">${m.toUpperCase()} · synth</span>`
      ).join(" ");
      const availChips = c.available.filter((m) => m !== "seg").map((m) =>
        `<span class="tag cyan">${m.toUpperCase()}</span>`
      ).join(" ");

      infoEl.innerHTML = `
        <div class="case-narrative">
          <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom: 14px;">
            ${availChips} ${synthChips}
          </div>
          ${c.narrative}
        </div>

        <div class="confidence-card">
          <div style="display:flex; align-items:baseline; justify-content:space-between;">
            <h3 style="margin:0">Model prediction</h3>
            <span class="tag" style="color: ${verdictColor}; border-color: ${verdictColor};">
              ${verdictLbl}
            </span>
          </div>
          <p style="margin-top: 12px; font-size: 1.02rem;">
            <strong style="color: var(--text); font-size: 1.45rem;">${c.prediction}</strong>
            <span style="color: var(--text-dim); margin-left: 6px;">@ p = ${c.confidence.toFixed(2)}</span>
          </p>
          <div class="confidence-bar"><div class="confidence-fill" style="width: ${c.confidence * 100}%;"></div></div>
          <div class="confidence-meta">
            <span>Necrosis</span>
            <span>p = ${c.confidence.toFixed(2)}</span>
            <span>Recurrence</span>
          </div>
          <div class="biomarkers">
            <div class="biomarker"><div class="lbl">Euler χ</div><div class="val">${c.biomarkers.euler_chi}</div></div>
            <div class="biomarker"><div class="lbl">T1ce ratio</div><div class="val">${c.biomarkers.t1ce_ratio.toFixed(2)}</div></div>
            <div class="biomarker"><div class="lbl">Sphericity</div><div class="val">${c.biomarkers.sphericity.toFixed(2)}</div></div>
            <div class="biomarker"><div class="lbl">Volume</div><div class="val">${c.biomarkers.volume_ml.toFixed(1)} mL</div></div>
          </div>
        </div>
      `;
    }

    function updatePills(c) {
      toolbar.querySelectorAll("[data-mod]").forEach((p) => {
        const mod = p.dataset.mod;
        const present = c.available.includes(mod);
        const synth = c.synthesized.includes(mod);
        p.classList.toggle("active", mod === activeMod);
        p.classList.toggle("synth", synth);
        p.style.opacity = present || synth ? 1 : 0.35;
        p.style.pointerEvents = (present || synth) ? "auto" : "none";
      });
      toolbar.querySelector("[data-overlay='seg']").classList.toggle("active", segOn);
      toolbar.querySelector("[data-axial='multi']").classList.toggle("active", multi);
    }

    async function loadVolumes(c) {
      if (!nv) return;
      const token = ++pendingLoadToken;
      loading.classList.remove("hidden");
      const volumes = [];
      const url = c.modalities[activeMod];
      if (url) volumes.push({ url, colorMap: "gray", opacity: 1.0 });
      if (segOn && c.modalities.seg) {
        volumes.push({ url: c.modalities.seg, colorMap: "warm", opacity: 0.55 });
      }
      try {
        await nv.loadVolumes(volumes);
      } catch (err) {
        console.warn("nv.loadVolumes failed", err);
      }
      if (token !== pendingLoadToken) return;
      nv.setSliceType(multi ? nv.sliceTypeMultiplanar : nv.sliceTypeAxial);
      loading.classList.add("hidden");
    }

    function selectCase(id) {
      if (id === activeCaseId) return;
      const c = cases.find((x) => x.id === id);
      if (!c) return;
      activeCaseId = id;
      activeMod = c.available.includes("t1ce") ? "t1ce" :
                  c.available.includes("t1")   ? "t1" :
                  c.available[0];
      segOn = false;
      tabsEl.querySelectorAll(".case-tab").forEach((t) =>
        t.classList.toggle("active", t.dataset.case === id));
      renderInfo(c);
      updatePills(c);
      loadVolumes(c);
    }

    toolbar.addEventListener("click", (e) => {
      const t = e.target.closest("[data-mod], [data-overlay], [data-axial]");
      if (!t) return;
      const c = cases.find((x) => x.id === activeCaseId);
      if (!c) return;
      if (t.dataset.mod) {
        if (t.style.pointerEvents === "none") return;
        activeMod = t.dataset.mod;
      } else if (t.dataset.overlay === "seg") {
        segOn = !segOn;
      } else if (t.dataset.axial === "multi") {
        multi = !multi;
      }
      updatePills(c);
      loadVolumes(c);
    });

    // ---- Bootstrap --------------------------------------------------
    buildTabs();
    renderInfo(cases[0]);
    activeCaseId = cases[0].id;
    activeMod = cases[0].available.includes("t1ce") ? "t1ce" : cases[0].available[0];
    updatePills(cases[0]);

    const Niivue = await BrainTT.ensureNiivue();
    nv = new Niivue({
      backColor: [0.02, 0.03, 0.08, 1],
      crosshairColor: [0.13, 0.83, 0.93, 1],
      crosshairWidth: 1,
      isResizeCanvas: true,
      show3Dcrosshair: false,
      isOrientCube: false,
      textHeight: 0.04,
      meshThicknessOn2D: 0,
    });
    nv.attachToCanvas(canvas);
    await loadVolumes(cases[0]);
  });
})();
