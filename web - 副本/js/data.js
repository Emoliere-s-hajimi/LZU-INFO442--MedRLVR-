// Shared loaders + perf helpers.
window.BrainTT = window.BrainTT || {};

BrainTT.dataPromise = (async () => {
  const [cases, metrics] = await Promise.all([
    fetch("data/cases.json").then((r) => r.json()),
    fetch("data/metrics.json").then((r) => r.json()),
  ]);
  return { cases: cases.cases, metrics };
})();

BrainTT.fmtPct = (v) => `${(v * 100).toFixed(1)}%`;
BrainTT.fmt3   = (v) => v.toFixed(3);
BrainTT.fmt2   = (v) => v.toFixed(2);
BrainTT.signed = (v) => (v >= 0 ? "+" : "") + v.toFixed(3);

/* ---------- lazy-init helper ---------------------------------------
   Run `fn` once when `el` first scrolls within rootMargin. This is
   how every heavy chart bootstraps — Plotly never gets a chance to
   parse data the user hasn't scrolled to. */
BrainTT.lazy = function (el, fn, rootMargin = "200px 0px") {
  if (!el) return;
  if (!("IntersectionObserver" in window)) {
    fn();
    return;
  }
  let done = false;
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting && !done) {
          done = true;
          io.disconnect();
          fn();
        }
      });
    },
    { rootMargin }
  );
  io.observe(el);
};

/* ---------- shared Plotly theme ------------------------------------ */
BrainTT.plotly = (function () {
  const baseFont = { family: "JetBrains Mono, ui-monospace, monospace", color: "#95a8c8", size: 11 };
  const axis = {
    color: "#5a6c8b",
    gridcolor: "rgba(120, 180, 255, 0.07)",
    zerolinecolor: "rgba(120, 180, 255, 0.15)",
    linecolor: "rgba(120, 180, 255, 0.25)",
    tickfont: baseFont,
    titlefont: { ...baseFont, color: "#95a8c8" },
  };
  return {
    baseFont,
    axis,
    layout(extra = {}) {
      return {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: baseFont,
        margin: { l: 60, r: 20, t: 24, b: 50 },
        showlegend: true,
        legend: { font: baseFont, bgcolor: "rgba(0,0,0,0)", bordercolor: "rgba(120,180,255,0.15)", borderwidth: 1 },
        ...extra,
      };
    },
    config: { responsive: true, displayModeBar: false, doubleClick: "reset" },
    colors: {
      cyan: "#22d3ee", magenta: "#f472b6", violet: "#a78bfa", gold: "#facc15",
      green: "#34d399", red: "#f87171", text: "#e6f0ff", soft: "#95a8c8",
    },
  };
})();

/* ---------- Plotly CDN loader, on demand --------------------------- */
let _plotlyPromise = null;
BrainTT.ensurePlotly = function () {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (_plotlyPromise) return _plotlyPromise;
  _plotlyPromise = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.30.0.min.js";
    s.async = true;
    s.onload = () => res(window.Plotly);
    s.onerror = rej;
    document.head.appendChild(s);
  });
  return _plotlyPromise;
};

/* ---------- Three.js CDN loader, on demand ------------------------- */
let _threePromise = null;
BrainTT.ensureThree = function () {
  if (window.THREE) return Promise.resolve(window.THREE);
  if (_threePromise) return _threePromise;
  _threePromise = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.min.js";
    s.async = true;
    s.onload = () => res(window.THREE);
    s.onerror = rej;
    document.head.appendChild(s);
  });
  return _threePromise;
};

/* ---------- NiiVue CDN loader, on demand --------------------------- */
// We use esm.sh (not jsdelivr) because @niivue/niivue ships an ESM with
// bare specifiers like `import { mat4 } from "gl-matrix"` — browsers can't
// resolve those without a bundler or import-map. esm.sh re-writes every
// bare specifier into a same-origin URL so the browser can follow them
// natively. Pinned version for reproducibility.
const NIIVUE_CDN = "https://esm.sh/@niivue/niivue@0.69.0";
let _niivuePromise = null;
BrainTT.ensureNiivue = function () {
  if (window.Niivue) return Promise.resolve(window.Niivue);
  if (_niivuePromise) return _niivuePromise;
  _niivuePromise = (async () => {
    // 20s timeout — fail fast and surface the error if the CDN is unreachable.
    const timeout = new Promise((_, rej) =>
      setTimeout(() => rej(new Error("NiiVue load timed out (20s)")), 20000)
    );
    try {
      const mod = await Promise.race([import(NIIVUE_CDN), timeout]);
      if (!mod || !mod.Niivue) {
        throw new Error("NiiVue module loaded but does not export `Niivue`");
      }
      window.Niivue = mod.Niivue;
      return mod.Niivue;
    } catch (err) {
      console.error("[BrainTT] NiiVue failed to load:", err);
      throw err;
    }
  })();
  return _niivuePromise;
};

/* ---------- rAF-throttled debouncer for sliders -------------------- */
BrainTT.rafDebounce = function (fn) {
  let id = 0;
  return function () {
    if (id) return;
    const ctx = this, args = arguments;
    id = requestAnimationFrame(() => {
      id = 0;
      fn.apply(ctx, args);
    });
  };
};
