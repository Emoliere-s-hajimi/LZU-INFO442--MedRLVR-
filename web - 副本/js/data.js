// Shared data loaders. Resolves once, then memoised.
window.BrainTT = window.BrainTT || {};

BrainTT.dataPromise = (async () => {
  const [cases, metrics] = await Promise.all([
    fetch("data/cases.json").then(r => r.json()),
    fetch("data/metrics.json").then(r => r.json())
  ]);
  return { cases: cases.cases, metrics };
})();

BrainTT.fmtPct = (v) => `${(v * 100).toFixed(1)}%`;
BrainTT.fmt3   = (v) => v.toFixed(3);
BrainTT.fmt2   = (v) => v.toFixed(2);
BrainTT.signed = (v) => (v >= 0 ? "+" : "") + v.toFixed(3);
