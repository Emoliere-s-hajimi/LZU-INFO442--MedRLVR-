// Architecture diagram + prior cards. Click a node (or a card) to lock-focus.

(async function () {
  const { metrics } = await BrainTT.dataPromise;
  const priors = metrics.priors;
  const detail = document.getElementById("archDetail");
  if (!detail) return;

  // Render prior cards from JSON
  detail.innerHTML = priors.map((p, i) => `
    <div class="prior-card${i === 0 ? " active" : ""}" data-prior-node="${p.id}">
      <h3>${p.name}</h3>
      <div class="tagline">${p.tagline}</div>
      <p>${p.summary}</p>
      <div class="formula">${p.math}</div>
      <span class="delta">Δ AUC ${p.delta_auc >= 0 ? "+" : ""}${p.delta_auc.toFixed(3)}</span>
    </div>
  `).join("");

  // Default lock state
  let locked = "modality";

  const setActive = (nodeId) => {
    locked = nodeId;
    document.querySelectorAll(".arch-node").forEach((n) => {
      n.classList.toggle("active", n.dataset.node === nodeId);
    });
    document.querySelectorAll(".prior-card").forEach((c) => {
      c.classList.toggle("active", c.dataset.priorNode === nodeId);
    });
  };

  // node → prior id mapping
  const nodeToPrior = {
    modality: "modality",
    topology: "topology",
    anatomy: "anatomy",
    bottleneck: "topology", // bottleneck shows the topology prior on click
    input: "modality",
    seg: "modality",
    cls: "topology",
  };

  document.querySelectorAll(".arch-node").forEach((n) => {
    n.addEventListener("mouseenter", () => {
      const p = nodeToPrior[n.dataset.node];
      if (!p) return;
      document.querySelectorAll(".arch-node").forEach((m) =>
        m.classList.toggle("active", m.dataset.node === n.dataset.node)
      );
      document.querySelectorAll(".prior-card").forEach((c) =>
        c.classList.toggle("active", c.dataset.priorNode === p)
      );
    });
    n.addEventListener("mouseleave", () => setActive(locked));
    n.addEventListener("click", () => {
      const p = nodeToPrior[n.dataset.node];
      if (p) setActive(p);
    });
  });

  document.querySelectorAll(".prior-card").forEach((c) => {
    c.addEventListener("click", () => setActive(c.dataset.priorNode));
  });

  setActive(locked);
})();
