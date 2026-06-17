// Scroll reveal, nav highlighting, modality-bar fill-in, donut chart.

(function () {
  // --- Scroll reveal -------------------------------------------------
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal, .reveal-stagger").forEach((el) => io.observe(el));

  // --- Modality bars fill -------------------------------------------
  const fillIo = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        const pct = Number(e.target.dataset.pct) / 100;
        e.target.style.transform = `scaleX(${pct})`;
        fillIo.unobserve(e.target);
      }
    });
  }, { threshold: 0.6 });
  document.querySelectorAll(".modality-bar .fill").forEach((f) => fillIo.observe(f));

  // --- Donut chart for cohort ---------------------------------------
  const donut = document.getElementById("donutSvg");
  if (donut) {
    const segments = [
      { value: 199, color: "#f472b6" }, // R
      { value: 71,  color: "#a78bfa" }, // RN
      { value: 52,  color: "#22d3ee" }, // N
    ];
    const total = segments.reduce((a, b) => a + b.value, 0);
    const cx = 50, cy = 50, r = 38, sw = 14;
    const circumf = 2 * Math.PI * r;
    let offset = 0;
    // Background track
    const ns = "http://www.w3.org/2000/svg";
    const bg = document.createElementNS(ns, "circle");
    bg.setAttribute("cx", cx); bg.setAttribute("cy", cy); bg.setAttribute("r", r);
    bg.setAttribute("fill", "none"); bg.setAttribute("stroke", "rgba(255,255,255,0.05)");
    bg.setAttribute("stroke-width", sw);
    donut.appendChild(bg);
    segments.forEach((seg) => {
      const arcLen = (seg.value / total) * circumf;
      const path = document.createElementNS(ns, "circle");
      path.setAttribute("cx", cx); path.setAttribute("cy", cy); path.setAttribute("r", r);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", seg.color);
      path.setAttribute("stroke-width", sw);
      path.setAttribute("stroke-dasharray", `${arcLen} ${circumf}`);
      path.setAttribute("stroke-dashoffset", -offset);
      path.style.transition = "stroke-dashoffset 0.8s ease, stroke-dasharray 1.2s ease";
      path.setAttribute("stroke-linecap", "butt");
      path.style.filter = `drop-shadow(0 0 4px ${seg.color}88)`;
      donut.appendChild(path);
      offset += arcLen;
    });
  }

  // --- Nav active link as you scroll --------------------------------
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll("nav.topbar .links a");
  const navIo = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          const id = e.target.id;
          navLinks.forEach((a) => {
            a.classList.toggle("active", a.getAttribute("href") === `#${id}`);
          });
        }
      });
    },
    { rootMargin: "-30% 0px -55% 0px", threshold: 0 }
  );
  sections.forEach((s) => navIo.observe(s));
})();
