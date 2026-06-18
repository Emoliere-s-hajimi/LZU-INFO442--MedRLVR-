// Keyboard navigation + scroll-progress ring + help overlay.
//
// Shortcuts:
//   j / k          next / previous section
//   g g            top
//   G              bottom
//   1 .. 9         jump to nth section
//   /              jump to Cohort Explorer
//   ?              toggle help overlay
//   Esc            close help overlay
//
// The progress ring on the right shows scroll progress + current section
// number; updated cheaply via requestAnimationFrame.

(function () {
  const sections = Array.from(document.querySelectorAll("section[id]"));
  if (!sections.length) return;

  // ---- Section navigation ------------------------------------------------
  function currentSectionIdx() {
    const y = window.scrollY + window.innerHeight * 0.3;
    for (let i = sections.length - 1; i >= 0; i--) {
      if (sections[i].offsetTop <= y) return i;
    }
    return 0;
  }

  function jumpTo(idx) {
    idx = Math.max(0, Math.min(sections.length - 1, idx));
    sections[idx].scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---- Scroll-progress ring ---------------------------------------------
  const ring = document.getElementById("scrollProgress");
  const bar = document.getElementById("progressBar");
  const numLabel = document.getElementById("sectionNum");
  const CIRCUMF = 91; // 2π · 14.5 ≈ 91, matches the stroke-dasharray in CSS

  let pendingFrame = false;
  function updateProgress() {
    if (pendingFrame) return;
    pendingFrame = true;
    requestAnimationFrame(() => {
      pendingFrame = false;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const t = Math.max(0, Math.min(1, window.scrollY / Math.max(1, max)));
      if (bar) bar.style.strokeDashoffset = (CIRCUMF * (1 - t)).toFixed(1);
      const idx = currentSectionIdx();
      if (numLabel) numLabel.textContent = String(idx + 1).padStart(2, "0");
      if (ring) ring.style.opacity = window.scrollY > 200 ? 1 : 0;
    });
  }
  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);
  updateProgress();

  // ---- Help overlay -----------------------------------------------------
  const overlay = document.getElementById("helpOverlay");
  const fab = document.getElementById("helpFab");
  const close = document.getElementById("helpClose");
  function setHelp(open) {
    overlay.classList.toggle("open", open);
    overlay.setAttribute("aria-hidden", open ? "false" : "true");
    if (open && close) close.focus();
  }
  fab.addEventListener("click", () => setHelp(true));
  close.addEventListener("click", () => setHelp(false));
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) setHelp(false);
  });

  // ---- Keyboard handler --------------------------------------------------
  let gPending = false;
  let gTimer = null;
  function isTyping(e) {
    const t = e.target;
    if (!t) return false;
    const tn = (t.tagName || "").toLowerCase();
    if (tn === "input" || tn === "textarea" || tn === "select") return true;
    if (t.isContentEditable) return true;
    return false;
  }

  window.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isTyping(e)) return;

    // Help overlay always reachable
    if (e.key === "?" || (e.shiftKey && e.key === "/")) {
      e.preventDefault();
      setHelp(!overlay.classList.contains("open"));
      return;
    }
    if (e.key === "Escape" && overlay.classList.contains("open")) {
      e.preventDefault();
      setHelp(false);
      return;
    }
    if (overlay.classList.contains("open")) return;

    if (e.key === "j") {
      e.preventDefault();
      jumpTo(currentSectionIdx() + 1);
      return;
    }
    if (e.key === "k") {
      e.preventDefault();
      jumpTo(currentSectionIdx() - 1);
      return;
    }
    if (e.key === "G") {
      e.preventDefault();
      jumpTo(sections.length - 1);
      return;
    }
    if (e.key === "g") {
      if (gPending) {
        jumpTo(0);
        gPending = false;
        clearTimeout(gTimer);
        return;
      }
      gPending = true;
      gTimer = setTimeout(() => { gPending = false; }, 600);
      return;
    }
    if (/^[1-9]$/.test(e.key)) {
      e.preventDefault();
      jumpTo(parseInt(e.key, 10) - 1);
      return;
    }
    if (e.key === "/") {
      e.preventDefault();
      const ex = document.getElementById("explorer");
      if (ex) ex.scrollIntoView({ behavior: "smooth" });
      return;
    }
  });
})();
