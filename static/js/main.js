// Parley project page: theme toggle, section rail, results tabs, edit-language explorer, BibTeX copy.
(function () {
  const root = document.documentElement;

  // Theme: dark by default, as on the personal site; the choice is remembered when storage is available.
  const toggle = document.getElementById("theme-toggle");
  function setTheme(theme) {
    root.dataset.theme = theme;
    toggle.innerHTML = theme === "dark" ? '<i class="bi bi-sun"></i><span>Light</span>' : '<i class="bi bi-moon"></i><span>Dark</span>';
    try { localStorage.setItem("parley-theme", theme); } catch (e) { /* storage unavailable */ }
  }
  let saved = null;
  try { saved = localStorage.getItem("parley-theme"); } catch (e) { /* storage unavailable */ }
  setTheme(saved === "light" ? "light" : "dark");
  toggle.addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));

  // Section rail: mark the section in view.
  const links = Array.from(document.querySelectorAll(".rail a[href^='#']"));
  const sections = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach((s) => observer.observe(s));
  }

  // Reach on dialogues that need correction, from Table 5 of the paper (speech and click where the editor takes clicks).
  const REACH = {
    own: { n: 356, rows: [["Reasoning segmenter", "reasoning", 8.8], ["Rule-based parser + SAM 2", "parser", 15.4], ["Vision-language cascade + SAM 2", "cascade", 18.2], ["Parley", "parley", 37.5]] },
    personas: { n: 593, rows: [["Reasoning segmenter", "reasoning", 4.2], ["Rule-based parser + SAM 2", "parser", 10.8], ["Vision-language cascade + SAM 2", "cascade", 10.3], ["Parley", "parley", 47.9]] },
    separate: { n: 376, rows: [["Reasoning segmenter", "reasoning", 7.4], ["Rule-based parser + SAM 2", "parser", 5.6], ["Vision-language cascade + SAM 2", "cascade", 9.4], ["Parley", "parley", 17.7]] },
  };
  const NOTES = {
    own: "Own simulator: validation images, 24 simulated clinicians. Parley was developed and selected on these dialogues.",
    personas: "Separately written personas: validation images, 36 simulated clinicians, new margins in the familiar phrasing.",
    separate: "Separately authored simulator: test images, 24 simulated clinicians, its own language and voices. The interval of Parley's lead over the strongest editor includes zero.",
  };
  const bars = document.getElementById("bars");
  const note = document.getElementById("bars-note");
  function drawBars(sim) {
    const max = 50;
    bars.innerHTML = REACH[sim].rows.map(([name, cls, v]) =>
      `<div class="bar-row ${cls}"><span class="bar-name">${name}</span>` +
      `<span class="bar-track"><span class="bar-fill ${cls}" style="width:0%" data-w="${(100 * v) / max}"></span></span>` +
      `<span class="bar-val">${v.toFixed(1)}%</span></div>`).join("");
    note.textContent = `${NOTES[sim]} ${REACH[sim].n} dialogues need correction. Reach within 10 turns.`;
    requestAnimationFrame(() => bars.querySelectorAll(".bar-fill").forEach((b) => { b.style.width = b.dataset.w + "%"; }));
  }
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    drawBars(tab.dataset.sim);
  }));
  drawBars("own");

  // Edit-language explorer: what each sentence is meant to produce (an illustration, not model output).
  const EDITS = {
    "shrink-top": { show: ["top"], chips: [["Shrink"], ["Top"], ["Small"]], note: "Moves the top arc inward by a small fraction of the lesion's diameter." },
    "grow-all": { show: ["all"], chips: [["Grow"], ["Whole Contour"], ["Large"]], note: "A uniform change of margin: the whole contour moves outward." },
    "shrink-left": { show: ["left"], chips: [["Shrink"], ["Left"], ["Small"]], note: "Only the arc whose outward normal points left moves, following the lesion's shape." },
    "move": { show: ["move"], chips: [["Move", "gold"], ["Down-Left", "gold"]], note: "Translates the whole mask by a predicted vector, in units of the lesion's diameter." },
    "add-bottom": { show: ["add"], chips: [["Add", "teal"], ["Bottom", "teal"]], note: "A region head predicts the missing piece on the named side and joins it to the mask." },
    "remove-click": { show: ["remove"], chips: [["Remove", "teal"], ["At Click", "teal"]], note: "One optional click locates the region; speech gives the type, the click gives the place." },
    "accept": { show: [], chips: [["Accept", "gold"]], note: "The mask stays as it is, and the margin model counts the acceptance as evidence." },
  };
  const lesion = document.querySelector(".lesion");
  const program = document.getElementById("program");
  const programNote = document.getElementById("program-note");
  function showEdit(key) {
    const e = EDITS[key];
    lesion.setAttribute("class", "lesion " + e.show.map((s) => "show-" + s).join(" "));
    program.innerHTML = e.chips.map(([label, tone]) => `<span class="chip-prog ${tone || ""}">${label}</span>`).join("");
    programNote.textContent = e.note;
  }
  document.querySelectorAll(".phrase").forEach((p) => p.addEventListener("click", () => {
    document.querySelectorAll(".phrase").forEach((q) => q.classList.toggle("active", q === p));
    showEdit(p.dataset.edit);
  }));
  showEdit("shrink-top");

  // Copy the BibTeX entry.
  const copy = document.getElementById("copy-bib");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById("bib-text").textContent);
      copy.innerHTML = '<i class="bi bi-check2"></i> Copied';
    } catch (e) {
      copy.textContent = "Select and copy";
    }
    setTimeout(() => { copy.innerHTML = '<i class="bi bi-clipboard"></i> Copy'; }, 1800);
  });
})();
