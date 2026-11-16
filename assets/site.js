(() => {
  const root = document.documentElement;
  const toggle = document.querySelector("[data-theme-toggle]");
  if (!toggle) return;

  const label = () => {
    const text = root.dataset.theme === "dark" ? "切换到浅色模式" : "切换到深色模式";
    toggle.setAttribute("aria-label", text);
    toggle.setAttribute("title", text);
  };

  label();
  toggle.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("diveli-theme", root.dataset.theme);
    } catch {}
    label();
  });
})();
