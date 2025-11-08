const primaryColorScheme = ""; // "light" | "dark"
const themeStorageKey = "theme";

const legacyThemeMap = {
  catppuccin: "catppuccin-dark",
  gruvbox: "gruvbox-dark",
};

const supportedThemes = [
  "light",
  "dark",
  "catppuccin-light",
  "catppuccin-dark",
  "gruvbox-light",
  "gruvbox-dark",
  "tokyo-storm",
];

const systemThemes = new Set(["light", "dark"]);

function normalizeTheme(theme) {
  if (!theme) return undefined;
  const mapped = legacyThemeMap[theme] ?? theme;
  return supportedThemes.includes(mapped) ? mapped : undefined;
}

function resolvePreferredTheme() {
  const stored = normalizeTheme(localStorage.getItem(themeStorageKey));
  if (stored) return stored;

  const primary = normalizeTheme(primaryColorScheme);
  if (primary) return primary;

  const system = window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
  return normalizeTheme(system) ?? "light";
}

let themeValue = resolvePreferredTheme();

if (normalizeTheme(localStorage.getItem(themeStorageKey)) !== themeValue) {
  localStorage.setItem(themeStorageKey, themeValue);
}

function datasetTheme(value) {
  return normalizeTheme(value ?? "");
}

function themeButtons() {
  return document.querySelectorAll("[data-theme-group]");
}

function setPreference(nextTheme) {
  const normalized = normalizeTheme(nextTheme) ?? "light";
  themeValue = normalized;
  localStorage.setItem(themeStorageKey, themeValue);
  reflectPreference();
}

function reflectPreference() {
  document.firstElementChild?.setAttribute("data-theme", themeValue);

  themeButtons().forEach(button => {
    const light = datasetTheme(button.getAttribute("data-theme-light"));
    const dark = datasetTheme(button.getAttribute("data-theme-dark"));
    const label = button.getAttribute("data-theme-label");

    let state = "inactive";
    if (light && themeValue === light) state = "light";
    else if (dark && themeValue === dark) state = "dark";

    const pressed = state !== "inactive";
    button.setAttribute("aria-pressed", String(pressed));

    if (pressed) {
      button.setAttribute("data-state", state);
    } else {
      button.removeAttribute("data-state");
    }

    if (label) {
      const descriptor =
        state === "inactive"
          ? `Activate ${label} theme`
          : `Switch ${label} theme (current ${state})`;
      button.setAttribute("aria-label", descriptor);
    }
  });

  const body = document.body;
  if (body) {
    const computedStyles = window.getComputedStyle(body);
    const bgColor = computedStyles.backgroundColor;
    document
      .querySelector("meta[name='theme-color']")
      ?.setAttribute("content", bgColor);
  }
}

reflectPreference();

window.onload = () => {
  function handleThemeButtonClick(event) {
    const button = event.currentTarget;
    if (!(button instanceof HTMLElement)) return;

    const light = datasetTheme(button.getAttribute("data-theme-light"));
    const dark = datasetTheme(button.getAttribute("data-theme-dark"));

    let nextTheme = light ?? dark ?? themeValue;

    if (light && themeValue === light && dark) {
      nextTheme = dark;
    } else if (dark && themeValue === dark && light) {
      nextTheme = light;
    }

    setPreference(nextTheme);
  }

  function setThemeFeature() {
    reflectPreference();

    themeButtons().forEach(button => {
      button.removeEventListener("click", handleThemeButtonClick);
      button.addEventListener("click", handleThemeButtonClick);
    });
  }

  setThemeFeature();
  document.addEventListener("astro:after-swap", setThemeFeature);
};

document.addEventListener("astro:before-swap", event => {
  const bgColor = document
    .querySelector("meta[name='theme-color']")
    ?.getAttribute("content");

  event.newDocument
    .querySelector("meta[name='theme-color']")
    ?.setAttribute("content", bgColor);
});

window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", ({ matches: isDark }) => {
    const storedTheme = normalizeTheme(localStorage.getItem(themeStorageKey));

    if (storedTheme && !systemThemes.has(storedTheme)) {
      return;
    }

    setPreference(isDark ? "dark" : "light");
  });
