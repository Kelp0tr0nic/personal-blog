import { useEffect, useState } from "react";
import Giscus, { type Theme } from "@giscus/react";
import { GISCUS } from "@/constants";

type ThemeToken =
  | "light"
  | "dark"
  | "catppuccin-light"
  | "catppuccin-dark"
  | "gruvbox-light"
  | "gruvbox-dark";

interface CommentsProps {
  lightTheme?: Theme;
  darkTheme?: Theme;
}

function resolveTheme(): ThemeToken {
  const stored = localStorage.getItem("theme");
  if (stored) {
    return stored as ThemeToken;
  }

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

function toUiTheme(theme: ThemeToken, lightTheme: Theme, darkTheme: Theme): Theme {
  if (theme.endsWith("dark")) {
    return darkTheme;
  }
  return lightTheme;
}

export default function Comments({
  lightTheme = "light",
  darkTheme = "dark",
}: CommentsProps) {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof window !== "undefined"
      ? toUiTheme(resolveTheme(), lightTheme, darkTheme)
      : lightTheme
  );

  useEffect(() => {
    const applyThemeFromDom = () => {
      const current = (document.documentElement.getAttribute("data-theme") ??
        resolveTheme()) as ThemeToken;
      setTheme(toUiTheme(current, lightTheme, darkTheme));
    };

    applyThemeFromDom();

    const observer = new MutationObserver(applyThemeFromDom);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => observer.disconnect();
  }, [lightTheme, darkTheme]);

  return (
    <section id="comments" className="mt-8">
      <Giscus theme={theme} {...GISCUS} />
    </section>
  );
}
