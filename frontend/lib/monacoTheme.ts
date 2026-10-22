import type { Monaco } from "@monaco-editor/react";

export const THEME = "epochzero";

export function defineTheme(monaco: Monaco) {
  monaco.editor.defineTheme(THEME, {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": "#0E1116",
      "editor.lineHighlightBackground": "#161B22",
      "editor.lineHighlightBorder": "#161B22",
      "editorLineNumber.foreground": "#4B5563",
      "editorLineNumber.activeForeground": "#8B949E",
      "editorGutter.background": "#0E1116",
      "editorWidget.background": "#161B22",
      "editorWidget.border": "#2A313C",
      "editorHoverWidget.background": "#161B22",
      "editorHoverWidget.border": "#2A313C",
      "editorError.foreground": "#F85149",
      "editorWarning.foreground": "#D29922",
      "editorInfo.foreground": "#8B949E",
      "diffEditor.insertedTextBackground": "#3FB95026",
      "diffEditor.removedTextBackground": "#F8514926",
      "diffEditor.insertedLineBackground": "#3FB95014",
      "diffEditor.removedLineBackground": "#F8514914",
      "scrollbarSlider.background": "#2A313C99",
      "scrollbarSlider.hoverBackground": "#2A313C",
      "editor.selectionBackground": "#4C8DFF40",
    },
  });
}

export const EDITOR_OPTIONS = {
  fontFamily: "var(--font-jetbrains), Consolas, monospace",
  fontSize: 13,
  lineHeight: 20,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  renderLineHighlight: "line" as const,
  glyphMargin: true,
  padding: { top: 12, bottom: 12 },
  tabSize: 4,
  automaticLayout: true,
  scrollbar: { useShadows: false },
  overviewRulerBorder: false,
  hideCursorInOverviewRuler: true,
};
