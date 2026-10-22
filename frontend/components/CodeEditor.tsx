"use client";

import Editor, { type Monaco, type OnMount } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import { useEffect, useRef } from "react";

import { defineTheme, EDITOR_OPTIONS, THEME } from "@/lib/monacoTheme";
import type { Finding } from "@/lib/types";

interface Props {
  code: string;
  onChange: (code: string) => void;
  findings: Finding[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRun: () => void;
}

export default function CodeEditor({ code, onChange, findings, selectedId, onSelect, onRun }: Props) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const decorations = useRef<editor.IEditorDecorationsCollection | null>(null);
  const latest = useRef({ findings, onSelect, onRun });
  useEffect(() => {
    latest.current = { findings, onSelect, onRun };
  });

  const handleMount: OnMount = (ed, monaco) => {
    editorRef.current = ed;
    monacoRef.current = monaco;
    decorations.current = ed.createDecorationsCollection();
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => latest.current.onRun());
    ed.onMouseDown((e) => {
      const line = e.target.position?.lineNumber;
      if (!line) return;
      const hit = latest.current.findings.find((f) => line >= f.line && line <= f.end_line);
      if (hit) latest.current.onSelect(hit.id);
    });
    applyMarkers();
  };

  function applyMarkers() {
    const ed = editorRef.current;
    const monaco = monacoRef.current;
    const model = ed?.getModel();
    if (!ed || !monaco || !model) return;
    const severity = { crash: monaco.MarkerSeverity.Error, silent: monaco.MarkerSeverity.Warning, risk: monaco.MarkerSeverity.Info };
    const lineCount = model.getLineCount();
    monaco.editor.setModelMarkers(
      model,
      "epochzero",
      findings
        .filter((f) => f.line >= 1 && f.line <= lineCount)
        .map((f) => {
          const end = Math.min(f.end_line, lineCount);
          return {
            severity: severity[f.impact],
            message: `${f.title}\n${f.message}`,
            source: `EpochZero ${f.rule_id}`,
            startLineNumber: f.line,
            startColumn: model.getLineFirstNonWhitespaceColumn(f.line) || 1,
            endLineNumber: end,
            endColumn: model.getLineMaxColumn(end),
          };
        }),
    );
    const selected = findings.find((f) => f.id === selectedId);
    decorations.current?.set(
      selected && selected.line <= lineCount
        ? [
            {
              range: new monaco.Range(selected.line, 1, Math.min(selected.end_line, lineCount), 1),
              options: { isWholeLine: true, className: "ez-selected-line", glyphMarginClassName: "ez-selected-glyph" },
            },
          ]
        : [],
    );
    if (selected) ed.revealLineInCenterIfOutsideViewport(selected.line);
  }

  useEffect(applyMarkers);

  return (
    <Editor
      height="100%"
      language="python"
      theme={THEME}
      value={code}
      onChange={(value) => onChange(value ?? "")}
      beforeMount={defineTheme}
      onMount={handleMount}
      options={EDITOR_OPTIONS}
      loading={<div className="p-4 text-sm text-muted">Loading editor</div>}
    />
  );
}
