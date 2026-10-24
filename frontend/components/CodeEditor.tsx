"use client";

import Editor, { type Monaco, type OnMount } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import { useEffect, useRef } from "react";

import { defineTheme, EDITOR_OPTIONS, THEME } from "@/lib/monacoTheme";
import type { Finding } from "@/lib/types";

export interface AppliedRange {
  start: number;
  end: number;
  title: string;
}

export interface Focus {
  line: number;
  end: number;
  nonce: number; // changes on every request, so the same line can be focused twice
}

interface Props {
  code: string;
  onChange: (code: string) => void;
  findings: Finding[];
  applied: AppliedRange[];
  focus: Focus | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRun: () => void;
}

export default function CodeEditor({ code, onChange, findings, applied, focus, selectedId, onSelect, onRun }: Props) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const decorations = useRef<editor.IEditorDecorationsCollection | null>(null);
  const latest = useRef({ findings, onSelect, onRun, focus });
  useEffect(() => {
    latest.current = { findings, onSelect, onRun, focus };
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
    // The editor remounts when the diff view closes; return to the line that was just fixed, not the top.
    if (latest.current.focus) reveal(latest.current.focus);
  };

  function reveal(f: Focus) {
    const ed = editorRef.current;
    const model = ed?.getModel();
    if (!ed || !model) return;
    const line = Math.min(Math.max(1, f.line), model.getLineCount());
    ed.revealLineInCenter(line);
    ed.setPosition({ lineNumber: line, column: model.getLineFirstNonWhitespaceColumn(line) || 1 });
  }

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
    const decos: editor.IModelDeltaDecoration[] = applied
      .filter((r) => r.start <= lineCount)
      .map((r) => ({
        range: new monaco.Range(r.start, 1, Math.min(r.end, lineCount), 1),
        options: {
          isWholeLine: true,
          className: "ez-applied-line",
          glyphMarginClassName: "ez-applied-glyph",
          hoverMessage: { value: `Fixed by EpochZero: ${r.title}` },
        },
      }));
    if (selected && selected.line <= lineCount) {
      decos.push({
        range: new monaco.Range(selected.line, 1, Math.min(selected.end_line, lineCount), 1),
        options: { isWholeLine: true, className: "ez-selected-line", glyphMarginClassName: "ez-selected-glyph" },
      });
    }
    decorations.current?.set(decos);
  }

  useEffect(applyMarkers);

  useEffect(() => {
    if (focus) reveal(focus);
  }, [focus]);

  useEffect(() => {
    const selected = findings.find((f) => f.id === selectedId);
    if (selected) editorRef.current?.revealLineInCenterIfOutsideViewport(selected.line);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

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
