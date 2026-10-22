"use client";

import { DiffEditor, type DiffOnMount } from "@monaco-editor/react";
import { useMemo } from "react";

import StatusBadge from "@/components/StatusBadge";
import { defineTheme, EDITOR_OPTIONS, THEME } from "@/lib/monacoTheme";
import { applyPatch } from "@/lib/patch";
import type { Suggestion } from "@/lib/types";

interface Props {
  original: string;
  suggestion: Suggestion;
  onAccept: () => void;
  onReject: () => void;
  onClose: () => void;
}

export default function PatchDiff({ original, suggestion, onAccept, onReject, onClose }: Props) {
  const { finding, patch, verify_status, verify_detail, attempts } = suggestion;
  const modified = useMemo(() => {
    if (!patch) return original;
    try {
      return applyPatch(original, patch);
    } catch {
      return original;
    }
  }, [original, patch]);

  const handleMount: DiffOnMount = (ed) => {
    if (patch) setTimeout(() => ed.getModifiedEditor().revealLineInCenter(patch.start_line), 50);
  };

  const blocked = verify_status === "unresolved" || !patch;

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 items-start gap-3 border-b border-line bg-surface px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate font-medium">{finding.title}</span>
            <span className="shrink-0 font-mono text-xs text-muted">line {finding.line}</span>
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-muted">
            <StatusBadge kind="status" value={verify_status} />
            <span className="truncate">{verify_detail}</span>
            {attempts > 1 && <span className="shrink-0">({attempts} attempts)</span>}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={onClose}
            className="h-8 rounded-md border border-line px-3 text-sm text-ink hover:bg-raised"
          >
            Back to editor
          </button>
          <button
            type="button"
            onClick={onReject}
            className="h-8 rounded-md border border-crash px-3 text-sm text-crash hover:bg-raised"
          >
            Reject
          </button>
          <button
            type="button"
            onClick={onAccept}
            disabled={blocked}
            title={blocked ? "This patch did not pass verification" : undefined}
            className="h-8 rounded-md bg-accent px-3 text-sm font-medium text-on-accent hover:opacity-90 disabled:bg-raised disabled:text-muted disabled:opacity-100"
          >
            Accept
          </button>
        </div>
      </div>
      {verify_status === "unresolved" && (
        <p className="border-b border-line bg-surface px-4 py-2 text-xs text-unresolved">
          This patch did not pass verification and cannot be applied.
        </p>
      )}
      {verify_status === "inconclusive" && (
        <p className="border-b border-line bg-surface px-4 py-2 text-xs text-inconclusive">
          Unverified: the check could not run on the patched code. Review the change before accepting.
        </p>
      )}
      <div className="min-h-0 flex-1">
        <DiffEditor
          height="100%"
          language="python"
          theme={THEME}
          original={original}
          modified={modified}
          beforeMount={defineTheme}
          onMount={handleMount}
          options={{ ...EDITOR_OPTIONS, readOnly: true, renderSideBySide: true, originalEditable: false }}
          loading={<div className="p-4 text-sm text-muted">Loading diff</div>}
        />
      </div>
    </div>
  );
}
