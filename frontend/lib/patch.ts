import type { Patch } from "./types";

// Same logic as backend/app/agent/patching.py.
export function applyPatch(code: string, patch: Patch): string {
  const newline = code.includes("\r\n") ? "\r\n" : "\n";
  const lines = code.split(newline);
  if (patch.start_line < 1 || patch.end_line < patch.start_line || patch.end_line > lines.length) {
    throw new Error(`Patch range ${patch.start_line}-${patch.end_line} is outside the file`);
  }
  const inserted = replacementLines(patch);
  return [...lines.slice(0, patch.start_line - 1), ...inserted, ...lines.slice(patch.end_line)].join(newline);
}

export function replacementLines(patch: Patch): string[] {
  if (patch.replacement === "") return [];
  let text = patch.replacement.replace(/\r\n/g, "\n");
  if (text.endsWith("\n")) text = text.slice(0, -1);
  return text.split("\n");
}

// How many lines a patch adds (positive) or removes (negative).
export function patchDelta(patch: Patch): number {
  return replacementLines(patch).length - (patch.end_line - patch.start_line + 1);
}

export function overlaps(a: Patch, b: Patch): boolean {
  return a.start_line <= b.end_line && b.start_line <= a.end_line;
}

// All patches refer to line numbers in the analyzed code. After some of them are applied, a line of the analyzed
// code moves by the size change of every applied patch that ends above it.
export function shiftLine(line: number, applied: Patch[]): number {
  return applied.reduce((n, p) => (p.end_line < line ? n + patchDelta(p) : n), line);
}

export function rebase(patch: Patch, applied: Patch[]): Patch {
  return { ...patch, start_line: shiftLine(patch.start_line, applied), end_line: shiftLine(patch.end_line, applied) };
}

// Where an applied patch's new lines sit in the current code (1-indexed, inclusive). A pure deletion marks the line
// where the removed text used to start.
export function appliedRange(patch: Patch, applied: Patch[]): { start: number; end: number } {
  const start = shiftLine(patch.start_line, applied.filter((a) => a !== patch));
  return { start, end: start + Math.max(replacementLines(patch).length, 1) - 1 };
}

// Apply non-overlapping patches (all relative to `base`) bottom-up, so earlier line numbers stay valid.
export function composePatches(base: string, patches: Patch[]): string {
  return [...patches].sort((a, b) => b.start_line - a.start_line).reduce((code, p) => applyPatch(code, p), base);
}
