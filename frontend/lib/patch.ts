import type { Patch } from "./types";

// Same logic as backend/app/agent/patching.py.
export function applyPatch(code: string, patch: Patch): string {
  const newline = code.includes("\r\n") ? "\r\n" : "\n";
  const lines = code.split(newline);
  if (patch.start_line < 1 || patch.end_line < patch.start_line || patch.end_line > lines.length) {
    throw new Error(`Patch range ${patch.start_line}-${patch.end_line} is outside the file`);
  }
  let replacement = patch.replacement.replace(/\r\n/g, "\n");
  if (replacement.endsWith("\n")) replacement = replacement.slice(0, -1);
  const inserted = patch.replacement === "" ? [] : replacement.split("\n");
  return [...lines.slice(0, patch.start_line - 1), ...inserted, ...lines.slice(patch.end_line)].join(newline);
}
