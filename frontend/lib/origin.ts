import type { Finding } from "./types";

// Where a finding in a re-check comes from, compared with the check before it in the same log.
export type Origin =
  | { kind: "from_fix"; fix: string } // on lines a fix you applied changed
  | { kind: "still" } // the previous check reported the same problem and it was not fixed
  | { kind: "new" }; // not reported before: newly noticed, or uncovered once an earlier problem was fixed

export interface Snapshot {
  // Lines changed by applied fixes, in the coordinates of the code being re-checked.
  changed: { start: number; end: number; title: string }[];
  // Unfixed findings of the previous check, with line numbers moved to the code being re-checked.
  previous: { rule_id: string; title: string; line: number }[];
}

function words(text: string): Set<string> {
  return new Set(text.toLowerCase().split(/[^a-z0-9_]+/).filter((w) => w.length > 3));
}

function similar(a: string, b: string): boolean {
  const wa = words(a);
  const wb = words(b);
  if (!wa.size || !wb.size) return false;
  let common = 0;
  wa.forEach((w) => wb.has(w) && common++);
  return common / Math.min(wa.size, wb.size) >= 0.5;
}

export function classify(finding: Finding, snap: Snapshot | null): Origin | null {
  if (!snap) return null; // first check of a log: nothing to compare with
  const touched = snap.changed.find((r) => finding.line <= r.end + 1 && finding.end_line >= r.start - 1);
  if (touched) return { kind: "from_fix", fix: touched.title };
  const same = snap.previous.find(
    (p) => p.rule_id === finding.rule_id && (Math.abs(p.line - finding.line) <= 3 || similar(p.title, finding.title)),
  );
  return same ? { kind: "still" } : { kind: "new" };
}
