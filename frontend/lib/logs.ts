import type { Tier, VerifyStatus } from "./types";

// A log is one working session on one script: its code, goal, and every fix applied to it. Re-checking stays in the
// same log, so its history keeps growing; New log starts from an empty editor and an empty history.
// Logs live in this browser's localStorage only. Every access is guarded, so the app works when storage is blocked.

export interface HistoryEntry {
  id: string;
  at: number; // epoch milliseconds
  check: number; // which check of this log the fix came from
  findingId: string; // finding id within that check
  title: string;
  ruleId: string;
  tier: Tier;
  status: VerifyStatus;
  explanation: string;
  line: number; // first line of the change in the code at the time it was applied
  before: string[];
  after: string[];
}

export interface Log {
  id: string;
  name: string;
  createdAt: number;
  code: string;
  goal: string;
  inputShape: string;
  checks: number; // number of checks run in this log
  history: HistoryEntry[];
}

interface Store {
  activeId: string;
  logs: Log[];
}

const KEY = "epochzero.logs.v1";

export function newLog(existing: Log[]): Log {
  const n = existing.length + 1;
  const now = Date.now();
  return { id: `log-${now}-${n}`, name: `Log ${n}`, createdAt: now, code: "", goal: "", inputShape: "", checks: 0, history: [] };
}

export function loadStore(): Store {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Store;
      if (parsed.logs?.length && parsed.logs.some((l) => l.id === parsed.activeId)) return parsed;
    }
  } catch {
    // storage unavailable or corrupt: start fresh
  }
  const first = newLog([]);
  return { activeId: first.id, logs: [first] };
}

export function saveStore(store: Store): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(store));
  } catch {
    // storage full or blocked: the session still works, it just is not remembered
  }
}

export function formatTime(ms: number): string {
  return new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}
