import type { Log } from "@/lib/logs";
import type { Example } from "@/lib/types";

interface Props {
  statusText: string;
  busy: boolean;
  canCheck: boolean;
  examples: Example[];
  logs: Log[];
  activeLogId: string;
  onCheck: () => void;
  onLoadExample: (example: Example) => void;
  onSwitchLog: (id: string) => void;
  onNewLog: () => void;
}

export default function TopBar({
  statusText, busy, canCheck, examples, logs, activeLogId, onCheck, onLoadExample, onSwitchLog, onNewLog,
}: Props) {
  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-line bg-surface px-4">
      <span className="text-[15px] font-semibold tracking-tight">EpochZero</span>
      <span className="hidden text-xs text-muted xl:inline">Pre-training validation agent for PyTorch</span>
      <div className="flex items-center gap-2">
        <select
          aria-label="Current log"
          className="h-8 max-w-48 rounded-md border border-line bg-canvas px-2 text-xs text-ink hover:bg-raised"
          value={activeLogId}
          disabled={busy}
          onChange={(e) => onSwitchLog(e.target.value)}
        >
          {[...logs].reverse().map((log) => (
            <option key={log.id} value={log.id}>
              {log.name}
              {log.history.length ? ` (${log.history.length} fix${log.history.length === 1 ? "" : "es"})` : ""}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={onNewLog}
          disabled={busy}
          className="h-8 rounded-md border border-line px-3 text-xs text-ink hover:bg-raised disabled:opacity-50"
        >
          New log
        </button>
      </div>
      <div className="ml-auto flex items-center gap-3">
        <span className="text-xs text-muted" aria-live="polite">
          {statusText}
        </span>
        {examples.length > 0 && (
          <select
            aria-label="Load an example script"
            className="h-8 rounded-md border border-line bg-canvas px-2 text-xs text-ink hover:bg-raised"
            value=""
            onChange={(e) => {
              const ex = examples.find((x) => x.id === e.target.value);
              if (ex) onLoadExample(ex);
            }}
          >
            <option value="" disabled>
              Load example
            </option>
            {examples.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {ex.title}
              </option>
            ))}
          </select>
        )}
        <button
          type="button"
          onClick={onCheck}
          disabled={busy || !canCheck}
          className="flex h-8 items-center gap-2 rounded-md bg-accent px-3 text-sm font-medium text-on-accent hover:opacity-90 disabled:bg-raised disabled:text-muted disabled:opacity-100"
        >
          {busy ? "Checking" : "Check"}
          <span className="font-mono text-[11px] opacity-70">Ctrl+Enter</span>
        </button>
      </div>
    </header>
  );
}
