import StatusBadge from "@/components/StatusBadge";
import { formatTime, type HistoryEntry } from "@/lib/logs";

interface Props {
  logName: string;
  history: HistoryEntry[];
  // Fixes from the current check can still be undone or opened; older ones are part of the code now.
  isCurrent: (entry: HistoryEntry) => boolean;
  onOpen: (entry: HistoryEntry) => void;
  onUndo: (entry: HistoryEntry) => void;
}

export default function HistoryPanel({ logName, history, isCurrent, onOpen, onUndo }: Props) {
  if (!history.length) {
    return (
      <p className="p-6 text-sm text-muted">
        Fixes you apply in {logName} are listed here, newest first. The history stays when you check again; start a
        new log to begin a fresh one.
      </p>
    );
  }
  const checks = [...new Set(history.map((h) => h.check))].sort((a, b) => b - a);
  return (
    <div className="space-y-5 p-4">
      {checks.map((check) => {
        const items = history.filter((h) => h.check === check).sort((a, b) => b.at - a.at);
        return (
          <section key={check}>
            <h2 className="mb-2 text-[12px] font-medium uppercase tracking-[0.04em] text-muted">
              Check {check} <span className="ml-1 font-mono">{items.length}</span>
            </h2>
            <div className="space-y-2">
              {items.map((h) => {
                const current = isCurrent(h);
                return (
                  <article
                    key={h.id}
                    onClick={() => current && onOpen(h)}
                    className={`rounded-md border border-line border-l-[3px] border-l-resolved bg-surface p-3 ${
                      current ? "cursor-pointer hover:bg-raised" : ""
                    }`}
                  >
                    <header className="flex items-start gap-2">
                      <h3 className="min-w-0 flex-1 text-[13px] font-semibold leading-5">{h.title}</h3>
                      <span className="shrink-0 font-mono text-xs text-muted">line {h.line}</span>
                    </header>
                    <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                      <StatusBadge kind="tier" value={h.tier} />
                      <StatusBadge kind="status" value={h.tier === "advisory" && h.status === "inconclusive" ? "unverified" : h.status} />
                      <span className="font-mono text-[11px] text-muted">{h.ruleId}</span>
                      <span className="text-[11px] text-muted">{formatTime(h.at)}</span>
                    </div>
                    <p className="mt-2 text-[13px] leading-5 text-ink/90">{h.explanation}</p>
                    <pre className="mt-2 overflow-x-auto rounded border border-line bg-canvas py-1 font-mono text-[12px] leading-5">
                      {h.before.map((l, i) => (
                        <div key={`b${i}`} className="bg-crash/10 px-2 text-ink/80">
                          <span className="mr-2 select-none text-crash">-</span>
                          {l || " "}
                        </div>
                      ))}
                      {h.after.map((l, i) => (
                        <div key={`a${i}`} className="bg-resolved/10 px-2">
                          <span className="mr-2 select-none text-resolved">+</span>
                          {l || " "}
                        </div>
                      ))}
                    </pre>
                    {current && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onUndo(h);
                        }}
                        className="mt-3 h-7 rounded-md border border-line px-2.5 text-xs text-ink hover:bg-canvas"
                      >
                        Undo
                      </button>
                    )}
                  </article>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}
