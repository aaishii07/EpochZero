import FindingCard from "@/components/FindingCard";
import type { Finding, Impact, Suggestion } from "@/lib/types";

interface Props {
  findings: Finding[];
  suggestions: Record<string, Suggestion>;
  repairing: Record<string, number>;
  selectedId: string | null;
  hasRun: boolean;
  done: boolean;
  onSelect: (id: string) => void;
  onViewPatch: (id: string) => void;
}

const GROUPS: { impact: Impact; label: string }[] = [
  { impact: "crash", label: "Crash" },
  { impact: "silent", label: "Silent" },
  { impact: "risk", label: "Risk" },
];

export default function FindingsPanel({ findings, suggestions, repairing, selectedId, hasRun, done, onSelect, onViewPatch }: Props) {
  if (!hasRun) {
    return <Empty text="Paste a training script, describe your goal, and press Check." />;
  }
  if (!findings.length) {
    return <Empty text={done ? "No issues found in the checks that ran. See Coverage for what was checked." : "Looking for issues."} />;
  }
  return (
    <div className="space-y-5 p-4">
      {GROUPS.map(({ impact, label }) => {
        const items = findings.filter((f) => f.impact === impact).sort((a, b) => a.line - b.line);
        if (!items.length) return null;
        return (
          <section key={impact}>
            <h2 className="mb-2 text-[12px] font-medium uppercase tracking-[0.04em] text-muted">
              {label} <span className="ml-1 font-mono">{items.length}</span>
            </h2>
            <div className="space-y-2">
              {items.map((f) => (
                <FindingCard
                  key={f.id}
                  finding={f}
                  suggestion={suggestions[f.id]}
                  repairing={repairing[f.id] ?? null}
                  selected={selectedId === f.id}
                  onSelect={() => onSelect(f.id)}
                  onViewPatch={() => onViewPatch(f.id)}
                />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="p-6 text-sm text-muted">{text}</p>;
}
