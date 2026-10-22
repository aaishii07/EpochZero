import StatusBadge from "@/components/StatusBadge";
import type { Finding, Suggestion } from "@/lib/types";

interface Props {
  finding: Finding;
  suggestion: Suggestion | undefined;
  repairing: number | null;
  selected: boolean;
  onSelect: () => void;
  onViewPatch: () => void;
}

const BORDER = { crash: "border-l-crash", silent: "border-l-silent", risk: "border-l-risk" };

export default function FindingCard({ finding, suggestion, repairing, selected, onSelect, onViewPatch }: Props) {
  const body = suggestion?.explanation || finding.message;
  return (
    <article
      onClick={onSelect}
      className={`cursor-pointer rounded-md border border-line border-l-[3px] ${BORDER[finding.impact]} p-3 ${
        selected ? "bg-raised" : "bg-surface hover:bg-raised"
      }`}
    >
      <header className="flex items-start gap-2">
        <h3 className="min-w-0 flex-1 text-[13px] font-semibold leading-5">{finding.title}</h3>
        <span className="shrink-0 font-mono text-xs text-muted">
          {finding.end_line > finding.line ? `lines ${finding.line}-${finding.end_line}` : `line ${finding.line}`}
        </span>
      </header>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <StatusBadge kind="tier" value={finding.tier} />
        {suggestion ? (
          <StatusBadge kind="status" value={suggestion.verify_status} />
        ) : repairing ? (
          <StatusBadge kind="status" value="repairing" />
        ) : null}
        <span className="font-mono text-[11px] text-muted">{finding.rule_id}</span>
      </div>
      <p className="mt-2 text-[13px] leading-5 text-ink/90">{body}</p>
      {repairing && !suggestion && <p className="mt-1 text-xs text-accent">Repairing, attempt {repairing} of 3</p>}
      {suggestion && suggestion.verify_status !== "not_attempted" && (
        <p className="mt-1 text-xs text-muted">{suggestion.verify_detail}</p>
      )}
      {selected && (
        <div className="mt-3 space-y-2 border-t border-line pt-2">
          <ul className="space-y-1">
            {finding.evidence.map((ev, i) => (
              <li key={i} className="text-xs leading-4 text-muted">
                <span className="font-mono text-ink">{ev.tool}</span> {ev.summary}
              </li>
            ))}
          </ul>
          {finding.docs && (
            <p className="text-xs leading-4 text-muted">
              {finding.docs.snippet}{" "}
              <a
                href={finding.docs.url}
                target="_blank"
                rel="noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-accent hover:underline"
              >
                {finding.docs.title}
              </a>
            </p>
          )}
        </div>
      )}
      {suggestion?.patch && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onViewPatch();
          }}
          className="mt-3 h-7 rounded-md border border-line px-2.5 text-xs text-ink hover:bg-canvas"
        >
          View patch
        </button>
      )}
    </article>
  );
}
