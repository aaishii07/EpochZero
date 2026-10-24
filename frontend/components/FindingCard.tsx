import StatusBadge from "@/components/StatusBadge";
import { CATEGORY_LABEL, displayStatus } from "@/lib/categories";
import type { Origin } from "@/lib/origin";
import type { Finding, Suggestion } from "@/lib/types";

// ready: the fix can be applied; applied: already in the editor; conflict: overlaps an applied fix;
// paused: the script was edited by hand, so line numbers can no longer be trusted until the next check.
export type FixState = "ready" | "applied" | "conflict" | "paused";

interface Props {
  finding: Finding;
  suggestion: Suggestion | undefined;
  repairing: number | null;
  selected: boolean;
  state: FixState;
  canApply: boolean;
  onOpen: () => void;
  onApply: () => void;
  onUndo: () => void;
  showCategory: boolean; // off when the list is already grouped by category
  checking: boolean; // being checked together with the other fixes
  clash: string | null; // held back by the combined check, with the reason
  origin: Origin | null; // compared with the previous check of this log
}

const BORDER = { crash: "border-l-crash", silent: "border-l-silent", risk: "border-l-risk" };

export default function FindingCard({
  finding, suggestion, repairing, selected, state, canApply, onOpen, onApply, onUndo, showCategory, checking, clash, origin,
}: Props) {
  const body = suggestion?.explanation || finding.message;
  const hasPatch = !!suggestion?.patch;
  const applied = state === "applied";
  return (
    <article
      onClick={onOpen}
      title={hasPatch && state === "ready" ? "Open the proposed change" : undefined}
      className={`cursor-pointer rounded-md border border-line border-l-[3px] ${applied ? "border-l-resolved" : BORDER[finding.impact]} p-3 ${
        selected ? "bg-raised" : "bg-surface hover:bg-raised"
      } ${applied ? "opacity-70" : ""}`}
    >
      <header className="flex items-start gap-2">
        <h3 className="min-w-0 flex-1 text-[13px] font-semibold leading-5">{finding.title}</h3>
        {!applied && (
          <span className="shrink-0 font-mono text-xs text-muted">
            {finding.end_line > finding.line ? `lines ${finding.line}-${finding.end_line}` : `line ${finding.line}`}
          </span>
        )}
      </header>
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <StatusBadge kind="tier" value={finding.tier} />
        {suggestion ? (
          <StatusBadge kind="status" value={displayStatus(finding, suggestion)} />
        ) : repairing ? (
          <StatusBadge kind="status" value="repairing" />
        ) : null}
        {showCategory && <span className="text-[11px] text-muted">{CATEGORY_LABEL[finding.category ?? "other"]}</span>}
        <span className="font-mono text-[11px] text-muted">{finding.rule_id}</span>
        {applied && <span className="text-[11px] font-medium text-resolved">Applied</span>}
        {!applied && origin?.kind === "from_fix" && (
          <span className="text-[11px] font-medium text-inconclusive" title="This problem is on lines a fix you applied changed">
            From your fix: {origin.fix}
          </span>
        )}
        {!applied && origin?.kind === "still" && <span className="text-[11px] text-muted">Still present</span>}
        {!applied && origin?.kind === "new" && <span className="text-[11px] font-medium text-accent">New in this check</span>}
      </div>
      {!applied && (
        <>
          <p className="mt-2 text-[13px] leading-5 text-ink/90">{body}</p>
          {repairing && !suggestion && <p className="mt-1 text-xs text-accent">Repairing, attempt {repairing} of 3</p>}
          {suggestion && suggestion.verify_status !== "not_attempted" && (
            <p className="mt-1 text-xs text-muted">{suggestion.verify_detail}</p>
          )}
          {state === "conflict" && (
            <p className="mt-1 text-xs text-inconclusive">This fix overlaps one you applied. Press Check to get an updated fix.</p>
          )}
          {clash && state === "ready" && (
            <p className="mt-1 text-xs text-crash">Held back: {clash}</p>
          )}
        </>
      )}
      {selected && !applied && (
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
      {hasPatch && (
        <div className="mt-3 flex items-center gap-2">
          {applied ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onUndo();
              }}
              className="h-7 rounded-md border border-line px-2.5 text-xs text-ink hover:bg-canvas"
            >
              Undo
            </button>
          ) : (
            <button
              type="button"
              disabled={!canApply}
              onClick={(e) => {
                e.stopPropagation();
                onApply();
              }}
              className={
                suggestion!.verify_status === "unresolved" || clash
                  ? "h-7 rounded-md border border-crash px-2.5 text-xs text-crash hover:bg-canvas disabled:opacity-50"
                  : "h-7 rounded-md bg-accent px-2.5 text-xs font-medium text-on-accent hover:opacity-90 disabled:bg-raised disabled:text-muted"
              }
            >
              {checking
                ? "Checking with your other fixes"
                : suggestion!.verify_status === "unresolved" || clash
                  ? "Apply anyway"
                  : suggestion!.verify_status === "inconclusive"
                    ? "Apply unverified fix"
                    : "Apply fix"}
            </button>
          )}
        </div>
      )}
    </article>
  );
}
