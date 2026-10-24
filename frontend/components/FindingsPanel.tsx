"use client";

import { useState } from "react";

import FindingCard, { type FixState } from "@/components/FindingCard";
import { CATEGORY_LABEL, CATEGORY_ORDER } from "@/lib/categories";
import type { Origin } from "@/lib/origin";
import type { Finding, Impact, Suggestion } from "@/lib/types";

interface Props {
  findings: Finding[];
  suggestions: Record<string, Suggestion>;
  repairing: Record<string, number>;
  selectedId: string | null;
  hasRun: boolean;
  done: boolean;
  reviewSkipped: string | null; // why the code review did not run, if it did not
  onOpen: (id: string) => void;
  fixState: (id: string) => FixState;
  canApply: (id: string) => boolean;
  onApply: (id: string) => void;
  onUndo: (id: string) => void;
  applyAllCount: number;
  unverifiedCount: number;
  onApplyAll: () => void;
  checking: Set<string>;
  clashes: Record<string, string>;
  origins: Record<string, Origin>;
  notice: string | null;
  onDismissNotice: () => void;
}

type GroupBy = "severity" | "area";

const SEVERITY: { impact: Impact; label: string }[] = [
  { impact: "crash", label: "Crash" },
  { impact: "silent", label: "Silent" },
  { impact: "risk", label: "Risk" },
];
const IMPACT_RANK: Record<Impact, number> = { crash: 0, silent: 1, risk: 2 };
const GROUP_KEY = "epochzero.groupBy";

function groups(findings: Finding[], by: GroupBy) {
  if (by === "severity") {
    return SEVERITY.map(({ impact, label }) => ({
      key: impact, label, items: findings.filter((f) => f.impact === impact).sort((a, b) => a.line - b.line),
    })).filter((g) => g.items.length);
  }
  // By area: pipeline order, and within an area the most severe first.
  return CATEGORY_ORDER.map((c) => ({
    key: c, label: CATEGORY_LABEL[c],
    items: findings.filter((f) => (f.category ?? "other") === c)
      .sort((a, b) => IMPACT_RANK[a.impact] - IMPACT_RANK[b.impact] || a.line - b.line),
  })).filter((g) => g.items.length);
}

function savedGroup(): GroupBy {
  try {
    return window.localStorage.getItem(GROUP_KEY) === "area" ? "area" : "severity";
  } catch {
    return "severity";
  }
}

export default function FindingsPanel({
  findings, suggestions, repairing, selectedId, hasRun, done, reviewSkipped, onOpen, fixState, canApply, onApply, onUndo, applyAllCount,
  unverifiedCount, onApplyAll, checking, clashes, origins, notice, onDismissNotice,
}: Props) {
  const [groupBy, setGroupBy] = useState<GroupBy>(savedGroup);
  const chooseGroup = (g: GroupBy) => {
    setGroupBy(g);
    try {
      window.localStorage.setItem(GROUP_KEY, g);
    } catch {
      // remembering the choice is a convenience only
    }
  };
  if (!hasRun) {
    return <Empty text="Paste a training script and press Check. The goal is optional." />;
  }
  if (!findings.length) {
    if (done && reviewSkipped) {
      return (
        <div className="m-4 rounded-md border border-line border-l-[3px] border-l-inconclusive bg-surface p-4 text-[13px] leading-5">
          <p className="font-medium text-inconclusive">The rule checks found nothing, but the code review did not run.</p>
          <p className="mt-1 text-muted">
            Reason: {reviewSkipped}. Most problems in real scripts are found by the code review, so this is not a
            clean result. Press Check again in a minute.
          </p>
        </div>
      );
    }
    return <Empty text={done ? "No issues found in the checks that ran. See Coverage for what was checked." : "Looking for issues."} />;
  }
  const fromFix = findings.filter((f) => origins[f.id]?.kind === "from_fix").length;
  const still = findings.filter((f) => origins[f.id]?.kind === "still").length;
  const fresh = findings.filter((f) => origins[f.id]?.kind === "new").length;
  const compared = fromFix + still + fresh > 0;
  return (
    <div className="space-y-5 p-4">
      {compared && (
        <p className="rounded-md border border-line bg-surface px-3 py-2 text-[13px] text-muted">
          Compared with the last check: <span className="text-inconclusive">{fromFix} on lines your fixes changed</span>,{" "}
          {still} still present, <span className="text-accent">{fresh} new</span>. New ones were either noticed this
          time or uncovered once an earlier problem was fixed.
        </p>
      )}
      {notice && (
        <div className="flex items-start gap-3 rounded-md border border-line border-l-[3px] border-l-accent bg-surface px-3 py-2">
          <p className="flex-1 text-[13px] text-muted">{notice}</p>
          <button type="button" onClick={onDismissNotice} className="text-xs text-muted hover:text-ink">
            Dismiss
          </button>
        </div>
      )}
      {checking.size > 1 && (
        <p className="text-[13px] text-accent">Checking {checking.size} fixes together with the ones already applied</p>
      )}
      {applyAllCount > 1 && (
        <div className="flex items-center gap-3 rounded-md border border-line bg-surface p-3">
          <p className="flex-1 text-[13px] text-muted">
            {applyAllCount} fixes ready
            {unverifiedCount > 0 && `, ${unverifiedCount} of them unverified (from the code review)`}. EpochZero first
            checks them together and holds back any that break the script in combination, then re-checks. Fixes that
            failed verification are not included.
          </p>
          <button
            type="button"
            onClick={onApplyAll}
            className="h-8 shrink-0 rounded-md bg-accent px-3 text-sm font-medium text-on-accent hover:opacity-90"
          >
            Apply all {applyAllCount}
          </button>
        </div>
      )}
      <div className="flex items-center gap-2 text-xs text-muted">
        <span>Group by</span>
        {(["severity", "area"] as GroupBy[]).map((g) => (
          <button
            key={g}
            type="button"
            onClick={() => chooseGroup(g)}
            aria-pressed={groupBy === g}
            className={`h-7 rounded-md border px-2.5 ${groupBy === g ? "border-accent text-ink" : "border-line hover:bg-raised"}`}
          >
            {g === "severity" ? "Severity" : "Area"}
          </button>
        ))}
      </div>
      {groups(findings, groupBy).map(({ key, label, items }) => {
        return (
          <section key={key}>
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
                  state={fixState(f.id)}
                  canApply={canApply(f.id)}
                  onOpen={() => onOpen(f.id)}
                  onApply={() => onApply(f.id)}
                  onUndo={() => onUndo(f.id)}
                  showCategory={groupBy === "severity"}
                  checking={checking.has(f.id)}
                  clash={clashes[f.id] ?? null}
                  origin={origins[f.id] ?? null}
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
