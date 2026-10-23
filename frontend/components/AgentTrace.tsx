"use client";

import { useEffect, useRef } from "react";

import StatusBadge from "@/components/StatusBadge";
import type { AgentEvent } from "@/lib/types";

export interface TimedEvent {
  at: number;
  event: AgentEvent;
}

const TOOL_LABEL: Record<string, string> = {
  run_static_rules: "static rules",
  check_goal_alignment: "goal alignment",
  probe_shapes: "shape probe",
  check_class_count: "class count",
  inspect_lines: "inspect lines",
};

export default function AgentTrace({ events, running }: { events: TimedEvent[]; running: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [events.length]);

  if (!events.length) {
    return <p className="p-6 text-sm text-muted">Each step the agent takes appears here while it works.</p>;
  }

  return (
    <div
      ref={box}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      }}
      className="h-full overflow-y-auto p-4"
    >
      <ol className="space-y-1.5">
        {events.map(({ at, event }, i) => {
          const row = describe(event);
          if (!row) return null;
          return (
            <li key={i} className="grid grid-cols-[52px_92px_1fr] gap-2 text-[13px] leading-5">
              <span className="font-mono text-[11px] leading-5 text-muted">{(at / 1000).toFixed(1)}s</span>
              <span className={`font-mono text-[11px] leading-5 ${row.tone}`}>{row.label}</span>
              <div className="min-w-0">
                {row.text}
                {event.type === "tool_call" && Object.keys(event.args).length > 0 && (
                  <details className="mt-0.5">
                    <summary className="cursor-pointer text-xs text-muted">arguments</summary>
                    <pre className="mt-1 overflow-x-auto rounded border border-line bg-canvas p-2 font-mono text-[11px]">
                      {JSON.stringify(event.args, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      {running && <p className="mt-3 font-mono text-[11px] text-muted">working</p>}
    </div>
  );
}

function describe(e: AgentEvent): { label: string; tone: string; text: React.ReactNode } | null {
  switch (e.type) {
    case "step":
      return { label: e.phase, tone: "text-muted", text: <span className="text-muted">{e.message}</span> };
    case "tool_call":
      return {
        label: "call",
        tone: "text-accent",
        text: (
          <span>
            <span className="font-mono">{TOOL_LABEL[e.tool] ?? e.tool}</span>
            {e.reason && <span className="text-muted">: {e.reason}</span>}
          </span>
        ),
      };
    case "tool_result":
      return {
        label: "result",
        tone: e.ok ? "text-ink" : "text-inconclusive",
        text: (
          <span>
            <span className="font-mono">{TOOL_LABEL[e.tool] ?? e.tool}</span> <span className="text-muted">{e.summary}</span>
          </span>
        ),
      };
    case "finding":
      return {
        label: "finding",
        tone: e.finding.impact === "crash" ? "text-crash" : e.finding.impact === "silent" ? "text-silent" : "text-risk",
        text: (
          <span>
            <span className="font-mono text-muted">{e.finding.rule_id}</span> {e.finding.title}
          </span>
        ),
      };
    case "attempt":
      return {
        label: "verify",
        tone: "text-muted",
        text: (
          <span className="inline-flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-muted">
              {e.finding_id} attempt {e.attempt}
            </span>
            <StatusBadge kind="status" value={e.status} />
            <span className="text-muted">{e.detail}</span>
          </span>
        ),
      };
    case "goal":
      return null; // the context step already describes the goal
    case "suggestion":
      return null;
    case "needs_input":
      return { label: "question", tone: "text-accent", text: e.question.prompt };
    case "coverage":
      return null;
    case "done":
      return { label: "done", tone: "text-resolved", text: <span className="text-muted">Finished in {(e.elapsed_ms / 1000).toFixed(1)}s</span> };
    case "error":
      return { label: "error", tone: "text-crash", text: e.message };
  }
}
