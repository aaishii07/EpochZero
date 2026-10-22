"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AgentTrace, { type TimedEvent } from "@/components/AgentTrace";
import CoveragePanel from "@/components/CoveragePanel";
import FindingsPanel from "@/components/FindingsPanel";
import GoalInput, { parseShape } from "@/components/GoalInput";
import QuestionCard from "@/components/QuestionCard";
import TopBar from "@/components/TopBar";
import { analyze, fetchExamples } from "@/lib/api";
import { applyPatch } from "@/lib/patch";
import type { AgentEvent, Coverage, Example, Finding, Question, Suggestion } from "@/lib/types";

const CodeEditor = dynamic(() => import("@/components/CodeEditor"), { ssr: false });
const PatchDiff = dynamic(() => import("@/components/PatchDiff"), { ssr: false });

type Status = "idle" | "analyzing" | "needs_input" | "done" | "error";
type Tab = "findings" | "trace" | "coverage";

export default function Home() {
  const [code, setCode] = useState("");
  const [goal, setGoal] = useState("");
  const [inputShape, setInputShape] = useState("");
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [examples, setExamples] = useState<Example[]>([]);

  const [status, setStatus] = useState<Status>("idle");
  const [events, setEvents] = useState<TimedEvent[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [suggestions, setSuggestions] = useState<Record<string, Suggestion>>({});
  const [repairing, setRepairing] = useState<Record<string, number>>({});
  const [question, setQuestion] = useState<Question | null>(null);
  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const [error, setError] = useState<{ message: string; line: number | null } | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const [tab, setTab] = useState<Tab>("findings");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [diffId, setDiffId] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [hasRun, setHasRun] = useState(false);

  // Code exactly as it was when the current analysis started; patches refer to its line numbers.
  const [analyzedCode, setAnalyzedCode] = useState("");
  const abort = useRef<AbortController | null>(null);
  const switchedToFindings = useRef(false);

  useEffect(() => {
    fetchExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const handleEvent = useCallback((event: AgentEvent, at: number) => {
    setEvents((prev) => [...prev, { at, event }]);
    switch (event.type) {
      case "finding":
        setFindings((prev) => [...prev, event.finding]);
        break;
      case "attempt":
        setRepairing((prev) => ({ ...prev, [event.finding_id]: event.attempt }));
        break;
      case "suggestion": {
        const s = event.suggestion;
        setSuggestions((prev) => ({ ...prev, [s.finding.id]: s }));
        setFindings((prev) => prev.map((f) => (f.id === s.finding.id ? s.finding : f)));
        if (!switchedToFindings.current) {
          switchedToFindings.current = true;
          setTab("findings");
        }
        break;
      }
      case "needs_input":
        setQuestion(event.question);
        setStatus("needs_input");
        setTab("findings");
        break;
      case "coverage":
        setCoverage(event.coverage);
        break;
      case "done":
        setElapsed(event.elapsed_ms);
        setStatus("done");
        break;
      case "error":
        setError({ message: event.message, line: event.line });
        setStatus("error");
        break;
    }
  }, []);

  const run = useCallback(
    async (opts?: { code?: string; answers?: Record<string, string> }) => {
      const runCode = opts?.code ?? code;
      const runAnswers = opts?.answers ?? answers;
      const shape = parseShape(inputShape);
      if (!runCode.trim() || !goal.trim() || shape === "invalid") return;

      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      setAnalyzedCode(runCode);
      switchedToFindings.current = false;
      setStatus("analyzing");
      setEvents([]);
      setFindings([]);
      setSuggestions({});
      setRepairing({});
      setQuestion(null);
      setCoverage(null);
      setError(null);
      setElapsed(null);
      setSelectedId(null);
      setDiffId(null);
      setDismissed(new Set());
      setHasRun(true);
      setTab("trace");

      const started = performance.now();
      try {
        await analyze(
          { code: runCode, goal, input_shape: shape, answers: runAnswers },
          (event) => handleEvent(event, performance.now() - started),
          controller.signal,
        );
        setStatus((s) => (s === "analyzing" ? "done" : s));
      } catch (e) {
        if (controller.signal.aborted) return;
        setError({ message: e instanceof Error ? `Could not reach the analysis server. ${e.message}` : "Could not reach the analysis server.", line: null });
        setStatus("error");
      }
    },
    [code, goal, inputShape, answers, handleEvent],
  );

  const runRef = useRef(run);
  useEffect(() => {
    runRef.current = run;
  }, [run]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        runRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const visibleFindings = useMemo(() => findings.filter((f) => !dismissed.has(f.id)), [findings, dismissed]);
  const diffSuggestion = diffId ? suggestions[diffId] : undefined;
  const busy = status === "analyzing";
  const codeChanged = hasRun && code !== analyzedCode;

  const statusText = (() => {
    if (status === "analyzing") return "Analyzing";
    if (status === "needs_input") return "Waiting for your answer";
    if (status === "error") return "Analysis failed";
    if (status === "done") {
      const n = visibleFindings.length;
      const resolved = Object.values(suggestions).filter((s) => s.verify_status === "resolved" && !dismissed.has(s.finding.id)).length;
      const time = elapsed !== null ? ` in ${(elapsed / 1000).toFixed(1)}s` : "";
      return n ? `${n} finding${n === 1 ? "" : "s"}, ${resolved} verified fix${resolved === 1 ? "" : "es"}${time}` : `No issues found${time}`;
    }
    return "";
  })();

  const answer = (value: string) => {
    if (!question) return;
    const next = { ...answers, [question.id]: value };
    setAnswers(next);
    if (question.id === "input_shape") setInputShape(value);
    run({ answers: next });
  };

  const accept = () => {
    if (!diffSuggestion?.patch) return;
    try {
      const patched = applyPatch(analyzedCode, diffSuggestion.patch);
      setCode(patched);
      setDiffId(null);
      run({ code: patched });
    } catch (e) {
      setError({ message: e instanceof Error ? e.message : "Patch could not be applied", line: null });
    }
  };

  const reject = () => {
    if (!diffId) return;
    setDismissed((prev) => new Set(prev).add(diffId));
    setDiffId(null);
  };

  const loadExample = (ex: Example) => {
    abort.current?.abort();
    setCode(ex.code);
    setGoal(ex.goal);
    setInputShape("");
    setAnswers({});
    setStatus("idle");
    setEvents([]);
    setFindings([]);
    setSuggestions({});
    setQuestion(null);
    setCoverage(null);
    setError(null);
    setDiffId(null);
    setSelectedId(null);
    setHasRun(false);
  };

  return (
    <div className="flex h-screen flex-col">
      <TopBar
        statusText={statusText}
        busy={busy}
        canCheck={!!code.trim() && !!goal.trim()}
        examples={examples}
        onCheck={() => run()}
        onLoadExample={loadExample}
      />
      <div className={busy ? "ez-progress" : "h-[2px] bg-surface"} role={busy ? "progressbar" : undefined} aria-label={busy ? "Analyzing" : undefined} />

      <main className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <section className="flex min-h-[60vh] flex-col border-line lg:min-h-0 lg:w-[62%] lg:border-r">
          <GoalInput
            goal={goal}
            inputShape={inputShape}
            onGoalChange={(v) => {
              setGoal(v);
              setAnswers({});
            }}
            onInputShapeChange={setInputShape}
          />
          {error && (
            <div className="flex items-start gap-3 border-b border-l-[3px] border-line border-l-crash bg-surface px-4 py-2.5 text-[13px]">
              <span className="flex-1">
                {error.message}
                {error.line ? <span className="font-mono text-muted"> (line {error.line})</span> : null}
              </span>
              <button type="button" onClick={() => setError(null)} className="text-xs text-muted hover:text-ink">
                Dismiss
              </button>
            </div>
          )}
          {codeChanged && !diffSuggestion && status !== "analyzing" && (
            <div className="border-b border-line bg-surface px-4 py-1.5 text-xs text-muted">
              The script changed since the last check. Line markers may be out of date.
            </div>
          )}
          <div className="min-h-0 flex-1">
            {diffSuggestion ? (
              <PatchDiff
                original={analyzedCode}
                suggestion={diffSuggestion}
                onAccept={accept}
                onReject={reject}
                onClose={() => setDiffId(null)}
              />
            ) : (
              <CodeEditor
                code={code}
                onChange={setCode}
                findings={codeChanged ? [] : visibleFindings}
                selectedId={selectedId}
                onSelect={(id) => {
                  setSelectedId(id);
                  setTab("findings");
                }}
                onRun={() => runRef.current()}
              />
            )}
          </div>
        </section>

        <aside className="flex min-h-[50vh] flex-col bg-canvas lg:min-h-0 lg:w-[38%] lg:min-w-[420px]">
          <nav className="flex shrink-0 border-b border-line bg-surface" role="tablist">
            {(
              [
                ["findings", `Findings${visibleFindings.length ? ` ${visibleFindings.length}` : ""}`],
                ["trace", "Agent trace"],
                ["coverage", "Coverage"],
              ] as [Tab, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={tab === id}
                onClick={() => setTab(id)}
                className={`h-10 border-b-2 px-4 text-[13px] ${
                  tab === id ? "border-accent text-ink" : "border-transparent text-muted hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {question && status === "needs_input" && <QuestionCard key={question.id + question.prompt} question={question} onAnswer={answer} />}
            {tab === "findings" && (
              <FindingsPanel
                findings={visibleFindings}
                suggestions={suggestions}
                repairing={repairing}
                selectedId={selectedId}
                hasRun={hasRun}
                done={status === "done" || status === "needs_input"}
                onSelect={setSelectedId}
                onViewPatch={(id) => {
                  setSelectedId(id);
                  setDiffId(id);
                }}
              />
            )}
            {tab === "trace" && <AgentTrace events={events} running={busy} />}
            {tab === "coverage" && <CoveragePanel coverage={coverage} />}
          </div>
        </aside>
      </main>
    </div>
  );
}
