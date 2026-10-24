"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import AgentTrace, { type TimedEvent } from "@/components/AgentTrace";
import type { AppliedRange, Focus } from "@/components/CodeEditor";
import CoveragePanel from "@/components/CoveragePanel";
import FindingsPanel from "@/components/FindingsPanel";
import GoalInput, { parseShape, shapeIsUnknownText } from "@/components/GoalInput";
import HistoryPanel from "@/components/HistoryPanel";
import QuestionCard from "@/components/QuestionCard";
import TopBar from "@/components/TopBar";
import { analyze, fetchExamples, verifyFixes } from "@/lib/api";
import { classify, type Origin, type Snapshot } from "@/lib/origin";
import { loadStore, newLog, saveStore, type HistoryEntry, type Log } from "@/lib/logs";
import { appliedRange, composePatches, overlaps, rebase, replacementLines, shiftLine } from "@/lib/patch";
import type { AgentEvent, Coverage, Example, Finding, Patch, Question, Suggestion } from "@/lib/types";

const CodeEditor = dynamic(() => import("@/components/CodeEditor"), { ssr: false });
const PatchDiff = dynamic(() => import("@/components/PatchDiff"), { ssr: false });

type Status = "idle" | "analyzing" | "needs_input" | "done" | "error";
type Tab = "findings" | "history" | "trace" | "coverage";

// The first check of a new log starts on its own once a script has been pasted; later checks are manual.
const AUTO_CHECK_MIN_LINES = 5;
const AUTO_CHECK_DELAY_MS = 1200;

export default function Workspace() {
  // Log: the working session. Its code, goal, shape, check count, and history are saved in the browser.
  // `savedLogs` holds every log as last saved; the active one's live fields are the states below.
  const [initial] = useState(() => {
    const store = loadStore();
    return { store, active: store.logs.find((l) => l.id === store.activeId)! };
  });
  const [savedLogs, setSavedLogs] = useState<Log[]>(initial.store.logs);
  const [activeLogId, setActiveLogId] = useState(initial.active.id);
  const [code, setCode] = useState(initial.active.code);
  const [goal, setGoal] = useState(initial.active.goal);
  const [inputShape, setInputShape] = useState(initial.active.inputShape);
  const [checks, setChecks] = useState(initial.active.checks);
  const [history, setHistory] = useState<HistoryEntry[]>(initial.active.history);

  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [examples, setExamples] = useState<Example[]>([]);

  // Current check.
  const [status, setStatus] = useState<Status>("idle");
  const [events, setEvents] = useState<TimedEvent[]>([]);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [suggestions, setSuggestions] = useState<Record<string, Suggestion>>({});
  const [repairing, setRepairing] = useState<Record<string, number>>({});
  const [question, setQuestion] = useState<Question | null>(null);
  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const [error, setError] = useState<{ message: string; line: number | null } | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [inferredGoal, setInferredGoal] = useState<{ summary: string; notes: string } | null>(null);

  const [tab, setTab] = useState<Tab>("findings");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [diffId, setDiffId] = useState<string | null>(null);
  const [focus, setFocus] = useState<Focus | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  // Fixes applied since the last check (finding ids). Their patches are composed onto analyzedCode.
  const [applied, setAppliedState] = useState<Set<string>>(new Set());
  const [hasRun, setHasRun] = useState(false);
  // Combined check: fixes being checked right now, and fixes held back with the reason.
  const [checking, setChecking] = useState<Set<string>>(new Set());
  const [clashes, setClashes] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<string | null>(null);
  // The agent's closing report for the current check.
  const [report, setReport] = useState<{ text: string; mode: "llm" | "fallback" | "offline" } | null>(null);
  // Re-check labels: where each finding comes from compared with the previous check in this log.
  const [origins, setOrigins] = useState<Record<string, Origin>>({});
  const compareRef = useRef<Snapshot | null>(null); // the previous check, frozen when a re-check starts
  const snapshotRef = useRef<Snapshot | null>(null); // the current check, kept up to date every render

  // Code exactly as it was when the current check started; patches refer to its line numbers.
  const [analyzedCode, setAnalyzedCode] = useState("");
  const abort = useRef<AbortController | null>(null);
  const switchedToFindings = useRef(false);

  // Every log with the active one's live contents merged in, and saved to the browser whenever it changes.
  const logs = useMemo(
    () => savedLogs.map((l) => (l.id === activeLogId ? { ...l, code, goal, inputShape, checks, history } : l)),
    [savedLogs, activeLogId, code, goal, inputShape, checks, history],
  );
  useEffect(() => {
    saveStore({ activeId: activeLogId, logs });
  }, [activeLogId, logs]);

  useEffect(() => {
    fetchExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const handleEvent = useCallback((event: AgentEvent, at: number) => {
    setEvents((prev) => [...prev, { at, event }]);
    switch (event.type) {
      case "finding": {
        const origin = classify(event.finding, compareRef.current);
        if (origin) setOrigins((prev) => ({ ...prev, [event.finding.id]: origin }));
        setFindings((prev) => [...prev, event.finding]);
        break;
      }
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
      case "report":
        setReport({ text: event.text, mode: event.mode });
        break;
      case "goal":
        setInferredGoal(event.inferred ? { summary: event.summary, notes: event.profile.notes } : null);
        break;
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

  const resetCheck = () => {
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
    setFocus(null);
    setDismissed(new Set());
    setAppliedState(new Set());
    setInferredGoal(null);
    setClashes({});
    setNotice(null);
    setReport(null);
    setOrigins({});
  };

  const run = useCallback(
    async (opts?: { code?: string; answers?: Record<string, string> }) => {
      const runCode = opts?.code ?? code;
      let runAnswers = opts?.answers ?? answers;
      const shape = parseShape(inputShape);
      if (!runCode.trim()) return;
      if (shape === "invalid") {
        setError({ message: "The input shape is not valid. Use positive integers separated by commas, or leave it empty.", line: null });
        return;
      }
      // "Don't know" typed in the shape box answers the shape question up front, so it is never asked.
      if (shapeIsUnknownText(inputShape) && !runAnswers.input_shape) runAnswers = { ...runAnswers, input_shape: "dont know" };

      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;
      // Freeze the previous check of this log so the new findings can be labelled against it.
      compareRef.current = snapshotRef.current;
      setAnalyzedCode(runCode);
      switchedToFindings.current = false;
      resetCheck();
      setStatus("analyzing");
      setHasRun(true);
      setChecks((n) => n + 1);
      setTab("trace");

      const started = performance.now();
      try {
        await analyze(
          {
            code: runCode,
            goal,
            input_shape: shape,
            answers: runAnswers,
            // The backend keeps no state: send what this log already did so the agent does not undo or repeat it.
            history: history.map((h) => ({
              title: h.title, rule_id: h.ruleId, check: h.check, line: h.line, before: h.before, after: h.after,
            })),
            previous_open: compareRef.current?.previous ?? [],
          },
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
    [code, goal, inputShape, answers, history, handleEvent],
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

  // First check of a log: start automatically once a script is in the editor and typing has paused.
  useEffect(() => {
    if (checks > 0 || hasRun || status !== "idle") return;
    if (code.split("\n").filter((l) => l.trim()).length < AUTO_CHECK_MIN_LINES) return;
    const timer = setTimeout(() => runRef.current(), AUTO_CHECK_DELAY_MS);
    return () => clearTimeout(timer);
  }, [checks, hasRun, status, code]);

  // ----- Fixes -----
  // Every patch refers to line numbers in analyzedCode. Applied fixes are tracked as a set, the editor shows the
  // analyzed code with those patches composed, and the remaining patches, markers, and line numbers are shifted to
  // match. So fixes can be applied one at a time without a re-check in between.
  const visibleFindings = useMemo(() => findings.filter((f) => !dismissed.has(f.id)), [findings, dismissed]);
  const busy = status === "analyzing";
  const patchOf = useCallback((id: string) => suggestions[id]?.patch ?? null, [suggestions]);
  const appliedList = useMemo(() => [...applied].map(patchOf).filter((p): p is Patch => !!p), [applied, patchOf]);
  // The code the editor should hold given the applied fixes; anything else means a hand edit.
  const expectedCode = useMemo(() => {
    try {
      return composePatches(analyzedCode, appliedList);
    } catch {
      return analyzedCode;
    }
  }, [analyzedCode, appliedList]);
  const codeChanged = hasRun && code !== expectedCode;
  const liveFindings = useMemo(
    () => visibleFindings.map((f) => (applied.has(f.id) ? f : { ...f, line: shiftLine(f.line, appliedList), end_line: shiftLine(f.end_line, appliedList) })),
    [visibleFindings, applied, appliedList],
  );
  const appliedRanges: AppliedRange[] = useMemo(
    () => [...applied].flatMap((id) => {
      const p = patchOf(id);
      return p ? [{ ...appliedRange(p, appliedList), title: suggestions[id].finding.title }] : [];
    }),
    [applied, appliedList, patchOf, suggestions],
  );

  // What a re-check started now would be compared with: the lines the applied fixes changed, and the findings still
  // open. Only meaningful while the editor holds exactly the analyzed code plus the applied fixes.
  useEffect(() => {
    if (!hasRun) {
      snapshotRef.current = null; // a new or switched log: its first check has nothing to compare with
      return;
    }
    if (status === "analyzing") return;
    snapshotRef.current = codeChanged
      ? { changed: [], previous: visibleFindings.filter((f) => !applied.has(f.id)).map((f) => ({ rule_id: f.rule_id, title: f.title, line: f.line })) }
      : {
          changed: appliedRanges,
          previous: liveFindings.filter((f) => !applied.has(f.id)).map((f) => ({ rule_id: f.rule_id, title: f.title, line: f.line })),
        };
  });

  const conflicts = (id: string) => {
    const p = patchOf(id);
    return !!p && !applied.has(id) && appliedList.some((a) => overlaps(a, p));
  };
  const pending = (s: Suggestion | undefined) =>
    !!s?.patch && !applied.has(s.finding.id) && !dismissed.has(s.finding.id) && !conflicts(s.finding.id) &&
    ["resolved", "inconclusive"].includes(s.verify_status);
  const applyAllIds = visibleFindings.map((f) => f.id).filter((id) => pending(suggestions[id]));
  const unverifiedCount = applyAllIds.filter((id) => suggestions[id].verify_status === "inconclusive").length;

  const commitApplied = (next: Set<string>, recheck: boolean, focusId?: string) => {
    const patches = [...next].map(patchOf).filter((p): p is Patch => !!p);
    try {
      const composed = composePatches(analyzedCode, patches);
      setAppliedState(next);
      setCode(composed);
      setDiffId(null);
      const target = focusId ? patchOf(focusId) : null;
      if (target) {
        const r = appliedRange(target, patches);
        setFocus({ line: r.start, end: r.end, nonce: Date.now() });
      }
      if (recheck) run({ code: composed });
      return patches;
    } catch (e) {
      setError({ message: e instanceof Error ? e.message : "Patch could not be applied", line: null });
      return null;
    }
  };

  // Applying runs the combined check first: the selected fixes are added, one at a time, on top of the fixes already
  // applied, and any that makes the whole script worse is held back with the reason. `force` skips the check
  // (Apply anyway on a held-back fix).
  const applyFixes = async (ids: string[], opts: { forceRecheck?: boolean; force?: boolean } = {}) => {
    const candidates: string[] = [];
    const taken = new Set(applied);
    for (const id of ids) {
      const p = patchOf(id);
      if (!p || taken.has(id)) continue;
      if ([...taken].some((other) => overlaps(patchOf(other)!, p))) continue; // overlapping fixes are skipped
      taken.add(id);
      candidates.push(id);
    }
    if (!candidates.length) return;

    let added = candidates;
    let heldBack = 0;
    if (!opts.force) {
      setChecking(new Set(candidates));
      try {
        const shape = parseShape(inputShape);
        const result = await verifyFixes({
          code: analyzedCode,
          goal,
          input_shape: shape === "invalid" ? null : shape,
          answers,
          patches: [...applied, ...candidates].map((id) => ({ id, patch: patchOf(id)! })),
          applied_ids: [...applied],
        });
        added = candidates.filter((id) => result.accepted.includes(id));
        heldBack = result.rejected.length;
        if (heldBack) setClashes((prev) => ({ ...prev, ...Object.fromEntries(result.rejected.map((r) => [r.id, r.reason])) }));
        if (candidates.length > 1) {
          setNotice(heldBack
            ? `${result.summary}. The held-back fixes say why on their cards; you can still apply them one by one.`
            : result.summary);
        }
      } catch {
        setNotice("Could not check the fixes together (backend unreachable), so they were applied without the combined check.");
      } finally {
        setChecking(new Set());
      }
    } else {
      setClashes((prev) => {
        const next = { ...prev };
        candidates.forEach((id) => delete next[id]);
        return next;
      });
    }
    if (!added.length) return;

    const next = new Set([...applied, ...added]);
    // Re-check automatically once every suggested fix is in (or after Apply all); otherwise wait for Check.
    // If some fixes were held back, stay so their reasons can be read.
    const remaining = visibleFindings.some((f) => {
      const p = patchOf(f.id);
      return !!p && !next.has(f.id) && !dismissed.has(f.id) &&
        ["resolved", "inconclusive"].includes(suggestions[f.id].verify_status) &&
        ![...next].some((other) => overlaps(patchOf(other)!, p));
    });
    const patches = commitApplied(next, !heldBack && (!!opts.forceRecheck || !remaining), added[0]);
    if (!patches) return;
    const lines = analyzedCode.split("\n");
    const now = Date.now();
    const entries: HistoryEntry[] = added.map((id, i) => {
      const s = suggestions[id];
      const p = s.patch!;
      return {
        id: `${now}-${i}-${id}`,
        at: now,
        check: checks,
        findingId: id,
        title: s.finding.title,
        ruleId: s.finding.rule_id,
        tier: s.finding.tier,
        status: s.verify_status,
        explanation: s.explanation,
        line: appliedRange(p, patches).start,
        before: lines.slice(p.start_line - 1, p.end_line),
        after: replacementLines(p),
      };
    });
    setHistory((prev) => [...prev, ...entries]);
  };

  const undoFix = (id: string) => {
    const next = new Set(applied);
    next.delete(id);
    commitApplied(next, false);
    setHistory((prev) => prev.filter((h) => !(h.check === checks && h.findingId === id)));
    const f = liveFindings.find((x) => x.id === id);
    if (f) setFocus({ line: shiftLine(f.line, [...next].map(patchOf).filter((p): p is Patch => !!p)), end: f.end_line, nonce: Date.now() });
  };

  const accept = () => {
    if (diffId) applyFixes([diffId], { force: !!clashes[diffId] });
  };

  const reject = () => {
    if (!diffId) return;
    setDismissed((prev) => new Set(prev).add(diffId));
    setDiffId(null);
  };

  const closeDiff = () => {
    const f = liveFindings.find((x) => x.id === diffId);
    if (f) setFocus({ line: f.line, end: f.end_line, nonce: Date.now() });
    setDiffId(null);
  };

  // The diff for the open card: a proposed fix is shown against the current code; an applied fix is shown against
  // the code without it, so it is clear what it changed.
  const diff = (() => {
    const s = diffId ? suggestions[diffId] : undefined;
    if (!s?.patch) return null;
    if (applied.has(s.finding.id)) {
      const others = appliedList.filter((p) => p !== s.patch);
      return { mode: "applied" as const, original: composePatches(analyzedCode, others), suggestion: { ...s, patch: rebase(s.patch, others) } };
    }
    return { mode: "proposed" as const, original: expectedCode, suggestion: { ...s, patch: rebase(s.patch, appliedList) } };
  })();

  // ----- Logs -----
  const switchToLog = (log: Log) => {
    setSavedLogs(logs); // keep the outgoing log's latest contents
    abort.current?.abort();
    resetCheck();
    setStatus("idle");
    setHasRun(false);
    setAnswers({});
    setAnalyzedCode("");
    setActiveLogId(log.id);
    setCode(log.code);
    setGoal(log.goal);
    setInputShape(log.inputShape);
    setChecks(log.checks);
    setHistory(log.history);
    setTab("findings");
  };

  const onSwitchLog = (id: string) => {
    const log = logs.find((l) => l.id === id);
    if (log && id !== activeLogId) switchToLog(log);
  };

  const startLog = (log: Log) => {
    switchToLog(log);
    setSavedLogs([...logs, log]);
  };

  const onNewLog = () => startLog(newLog(logs));

  const loadExample = (ex: Example) => {
    // An example is a different script, so it gets its own log; its first check starts automatically.
    startLog({ ...newLog(logs), name: ex.title, code: ex.code, goal: ex.goal });
  };

  const activeLog = logs.find((l) => l.id === activeLogId);
  const isCurrentEntry = (h: HistoryEntry) => h.check === checks && applied.has(h.findingId) && !!suggestions[h.findingId];

  // The code review finds most semantic problems; when it could not run, an empty list is not a clean result.
  const reviewSkipped = coverage?.skipped.find((s) => s.check === "code review")?.reason ?? null;

  const statusText = (() => {
    if (status === "analyzing") return "Analyzing";
    if (status === "needs_input") return "Waiting for your answer";
    if (status === "error") return "Analysis failed";
    if (status === "done") {
      const n = visibleFindings.length;
      if (applied.size > 0) {
        return `${applied.size} of ${n} applied. Press Check when you are ready to re-check.`;
      }
      const live = Object.values(suggestions).filter((s) => !dismissed.has(s.finding.id) && s.patch);
      const verified = live.filter((s) => s.verify_status === "resolved").length;
      const unverified = live.filter((s) => s.verify_status === "inconclusive").length;
      const time = elapsed !== null ? ` in ${(elapsed / 1000).toFixed(1)}s` : "";
      if (!n) return reviewSkipped ? "Code review did not run, so this is not a full result. Check again in a minute." : `No issues found${time}`;
      return `${n} finding${n === 1 ? "" : "s"}, ${verified} verified fix${verified === 1 ? "" : "es"}` +
        (unverified ? `, ${unverified} unverified` : "") + time;
    }
    return "";
  })();

  const answer = (value: string) => {
    if (!question) return;
    const next = { ...answers, [question.id]: value };
    setAnswers(next);
    if (question.id === "input_shape" && !/dont know/.test(value)) setInputShape(value);
    run({ answers: next });
  };

  return (
    <div className="flex h-screen flex-col">
      <TopBar
        statusText={statusText}
        busy={busy}
        canCheck={!!code.trim()}
        examples={examples}
        logs={logs}
        activeLogId={activeLogId}
        onCheck={() => run()}
        onLoadExample={loadExample}
        onSwitchLog={onSwitchLog}
        onNewLog={onNewLog}
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
            inferred={!goal.trim() ? inferredGoal : null}
            onUseInferred={() => {
              if (inferredGoal) setGoal(inferredGoal.summary);
            }}
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
          {codeChanged && !diff && status !== "analyzing" && (
            <div className="border-b border-line bg-surface px-4 py-1.5 text-xs text-muted">
              The script was edited by hand since the last check, so the suggested fixes are paused. Press Check to refresh them.
            </div>
          )}
          <div className="min-h-0 flex-1">
            {diff ? (
              <PatchDiff
                original={diff.original}
                suggestion={diff.suggestion}
                mode={diff.mode}
                onAccept={accept}
                onReject={reject}
                onUndo={() => diffId && undoFix(diffId)}
                onClose={closeDiff}
              />
            ) : (
              <CodeEditor
                code={code}
                onChange={setCode}
                findings={codeChanged ? [] : liveFindings.filter((f) => !applied.has(f.id) && !conflicts(f.id))}
                applied={codeChanged ? [] : appliedRanges}
                focus={focus}
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
                ["history", `History${history.length ? ` ${history.length}` : ""}`],
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
                findings={liveFindings}
                suggestions={suggestions}
                repairing={repairing}
                selectedId={selectedId}
                hasRun={hasRun}
                done={status === "done" || status === "needs_input"}
                reviewSkipped={reviewSkipped}
                report={report}
                onOpen={(id) => {
                  setSelectedId(id);
                  // A card with a fix opens its patch (proposed or applied); one without just highlights its lines.
                  if (suggestions[id]?.patch && !conflicts(id) && !codeChanged) setDiffId(id);
                  else {
                    const f = liveFindings.find((x) => x.id === id);
                    if (f) setFocus({ line: f.line, end: f.end_line, nonce: Date.now() });
                  }
                }}
                fixState={(id) =>
                  applied.has(id) ? "applied" : conflicts(id) ? "conflict" : codeChanged ? "paused" : "ready"
                }
                canApply={(id) => !busy && !checking.size && !codeChanged && !!patchOf(id) && !applied.has(id) && !conflicts(id)}
                onApply={(id) => applyFixes([id], { force: !!clashes[id] })}
                onUndo={undoFix}
                applyAllCount={busy || codeChanged || checking.size ? 0 : applyAllIds.filter((id) => !clashes[id]).length}
                unverifiedCount={unverifiedCount}
                onApplyAll={() => applyFixes(applyAllIds.filter((id) => !clashes[id]), { forceRecheck: true })}
                checking={checking}
                clashes={clashes}
                origins={origins}
                notice={notice}
                onDismissNotice={() => setNotice(null)}
              />
            )}
            {tab === "history" && (
              <HistoryPanel
                logName={activeLog?.name ?? "this log"}
                history={history}
                isCurrent={isCurrentEntry}
                onOpen={(h) => setDiffId(h.findingId)}
                onUndo={(h) => undoFix(h.findingId)}
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
