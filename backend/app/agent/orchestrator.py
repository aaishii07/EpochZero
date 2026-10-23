"""The single agent loop. Yields events for the frontend in the order documented in README section 6."""

from __future__ import annotations

import time
from typing import Iterator

from app import config
from app.agent import fixer
from app.agent.goal_reader import read_goal, summarize
from app.agent.planner import investigate
from app.agent.verifier import Verifier
from app.analysis.facts import build_facts
from app.schemas import (
    AnalyzeRequest, AttemptEvent, Coverage, CoverageEvent, DoneEvent, ErrorEvent, Finding, FindingEvent, GoalEvent,
    NeedsInputEvent, SkippedCheck, StepEvent, Suggestion, SuggestionEvent, ToolResultEvent,
)
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import run_tool

IMPACT_ORDER = {"crash": 0, "silent": 1, "risk": 2}
TIER_ORDER = {"verified": 0, "probed": 1, "advisory": 2}
CHECK_NAMES = {
    "run_static_rules": "static rules",
    "check_goal_alignment": "goal alignment",
    "probe_shapes": "shape probe",
    "check_class_count": "class count",
}


class _Findings:
    def __init__(self):
        self.items: list[Finding] = []
        self._keys: set[tuple] = set()

    def add(self, found: list[Finding]) -> list[Finding]:
        new = []
        for f in found:
            key = (f.rule_id, f.line, f.title)
            if key in self._keys:
                continue
            self._keys.add(key)
            f.id = f"f{len(self.items) + 1}"
            self.items.append(f)
            new.append(f)
        return new

    def sorted(self) -> list[Finding]:
        return sorted(self.items, key=lambda f: (IMPACT_ORDER[f.impact], TIER_ORDER[f.tier], f.line))


def run(req: AnalyzeRequest) -> Iterator:
    started = time.monotonic()
    try:
        yield from _run(req, started)
    except Exception as e:  # never let an exception escape to the stream
        yield ErrorEvent(message=f"Internal error: {type(e).__name__}: {e}")


def _run(req: AnalyzeRequest, started: float) -> Iterator:
    code = req.code.replace("\r\n", "\n")
    try:
        facts = build_facts(code)
    except SyntaxError as e:
        yield ErrorEvent(message=f"The script does not parse: {e.msg}", line=e.lineno)
        return
    n_models, n_train = len(facts.model_classes), sum(l.kind == "train" for l in facts.loops)
    yield StepEvent(phase="parse", message=f"Parsed {len(facts.lines)} lines: {n_models} model class{'es' if n_models != 1 else ''}, "
                                           f"{n_train} training loop{'s' if n_train != 1 else ''}")

    profile, source, inferred = read_goal(req.goal, facts, code)
    summary = summarize(profile)
    offline = " (offline reader)" if source == "offline" else ""
    if inferred:
        why = f" {profile.notes.rstrip('.')}." if profile.notes else ""
        yield StepEvent(phase="context", message=f"No goal given. Inferred from the code: {summary}.{why}{offline}")
    else:
        yield StepEvent(phase="context", message=f"Goal read as: {summary}{offline}")
    yield GoalEvent(profile=profile, inferred=inferred, summary=summary)
    # Downstream prompts always get a goal sentence; an inferred one is marked as such.
    goal_text = req.goal.strip() or f"(not stated by the developer; inferred from the code: {summary})"

    ctx = ToolContext(code=code, facts=facts, profile=profile, input_shape=req.input_shape, answers=dict(req.answers),
                      goal_inferred=inferred)
    found = _Findings()
    checked: list[str] = []
    skipped: list[SkippedCheck] = []

    def record(name: str, result: ToolResult) -> list[Finding]:
        if result.ok and CHECK_NAMES.get(name) and CHECK_NAMES[name] not in checked:
            checked.append(CHECK_NAMES[name])
        skipped.extend(s for s in result.skipped if s not in skipped)
        return found.add(result.findings)

    yield StepEvent(phase="baseline", message="Running deterministic checks")
    for name in ("run_static_rules", "check_goal_alignment"):
        result = run_tool(name, ctx)
        new = record(name, result)
        yield ToolResultEvent(tool=name, ok=result.ok, summary=result.summary, finding_ids=[f.id for f in new])
        for f in new:
            yield FindingEvent(finding=f)

    yield StepEvent(phase="plan", message="Planning further checks" + (" (offline planner)" if config.LLM_MODE == "mock" else ""))
    gen = investigate(ctx, goal_text, list(found.items))
    while True:
        try:
            event = next(gen)
        except StopIteration as stop:
            plan = stop.value
            break
        if isinstance(event, tuple):  # ("result", tool name, ToolResult) from the planner
            _, name, result = event
            new = record(name, result)
            yield ToolResultEvent(tool=name, ok=result.ok, summary=result.summary, finding_ids=[f.id for f in new])
            for f in new:
                yield FindingEvent(finding=f)
        else:
            yield event
    if plan.budget_hit:
        skipped.append(SkippedCheck(check="investigation", reason="investigation budget reached"))
    if facts.model_classes and "shape probe" not in checked and not any(s.check == "shape probe" for s in skipped):
        skipped.append(SkippedCheck(check="shape probe", reason="not run for this script"))

    if plan.question:
        yield CoverageEvent(coverage=Coverage(checked=checked, skipped=skipped))
        yield NeedsInputEvent(question=plan.question)
        return

    findings = found.sorted()
    yield StepEvent(phase="repair", message=f"{len(findings)} finding{'s' if len(findings) != 1 else ''} to review" if findings
                    else "No findings to repair")
    verifier = Verifier(code, profile, req.input_shape, dict(req.answers))
    for f in findings:
        if not f.patchable:
            reason = "Advisory finding; no automatic change is proposed" if f.tier == "advisory" else \
                "Not patched automatically; the right change depends on your intent"
            yield SuggestionEvent(suggestion=Suggestion(finding=f, explanation=f.message, patch=None,
                                                        verify_status="not_attempted", verify_detail=reason, attempts=0))
            continue
        feedback, proposal, status, detail, attempt = None, None, "not_attempted", "Repair unavailable", 0
        for attempt in range(1, config.FIX_MAX_ATTEMPTS + 1):
            try:
                proposal = fixer.propose(code, goal_text, profile, f, feedback, proposal.patch if proposal else None)
            except Exception as e:
                status, detail = "not_attempted", f"Repair unavailable: {e}"
                attempt -= 1
                break
            yield StepEvent(phase="verify", message=f"Verifying patch for {f.id} ({f.rule_id}), attempt {attempt}")
            status, detail = verifier.verify(proposal.patch, f)
            yield AttemptEvent(finding_id=f.id, attempt=attempt, status=status, detail=detail)
            if status in ("resolved", "inconclusive"):
                break
            feedback = detail
        yield SuggestionEvent(suggestion=Suggestion(
            finding=f,
            explanation=proposal.explanation if proposal else f.message,
            patch=proposal.patch if proposal else None,
            verify_status=status, verify_detail=detail, attempts=max(attempt, 0)))

    yield StepEvent(phase="report", message="Analysis complete")
    yield CoverageEvent(coverage=Coverage(checked=checked, skipped=skipped))
    yield DoneEvent(elapsed_ms=int((time.monotonic() - started) * 1000))
