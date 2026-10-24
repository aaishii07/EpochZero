"""Runs one check: parse, read the goal, baseline rules, then the agent (agent.py) and its guard.

Yields events for the frontend in the order documented in README section 6."""

from __future__ import annotations

import ast
import time
from typing import Iterator

from app import config
from app.agent import fixer
from app.agent.agent import Workbench, default_report, guard, run_agent
from app.agent.goal_reader import read_goal, summarize
from app.agent.patching import PatchError, apply_patch
from app.agent.verifier import Verifier
from app.analysis.facts import build_facts
from app.schemas import (
    AnalyzeRequest, AttemptEvent, Coverage, CoverageEvent, DoneEvent, ErrorEvent, Evidence, Finding, FindingEvent,
    GoalEvent, GoalProfile, NeedsInputEvent, Patch, ReportEvent, SkippedCheck, StepEvent, Suggestion, SuggestionEvent,
)
from app.tools.base import ToolContext, make_finding

IMPACT_ORDER = {"crash": 0, "silent": 1, "risk": 2}
TIER_ORDER = {"verified": 0, "probed": 1, "advisory": 2}


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


def collect_syntax_errors(code: str, limit: int = 10) -> list[tuple[int, str]]:
    """Python reports one syntax error at a time. To list several, each reported line is replaced by `pass`
    (keeping its indentation) and the file is parsed again."""
    lines = code.split("\n")
    errors: list[tuple[int, str]] = []
    for _ in range(limit):
        try:
            ast.parse("\n".join(lines))
            break
        except SyntaxError as e:
            line = e.lineno or 1
            if line > len(lines) or any(l == line for l, _ in errors):
                break
            errors.append((line, e.msg))
            text = lines[line - 1]
            indent = text[: len(text) - len(text.lstrip())]
            # A broken block header (def, if, for...) is followed by an indented body; keep the body valid with an
            # always-true block so the blanked line does not cause a second, fake "unexpected indent" error.
            following = next((l for l in lines[line:] if l.strip()), "")
            opens_block = len(following) - len(following.lstrip()) > len(indent)
            lines[line - 1] = indent + ("if True:" if opens_block else "pass")
    return errors


def _syntax_run(code: str, req: AnalyzeRequest, started: float) -> Iterator:
    errors = collect_syntax_errors(code)
    n = len(errors)
    yield StepEvent(phase="parse", message=f"The script does not parse: {n} syntax error{'s' if n != 1 else ''} found")
    found = _Findings()
    for line, msg in errors:
        found.add([make_finding(
            "EZ000", f"Syntax error: {msg}",
            f"Python cannot parse line {line} ({msg}), so the script stops before training starts.",
            line, [Evidence(tool="syntax", summary=f"ast.parse: {msg} at line {line}", lines=[line])],
        )])
    for f in found.items:
        yield FindingEvent(finding=f)
    goal_text = req.goal.strip() or "(not stated by the developer)"
    yield StepEvent(phase="repair", message="Repairing syntax errors")
    for f in found.items:
        status, detail, proposal, attempt, feedback = "not_attempted", "Repair unavailable", None, 0, None
        for attempt in range(1, config.FIX_MAX_ATTEMPTS + 1):
            try:
                proposal = fixer.propose(code, goal_text, GoalProfile(), f, feedback, proposal.patch if proposal else None)
            except Exception as e:
                status, detail, attempt = "not_attempted", f"Repair unavailable: {e}", attempt - 1
                break
            status, detail = _verify_syntax_fix(code, proposal.patch, f.line)
            yield AttemptEvent(finding_id=f.id, attempt=attempt, status=status, detail=detail)
            if status == "resolved":
                break
            feedback = detail
        yield SuggestionEvent(suggestion=Suggestion(
            finding=f, explanation=proposal.explanation if proposal else f.message,
            patch=proposal.patch if proposal else None, verify_status=status, verify_detail=detail,
            attempts=max(attempt, 0)))
    yield StepEvent(phase="report", message="Fix the syntax errors, then check again to run every other check")
    yield CoverageEvent(coverage=Coverage(checked=["syntax"], skipped=[
        SkippedCheck(check="all other checks", reason="they need the script to parse; apply the fixes and check again")]))
    yield DoneEvent(elapsed_ms=int((time.monotonic() - started) * 1000))


def _verify_syntax_fix(code: str, patch: Patch, line: int) -> tuple[str, str]:
    try:
        patched = apply_patch(code, patch)
    except PatchError as e:
        return "unresolved", f"Patch could not be applied: {e}"
    try:
        ast.parse(patched)
        return "resolved", "The patched script parses"
    except SyntaxError as e:
        in_patch = patch.start_line <= (e.lineno or 0) <= patch.start_line + patch.replacement.count("\n")
        if e.lineno == line or in_patch:
            return "unresolved", f"Still a syntax error at line {e.lineno}: {e.msg}"
        return "resolved", f"This error is fixed; Python now reports the next one at line {e.lineno}"


def _run(req: AnalyzeRequest, started: float) -> Iterator:
    code = req.code.replace("\r\n", "\n")
    try:
        facts = build_facts(code)
    except SyntaxError:
        yield from _syntax_run(code, req, started)
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
    wb = Workbench(req=req, ctx=ctx, goal_text=goal_text,
                   verifier=Verifier(code, profile, req.input_shape, dict(req.answers)))

    # The cheap deterministic rules always run first, so the agent starts from facts, not guesses.
    yield StepEvent(phase="baseline", message="Running deterministic checks")
    for name in ("run_static_rules", "check_goal_alignment"):
        yield from wb.run_check(name, {}, "Baseline: deterministic rules run on every check")

    yield StepEvent(phase="plan", message="Agent at work" + (" (offline: built-in plan)" if config.LLM_MODE == "mock" else ""))
    outcome = yield from run_agent(wb)
    if outcome.question is None:
        yield from guard(wb, outcome)
    if outcome.budget_hit:
        wb.note_skipped([SkippedCheck(check="agent", reason="the agent reached its step budget; the guard finished the job")])
    if facts.model_classes and "shape probe" not in wb.checked and not any(s.check == "shape probe" for s in wb.skipped):
        wb.note_skipped([SkippedCheck(check="shape probe", reason="not run for this script")])

    if outcome.question:
        yield CoverageEvent(coverage=Coverage(checked=wb.checked, skipped=wb.skipped))
        yield NeedsInputEvent(question=outcome.question)
        return

    report = outcome.report or default_report(wb)
    # The mode says who wrote the report: the agent model, or the built-in summary (offline, or after a fallback).
    mode = outcome.mode if outcome.report else ("offline" if config.LLM_MODE == "mock" else "fallback")
    yield ReportEvent(text=report, mode=mode)
    yield StepEvent(phase="report", message="Analysis complete")
    yield CoverageEvent(coverage=Coverage(checked=wb.checked, skipped=wb.skipped))
    yield DoneEvent(elapsed_ms=int((time.monotonic() - started) * 1000))
