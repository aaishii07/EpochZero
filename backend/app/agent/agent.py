"""The EpochZero agent.

One LLM-driven loop owns the whole job: it decides which checks to run, when to call the code reviewer, which
findings to repair, how to react when a repair fails verification, whether fixes work together, when to ask the
developer, and when it is done. Every action is a tool call, every tool result goes back to the agent, and it
finishes with a written report.

Deterministic tools decide what is proven. The LLM decides what to do next. A guard after the loop runs anything
essential the agent skipped, so a weak or rate-limited model can make the result slower, never less checked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Generator

from app import config
from app.agent import fixer, prompts, reviewer
from app.agent.planner import _compact_facts, _for_llm, _question_from, _with_reason
from app.agent.verifier import Verifier, verify_combination
from app.llm import mock
from app.llm.client import LLMError, tool_turn
from app.llm.text import plain
from app.schemas import (
    AgentNoteEvent, AnalyzeRequest, AttemptEvent, Finding, FindingEvent, FixProposal, Patch, Question, SkippedCheck,
    Suggestion, SuggestionEvent, ToolCallEvent, ToolResultEvent,
)
from app.tools.base import ToolContext, ToolResult, is_unknown
from app.tools.registry import TOOLS, as_llm_tools, run_tool

CHECK_NAMES = {
    "run_static_rules": "static rules",
    "check_goal_alignment": "goal alignment",
    "probe_shapes": "shape probe",
    "dry_run": "dry run (two training steps)",
    "check_class_count": "class count",
}
IMPACT_ORDER = {"crash": 0, "silent": 1, "risk": 2}
TIER_ORDER = {"verified": 0, "probed": 1, "advisory": 2}
MAX_AGENT_CALLS = 14
MAX_AGENT_SECONDS = 180
# The agent makes the decisions, so a short rate-limit wait is better than handing over to the built-in plan.
AGENT_MAX_WAIT = 30
MAX_ATTEMPTS = {"EZ100": 2}  # code-review fixes cannot be proven, so they get fewer retries; others use the default

AGENT_TOOLS = [
    {
        "name": "review_code",
        "description": "Ask the code reviewer (a separate LLM) to read the whole script for problems the rule "
                       "checks do not cover: pretrained loading, model output types, data pipeline, device, "
                       "checkpointing. Its findings are advisory. Call it once per check; pass a focus to point it "
                       "at an area, or null.",
        "input_schema": {"type": "object", "properties": {"focus": {"type": ["string", "null"]}},
                         "required": ["focus"], "additionalProperties": False},
    },
    {
        "name": "propose_fixes",
        "description": "Write fixes for the given finding ids and verify each one immediately (the same check that "
                       "found the problem is rerun on the patched code). Returns each fix's status: resolved, "
                       "unresolved (with the reason), or inconclusive. To retry an unresolved fix, call again with "
                       "guidance that addresses the reason.",
        "input_schema": {"type": "object", "properties": {
            "finding_ids": {"type": "array", "items": {"type": "string"}},
            "guidance": {"type": ["string", "null"]}},
            "required": ["finding_ids", "guidance"], "additionalProperties": False},
    },
    {
        "name": "test_fixes_together",
        "description": "Apply the given fixes together on a copy of the script and rerun the checks, adding them "
                       "one at a time. Returns which fixes are compatible and which break the script in "
                       "combination, with the reason. Use it before finishing when several fixes touch related code.",
        "input_schema": {"type": "object", "properties": {"finding_ids": {"type": "array", "items": {"type": "string"}}},
                         "required": ["finding_ids"], "additionalProperties": False},
    },
    {
        "name": "ask_user",
        "description": "Stop and ask the developer. Only allowed after a check result reports 'needs' "
                       "(input_shape from probe_shapes, num_classes from check_class_count); use that as question_id.",
        "input_schema": {"type": "object", "properties": {
            "question_id": {"type": "string"}, "prompt": {"type": "string"},
            "options": {"type": ["array", "null"], "items": {"type": "string"}}},
            "required": ["question_id", "prompt", "options"], "additionalProperties": False},
    },
    {
        "name": "finish",
        "description": "End the job with a report for the developer: what is wrong with the script, what you fixed "
                       "and how sure you are, what still needs their decision, and the order to apply the fixes in. "
                       "Plain sentences, at most 120 words.",
        "input_schema": {"type": "object", "properties": {"report": {"type": "string"}},
                         "required": ["report"], "additionalProperties": False},
    },
]


@dataclass
class Workbench:
    """Everything the agent works on during one check, shared with the orchestrator."""
    req: AnalyzeRequest
    ctx: ToolContext
    goal_text: str
    verifier: Verifier
    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: list[SkippedCheck] = field(default_factory=list)
    suggestions: dict[str, Suggestion] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    review_patches: dict[str, Patch] = field(default_factory=dict)
    reviewed: bool = False
    ran: set[str] = field(default_factory=set)
    _keys: set[tuple] = field(default_factory=set)

    @property
    def code(self) -> str:
        return self.ctx.code

    def by_id(self, fid: str) -> Finding | None:
        return next((f for f in self.findings if f.id == fid), None)

    def sorted(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (IMPACT_ORDER[f.impact], TIER_ORDER[f.tier], f.line))

    def add(self, found: list[Finding]) -> list[Finding]:
        new = []
        for f in found:
            key = (f.rule_id, f.line, f.title)
            if key in self._keys:
                continue
            self._keys.add(key)
            f.id = f"f{len(self.findings) + 1}"
            self.findings.append(f)
            new.append(f)
        return new

    def note_skipped(self, items: list[SkippedCheck]):
        self.skipped.extend(s for s in items if s not in self.skipped)

    # ----- tools -----

    def run_check(self, name: str, args: dict, reason: str) -> Generator:
        yield ToolCallEvent(tool=name, args=args, reason=reason)
        result = run_tool(name, self.ctx, **args)
        self.ran.add(name)
        if result.ok and CHECK_NAMES.get(name) and CHECK_NAMES[name] not in self.checked:
            self.checked.append(CHECK_NAMES[name])
        self.note_skipped(result.skipped)
        new = self.add(result.findings)
        yield ToolResultEvent(tool=name, ok=result.ok, summary=result.summary, finding_ids=[f.id for f in new])
        for f in new:
            yield FindingEvent(finding=f)
        return result, new

    def review(self, focus: str | None, reason: str) -> Generator:
        yield ToolCallEvent(tool="code_review", args={"focus": focus} if focus else {}, reason=reason)
        self.reviewed = True
        goal = self.goal_text + (f"\nFocus especially on: {focus}" if focus else "")
        try:
            reviewed = reviewer.review(self.code, goal, list(self.findings), self.req.history, self.req.previous_open)
        except Exception as e:
            why = str(e)[:120] or type(e).__name__
            self.note_skipped([SkippedCheck(check="code review", reason=f"the reviewer was unavailable ({why})")])
            yield ToolResultEvent(tool="code_review", ok=False, summary=f"Code review unavailable: {why}")
            return None, []
        new = self.add([f for f, _ in reviewed])
        self.review_patches.update({f.id: p for f, p in reviewed if f in new and p is not None})
        if "code review" not in self.checked:
            self.checked.append("code review")
        yield ToolResultEvent(tool="code_review", ok=True, summary=f"{len(new)} more issue{'s' if len(new) != 1 else ''} found",
                              finding_ids=[f.id for f in new])
        for f in new:
            yield FindingEvent(finding=f)
        return True, new

    def propose(self, ids: list[str], guidance: str | None, reason: str) -> Generator:
        yield ToolCallEvent(tool="propose_fixes", args={"finding_ids": ids, **({"guidance": guidance} if guidance else {})},
                            reason=reason)
        results = {}
        for fid in ids:
            f = self.by_id(fid)
            if f is None:
                results[fid] = {"status": "error", "detail": "unknown finding id"}
                continue
            if not f.patchable:
                s = Suggestion(finding=f, explanation=f.message, patch=None, verify_status="not_attempted",
                               verify_detail="Advisory finding; no automatic change is proposed" if f.tier == "advisory"
                               else "Not patched automatically; the right change depends on your intent", attempts=0)
                self.suggestions[fid] = s
                yield SuggestionEvent(suggestion=s)
                results[fid] = {"status": "not_patchable"}
                continue
            limit = MAX_ATTEMPTS.get(f.rule_id, config.FIX_MAX_ATTEMPTS)
            done = self.attempts.get(fid, 0)
            prev = self.suggestions.get(fid)
            if prev and prev.verify_status == "resolved":
                results[fid] = {"status": "resolved", "detail": "already resolved"}
                continue
            if done >= limit:
                results[fid] = {"status": prev.verify_status if prev else "not_attempted",
                                "detail": f"attempt limit ({limit}) reached"}
                continue
            feedback = None
            if prev and prev.verify_status == "unresolved":
                feedback = prev.verify_detail
            if guidance:
                feedback = f"{feedback + '. ' if feedback else ''}Guidance from the agent: {guidance}"
            try:
                if done == 0 and fid in self.review_patches and not guidance:
                    proposal = FixProposal(explanation=f.message, patch=self.review_patches[fid])
                else:
                    proposal = fixer.propose(self.code, self.goal_text, self.ctx.profile, f, feedback,
                                             prev.patch if prev else None,
                                             others=[o for o in self.findings if o.id != fid], history=self.req.history)
            except Exception as e:
                results[fid] = {"status": "error", "detail": f"repair unavailable: {str(e)[:100]}"}
                if not prev:
                    s = Suggestion(finding=f, explanation=f.message, patch=None, verify_status="not_attempted",
                                   verify_detail=f"Repair unavailable: {str(e)[:100]}", attempts=done)
                    self.suggestions[fid] = s
                    yield SuggestionEvent(suggestion=s)
                continue
            self.attempts[fid] = done + 1
            status, detail = self.verifier.verify(proposal.patch, f)
            yield AttemptEvent(finding_id=fid, attempt=done + 1, status=status, detail=detail)
            # Keep the better of the previous and the new attempt.
            if not prev or status in ("resolved", "inconclusive") or prev.verify_status == "unresolved":
                s = Suggestion(finding=f, explanation=proposal.explanation or f.message, patch=proposal.patch,
                               verify_status=status, verify_detail=detail, attempts=done + 1)
                self.suggestions[fid] = s
                yield SuggestionEvent(suggestion=s)
            results[fid] = {"status": status, "detail": detail[:200], "attempts": done + 1, "limit": limit}
        yield ToolResultEvent(tool="propose_fixes", ok=True, summary=_fix_summary(results))
        return results

    def test_together(self, ids: list[str], reason: str) -> Generator:
        yield ToolCallEvent(tool="test_fixes_together", args={"finding_ids": ids}, reason=reason)
        items, missing = [], []
        for fid in ids:
            s = self.suggestions.get(fid)
            if s and s.patch and s.verify_status in ("resolved", "inconclusive"):
                items.append((fid, s.patch))
            else:
                missing.append(fid)
        # Overlapping patches cannot be combined; keep the first of each overlap.
        chosen: list[tuple[str, Patch]] = []
        for fid, p in items:
            if any(p.start_line <= q.end_line and q.start_line <= p.end_line for _, q in chosen):
                missing.append(fid)
            else:
                chosen.append((fid, p))
        result = verify_combination(self.code, self.ctx.profile, self.req.input_shape, dict(self.req.answers), chosen)
        result["not_tested"] = missing
        yield ToolResultEvent(tool="test_fixes_together", ok=True, summary=result["summary"])
        return result

    # ----- views for the LLM -----

    def findings_view(self) -> list[dict]:
        out = []
        for f in self.sorted():
            s = self.suggestions.get(f.id)
            out.append({"id": f.id, "rule": f.rule_id, "tier": f.tier, "impact": f.impact, "area": f.category,
                        "line": f.line, "title": f.title, "patchable": f.patchable,
                        "fix": s.verify_status if s else "none", "attempts": self.attempts.get(f.id, 0)})
        return out


def _fix_summary(results: dict) -> str:
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counts.items())) or "nothing to fix"


def _initial_message(wb: Workbench) -> str:
    ctx = wb.ctx
    answers = {k: ("the developer does not know" if is_unknown(v) else v) for k, v in ctx.answers.items()}
    parts = [
        f"Developer goal: {wb.goal_text}",
        f"Task profile: {ctx.profile.model_dump_json()}",
        f"Input shape provided: {ctx.input_shape}",
        f"Developer answers so far: {json.dumps(answers)}",
        f"Findings so far (from the static rules, already run): {json.dumps(wb.findings_view())}",
    ]
    from app.agent import memory
    if wb.req.history:
        parts.append(memory.history_text(wb.req.history, wb.code))
    if wb.req.previous_open:
        parts.append(memory.open_text(wb.req.previous_open))
    parts.append(f"Script summary (use inspect_lines to read source): {json.dumps(_compact_facts(ctx), default=str)}")
    if config.LLM_MODE == "mock":
        parts.append("(offline)")
    return "\n\n".join(parts)


KEEP_FULL_RESULTS = 3
SHORT_RESULT_CHARS = 280


def _compact(messages: list) -> None:
    """Shorten all but the most recent tool results. Every turn resends the whole conversation, so this keeps each
    request small and the free-tier token budget lasting; the agent still sees what each older step returned."""
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    for m in tool_msgs[:-KEEP_FULL_RESULTS]:
        content = m["content"]
        if len(content) > SHORT_RESULT_CHARS and not content.endswith("[shortened]"):
            m["content"] = content[:SHORT_RESULT_CHARS] + " ... [shortened]"


@dataclass
class AgentOutcome:
    question: Question | None = None
    report: str = ""
    mode: str = "llm"
    budget_hit: bool = False


def run_agent(wb: Workbench) -> Generator:
    """Yields events; returns AgentOutcome."""
    if config.LLM_MODE == "mock":
        return (yield from _scripted(wb))

    outcome = AgentOutcome()
    tools = _with_reason(as_llm_tools()[:-2] + AGENT_TOOLS)  # check tools without the planner's ask/finish, plus agent tools
    messages: list = [{"role": "user", "content": _initial_message(wb)}]
    started, calls, seen = time.monotonic(), 0, set()
    pending_needs: list[ToolResult] = []
    yield AgentNoteEvent(kind="thought", text="Starting. The static rules have run; deciding what to check next.")

    while True:
        if calls >= MAX_AGENT_CALLS or time.monotonic() - started > MAX_AGENT_SECONDS:
            outcome.budget_hit = True
            yield AgentNoteEvent(kind="thought", text="Step budget reached; wrapping up.")
            break
        _compact(messages)
        try:
            turn = tool_turn(prompts.AGENT, messages, tools, effort="low", max_wait=AGENT_MAX_WAIT)
        except (LLMError, Exception) as e:
            if calls == 0:
                yield AgentNoteEvent(kind="thought", text=f"The agent model is unavailable ({str(e)[:80]}); continuing with the built-in plan.")
                outcome = yield from _scripted(wb)
                outcome.mode = "fallback"
                return outcome
            yield AgentNoteEvent(kind="thought", text=f"The agent model stopped responding ({str(e)[:80]}); finishing with what is done.")
            break
        messages.append(turn.assistant_message)
        if turn.text.strip():
            yield AgentNoteEvent(kind="thought", text=plain(turn.text)[:400])
        if not turn.tool_calls:
            break
        done = False

        def reply(call_id: str, content: str):
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

        for call in turn.tool_calls:
            args = dict(call.args)
            reason = plain(str(args.pop("reason", "")).strip())
            calls += 1
            if reason:
                yield AgentNoteEvent(kind="thought", text=reason)
            if call.name == "finish":
                outcome.report = plain(str(args.get("report", "")))
                done = True
                reply(call.id, "ok")
                continue
            if call.name == "ask_user":
                qid = str(args.get("question_id") or "clarification")
                if qid in wb.ctx.answers or qid not in {r.needs for r in pending_needs}:
                    reply(call.id, f"Error: no open check needs {qid}; run the relevant check first.")
                    continue
                options = args.get("options")
                outcome.question = Question(id=qid, prompt=str(args.get("prompt", "")),
                                            options=[str(o) for o in options] if isinstance(options, list) and options else None)
                done = True
                reply(call.id, "question sent")
                continue
            if call.name == "review_code":
                if wb.reviewed:
                    reply(call.id, "Error: the code review already ran in this check.")
                    continue
                ok, new = yield from wb.review(args.get("focus"), reason)
                reply(call.id, json.dumps({"ok": bool(ok), "new_findings": [
                    {"id": f.id, "line": f.line, "title": f.title, "impact": f.impact, "patchable": f.patchable} for f in new]}))
                continue
            if call.name == "propose_fixes":
                ids = [str(i) for i in args.get("finding_ids") or []]
                results = yield from wb.propose(ids, args.get("guidance"), reason)
                reply(call.id, json.dumps(results)[:4000])
                continue
            if call.name == "test_fixes_together":
                ids = [str(i) for i in args.get("finding_ids") or []]
                result = yield from wb.test_together(ids, reason)
                reply(call.id, json.dumps(result)[:3000])
                continue
            if call.name not in TOOLS:
                reply(call.id, "Error: unknown tool")
                continue
            if call.name == "probe_shapes" and not isinstance(args.get("input_shape"), list):
                args = {"input_shape": None}
            key = (call.name, json.dumps(args, sort_keys=True))
            if key in seen:
                reply(call.id, "Error: already run with these arguments; the result has not changed.")
                continue
            seen.add(key)
            result, new = yield from wb.run_check(call.name, args, reason)
            if result.needs:
                pending_needs.append(result)
            payload = json.loads(_for_llm(result))
            payload["new_findings"] = [{"id": f.id, "line": f.line, "title": f.title, "patchable": f.patchable} for f in new]
            reply(call.id, json.dumps(payload, default=str)[:4000])
        if done:
            break

    if outcome.question is None:
        for result in pending_needs:
            q = _question_from(result)
            if q and q.id not in wb.ctx.answers:
                outcome.question = q
                break
    return outcome


def _scripted(wb: Workbench) -> Generator:
    """The same tools in a fixed order, used offline or when the agent model is unavailable."""
    outcome = AgentOutcome(mode="offline" if config.LLM_MODE == "mock" else "fallback")
    yield AgentNoteEvent(kind="thought", text="Following the built-in plan: measure the model, run a training step, "
                                              "check class counts, then repair and test the fixes together.")
    reasons = {
        "probe_shapes": "Measure the real output shape and catch any layer that rejects its input.",
        "dry_run": "Run two real training steps to check the loss, targets, and gradients.",
        "check_class_count": "Compare the measured head size with every class count in the code and goal.",
    }
    f = wb.ctx.facts
    for name in mock.plan_steps(bool(f.model_classes), f.head is not None, wb.ctx.profile.task):
        result, _ = yield from wb.run_check(name, {}, reasons[name])
        q = _question_from(result)
        if q and q.id not in wb.ctx.answers:
            outcome.question = q
            return outcome
    return outcome


def guard(wb: Workbench, outcome: AgentOutcome) -> Generator:
    """Run what the agent skipped: core checks, the code review, and a repair attempt for every patchable finding."""
    f = wb.ctx.facts
    for name in mock.plan_steps(bool(f.model_classes), f.head is not None, wb.ctx.profile.task):
        if name not in wb.ran:
            result, _ = yield from wb.run_check(name, {}, "Guard: core check the agent did not run")
            q = _question_from(result)
            if q and q.id not in wb.ctx.answers and outcome.question is None:
                outcome.question = q
                return
    if config.LLM_MODE == "mock":
        wb.note_skipped([SkippedCheck(check="code review", reason="needs the LLM; the backend is in offline mode")])
    elif not wb.reviewed:
        yield from wb.review(None, "Guard: the code review always runs")
    missing = [x.id for x in wb.sorted() if x.id not in wb.suggestions]
    if missing:
        yield from wb.propose(missing, None, "Guard: every finding gets a repair attempt")
    ready = [x.id for x in wb.sorted() if (s := wb.suggestions.get(x.id)) and s.patch
             and s.verify_status in ("resolved", "inconclusive")]
    if len(ready) > 1 and not outcome.report:
        yield from wb.test_together(ready, "Guard: check the proposed fixes work together")


def default_report(wb: Workbench) -> str:
    sugg = list(wb.suggestions.values())
    resolved = sum(s.verify_status == "resolved" for s in sugg)
    unverified = sum(s.verify_status == "inconclusive" for s in sugg)
    failed = sum(s.verify_status == "unresolved" for s in sugg)
    manual = sum(s.verify_status == "not_attempted" for s in sugg)
    crash = sum(f.impact == "crash" for f in wb.findings)
    if not wb.findings:
        return "No problems found in the checks that ran."
    parts = [f"Found {len(wb.findings)} problem{'s' if len(wb.findings) != 1 else ''}"
             + (f", {crash} of which stop the run" if crash else "") + "."]
    def are(n: int) -> str:
        return "is" if n == 1 else "are"

    if resolved:
        parts.append(f"{resolved} fix{'es' if resolved != 1 else ''} verified.")
    if unverified:
        parts.append(f"{unverified} fix{'es' if unverified != 1 else ''} from the code review {are(unverified)} "
                     "unverified; review before applying.")
    if failed:
        parts.append(f"{failed} could not be fixed automatically.")
    if manual:
        parts.append(f"{manual} {'needs' if manual == 1 else 'need'} your decision.")
    parts.append("Apply crash fixes first, then check again.")
    return " ".join(parts)
