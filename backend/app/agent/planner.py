"""The investigation loop: the LLM chooses tools one at a time until it finishes, asks the developer, or hits the budget."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Generator

from app import config
from app.agent import prompts
from app.llm import mock
from app.llm.client import LLMError, tool_turn
from app.schemas import Finding, Question, SkippedCheck, ToolCallEvent
from app.tools.base import ToolContext, ToolResult, is_unknown
from app.tools.registry import TOOLS, as_llm_tools, run_tool


@dataclass
class PlanResult:
    results: list[tuple[str, ToolResult]] = field(default_factory=list)
    question: Question | None = None
    skipped: list[SkippedCheck] = field(default_factory=list)
    budget_hit: bool = False
    mode: str = "llm"


def _with_reason(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        schema = json.loads(json.dumps(t["input_schema"]))
        schema["properties"]["reason"] = {"type": "string", "description": "One sentence: what you expect to learn."}
        schema["required"] = list(schema.get("required", [])) + ["reason"]
        out.append({**t, "input_schema": schema})
    return out


def _for_llm(result: ToolResult) -> str:
    data = {k: v for k, v in result.data.items() if k not in {"records", "text"}}
    payload = {
        "ok": result.ok,
        "summary": result.summary,
        "findings": [{"rule_id": f.rule_id, "title": f.title, "message": f.message, "line": f.line,
                      "patchable": f.patchable} for f in result.findings],
        "data": data,
        "needs": result.needs,
        "question": result.question,
    }
    if "text" in result.data:
        payload["text"] = result.data["text"]
    return json.dumps(payload, default=str)[:5000]


def _compact_facts(ctx: ToolContext) -> dict:
    """A small view of the script. The planner reads source lines on demand with inspect_lines, which keeps each
    request well inside free-tier token limits."""
    f = ctx.facts
    return {
        "lines": len(f.lines),
        "model_classes": [f"{m.name}@{m.line}" for m in f.model_classes],
        "layers": [f"{l.name}={l.type}{tuple(l.args)}@{l.line}" for l in f.layers][:40],
        "head": f"{f.head.name}={f.head.type} out={f.head_out}@{f.head.line}" if f.head else None,
        "final_activation": f"{f.final_activation.kind}@{f.final_activation.line}" if f.final_activation else None,
        "loss": f"{f.loss.type}@{f.loss.line}" if f.loss else None,
        "optimizer": f"{f.optimizer.type} lr={f.optimizer.lr}@{f.optimizer.line}" if f.optimizer else None,
        "loops": [f"{l.kind}@{l.line}: " + ",".join(c.name for c in l.calls)[:160] for l in f.loops][:8],
        "transforms": [f"{t.var}@{t.line}: " + ",".join(i["type"] for i in t.items) for t in f.transforms][:6],
        "class_count_sources": [f"{c.kind}={c.value}@{c.line}" for c in f.class_count_sources],
        "data_hints": f.data_hints,
    }


def _initial_prompt(ctx: ToolContext, goal: str, baseline: list[Finding]) -> str:
    base = [{"rule_id": f.rule_id, "title": f.title, "line": f.line} for f in baseline]
    answers = {k: ("the developer does not know" if is_unknown(v) else v) for k, v in ctx.answers.items()}
    return "\n\n".join([
        f"Developer goal: {goal}",
        f"Task profile: {ctx.profile.model_dump_json()}",
        f"Input shape provided: {ctx.input_shape}",
        f"Developer answers so far: {json.dumps(answers)}",
        f"Baseline static findings: {json.dumps(base)}",
        f"Script summary (use inspect_lines to read source): {json.dumps(_compact_facts(ctx), default=str)}",
    ])


def _execute(ctx: ToolContext, name: str, args: dict, reason: str, plan: PlanResult) -> Generator:
    yield ToolCallEvent(tool=name, args=args, reason=reason)
    result = run_tool(name, ctx, **args)
    plan.results.append((name, result))
    plan.skipped += result.skipped
    # The orchestrator turns this into a ToolResultEvent plus FindingEvents with ids assigned.
    yield ("result", name, result)
    return result


def _question_from(result: ToolResult) -> Question | None:
    if not result.needs:
        return None
    options = result.data.get("options") if result.needs == "num_classes" else None
    return Question(id=result.needs, prompt=result.question or f"Please provide {result.needs}.", options=options)


def _offline(ctx: ToolContext) -> Generator:
    plan = PlanResult(mode="offline")
    f = ctx.facts
    reasons = {
        "probe_shapes": "Measure the real output shape and catch any layer that rejects its input.",
        "check_class_count": "Compare the measured head size with every class count in the code and goal.",
    }
    for name in mock.plan_steps(bool(f.model_classes), f.head is not None, ctx.profile.task):
        result = yield from _execute(ctx, name, {}, reasons[name], plan)
        q = _question_from(result)
        if q and q.id not in ctx.answers:
            plan.question = q
            break
    return plan


def investigate(ctx: ToolContext, goal: str, baseline: list[Finding]) -> Generator:
    """Yields ToolCallEvent / ToolResultEvent; returns PlanResult."""
    if config.LLM_MODE == "mock":
        return (yield from _offline(ctx))

    plan = PlanResult()
    tools = _with_reason(as_llm_tools())
    messages: list = [{"role": "user", "content": _initial_prompt(ctx, goal, baseline)}]
    started, calls, seen = time.monotonic(), 0, set()
    pending_needs: list[ToolResult] = []

    while True:
        if calls >= config.PLANNER_MAX_TOOL_CALLS or time.monotonic() - started > config.PLANNER_MAX_SECONDS:
            plan.budget_hit = True
            break
        try:
            turn = tool_turn(prompts.PLANNER, messages, tools, effort="low")
        except (LLMError, Exception):
            if not plan.results:
                return (yield from _offline(ctx))
            break
        messages.append(turn.assistant_message)
        if not turn.tool_calls:
            break
        done = False

        def reply(call_id: str, content: str):
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

        for call in turn.tool_calls:
            args = dict(call.args)
            reason = str(args.pop("reason", "")).strip()
            calls += 1
            if call.name == "finish":
                done = True
                reply(call.id, "ok")
                continue
            if call.name == "ask_user":
                qid = str(args.get("question_id") or "clarification")
                if qid in ctx.answers:
                    reply(call.id, f"Error: the developer already answered {qid}: {ctx.answers[qid]}")
                    continue
                if qid not in {r.needs for r in pending_needs}:
                    # Questions must come from a check that reported missing information, never from guessing.
                    reply(call.id, f"Error: no check has reported that it needs {qid}. Run the relevant check "
                                   "first (probe_shapes for input_shape, check_class_count for num_classes).")
                    continue
                options = args.get("options")
                plan.question = Question(id=qid, prompt=str(args.get("prompt", "")),
                                         options=[str(o) for o in options] if isinstance(options, list) and options else None)
                done = True
                reply(call.id, "question sent")
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
            result = yield from _execute(ctx, call.name, args, reason, plan)
            if result.needs:
                pending_needs.append(result)
            reply(call.id, _for_llm(result))
        if done:
            break

    # Coverage guard: the core checks always run, even if the planner finished without them.
    if plan.question is None:
        ran = {name for name, _ in plan.results}
        for name in mock.plan_steps(bool(ctx.facts.model_classes), ctx.facts.head is not None, ctx.profile.task):
            if name in ran:
                continue
            result = yield from _execute(ctx, name, {}, "Coverage guard: core check not yet run by the planner", plan)
            if result.needs:
                pending_needs.append(result)

    # Safety net: a tool said the fix would be a guess, and the planner did not ask. Ask rather than guess.
    if plan.question is None:
        for result in pending_needs:
            q = _question_from(result)
            if q and q.id not in ctx.answers:
                plan.question = q
                break
    return plan
