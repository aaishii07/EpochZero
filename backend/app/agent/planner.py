"""The investigation loop: the LLM chooses tools one at a time until it finishes, asks the developer, or hits the budget."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Generator

from app import config
from app.agent import prompts
from app.agent.fixer import numbered
from app.llm import mock
from app.llm.client import LLMError, tool_turn
from app.schemas import Finding, Question, SkippedCheck, ToolCallEvent, ToolResultEvent
from app.tools.base import ToolContext, ToolResult
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


def _initial_prompt(ctx: ToolContext, goal: str, baseline: list[Finding]) -> str:
    facts = json.dumps(ctx.facts.summary(), default=str)[:9000]
    base = [{"rule_id": f.rule_id, "title": f.title, "line": f.line} for f in baseline]
    return "\n\n".join([
        f"Developer goal: {goal}",
        f"Task profile: {ctx.profile.model_dump_json()}",
        f"Input shape provided: {ctx.input_shape}",
        f"Developer answers so far: {json.dumps(ctx.answers)}",
        f"Baseline static findings: {json.dumps(base)}",
        f"Extracted facts: {facts}",
        f"Script:\n{numbered(ctx.code)}",
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
            resp = tool_turn(prompts.PLANNER, messages, tools)
        except (LLMError, Exception):
            if not plan.results:
                return (yield from _offline(ctx))
            break
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not uses:
            break
        tool_results, done = [], False
        for use in uses:
            args = dict(use.input or {})
            reason = str(args.pop("reason", "")).strip()
            calls += 1
            if use.name == "finish":
                done = True
                tool_results.append({"type": "tool_result", "tool_use_id": use.id, "content": "ok"})
                continue
            if use.name == "ask_user":
                qid = str(args.get("question_id") or "clarification")
                if qid in ctx.answers:
                    tool_results.append({"type": "tool_result", "tool_use_id": use.id, "is_error": True,
                                         "content": f"The developer already answered {qid}: {ctx.answers[qid]}"})
                    continue
                plan.question = Question(id=qid, prompt=str(args.get("prompt", "")), options=args.get("options") or None)
                done = True
                tool_results.append({"type": "tool_result", "tool_use_id": use.id, "content": "question sent"})
                continue
            if use.name not in TOOLS:
                tool_results.append({"type": "tool_result", "tool_use_id": use.id, "is_error": True, "content": "unknown tool"})
                continue
            key = (use.name, json.dumps(args, sort_keys=True))
            if key in seen:
                tool_results.append({"type": "tool_result", "tool_use_id": use.id, "is_error": True,
                                     "content": "Already run with these arguments; the result has not changed."})
                continue
            seen.add(key)
            result = yield from _execute(ctx, use.name, args, reason, plan)
            if result.needs:
                pending_needs.append(result)
            tool_results.append({"type": "tool_result", "tool_use_id": use.id, "content": _for_llm(result)})
        if done:
            break
        messages.append({"role": "user", "content": tool_results})

    # Safety net: a tool said the fix would be a guess, and the planner did not ask. Ask rather than guess.
    if plan.question is None:
        for result in pending_needs:
            q = _question_from(result)
            if q and q.id not in ctx.answers:
                plan.question = q
                break
    return plan
