"""Helpers shared by the agent: tool schemas with a reason field, compact views of results and code facts.

The investigation loop that used to live here is now the agent loop in agent.py.
"""

from __future__ import annotations

import json

from app.schemas import Question
from app.tools.base import ToolContext, ToolResult


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


def _question_from(result: ToolResult) -> Question | None:
    if not result.needs:
        return None
    options = result.data.get("options") if result.needs == "num_classes" else None
    return Question(id=result.needs, prompt=result.question or f"Please provide {result.needs}.", options=options)
