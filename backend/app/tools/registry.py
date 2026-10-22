"""The single tool menu used by both the planner (via as_llm_tools) and the verifier (via run_tool)."""

from __future__ import annotations

from typing import Callable

from app.tools.base import ToolContext, ToolResult
from app.tools.dynamic import shape_probe
from app.tools.static import ast_rules, class_count, goal_rules, hyperparams


def _static(ctx: ToolContext) -> ToolResult:
    findings = ast_rules.run(ctx) + hyperparams.run(ctx)
    n = len(findings)
    return ToolResult(True, "No static issues found" if not n else f"{n} static issue{'s' if n != 1 else ''} found", findings=findings)


def _goal(ctx: ToolContext) -> ToolResult:
    if ctx.profile.task == "unknown":
        return ToolResult(True, "Goal task unknown; goal alignment skipped")
    findings = goal_rules.run(ctx)
    return ToolResult(True, f"Loss and output checked against {ctx.profile.task}" if not findings
                      else f"{len(findings)} goal mismatch{'es' if len(findings) != 1 else ''} found", findings=findings)


def _inspect(ctx: ToolContext, start: int, end: int) -> ToolResult:
    lines = ctx.facts.lines
    start, end = max(1, start), min(len(lines), end, start + 60)
    text = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return ToolResult(True, f"Lines {start}-{end}", data={"text": text})


TOOLS: dict[str, dict] = {
    "run_static_rules": {
        "fn": lambda ctx, **_: _static(ctx),
        "description": "Run the deterministic static rules (activation and loss pairing, augmentation in evaluation "
                       "transforms, zero_grad placement, loss accumulation, evaluation mode, target dtype, hyperparameters).",
        "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    "check_goal_alignment": {
        "fn": lambda ctx, **_: _goal(ctx),
        "description": "Compare the loss function and output activation with the expectations for the stated task.",
        "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    "probe_shapes": {
        "fn": lambda ctx, input_shape=None, **_: shape_probe.run(ctx, input_shape),
        "description": "Build the model and run one forward pass on meta tensors (no memory, no GPU). Returns the "
                       "output shape, or the layer that rejects its input. Pass input_shape to override the inferred "
                       "shape, or null to use the provided or inferred one.",
        "schema": {"type": "object", "properties": {"input_shape": {"type": ["array", "null"], "items": {"type": "integer"}}},
                   "required": ["input_shape"], "additionalProperties": False},
    },
    "check_class_count": {
        "fn": lambda ctx, **_: class_count.run(ctx),
        "description": "Gather every statement of the number of classes (constants, class lists, loss weights, goal, "
                       "measured head output) and report whether they agree. Run probe_shapes first so the measured "
                       "head output is included.",
        "schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    "inspect_lines": {
        "fn": lambda ctx, start=1, end=1, **_: _inspect(ctx, int(start), int(end)),
        "description": "Return the source lines from start to end (inclusive, at most 60 lines).",
        "schema": {"type": "object", "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                   "required": ["start", "end"], "additionalProperties": False},
    },
}

# Evidence tool name -> registry tool that reproduces it. Order matters: shape probe before class count.
EVIDENCE_TO_TOOL = {
    "shape_probe": "probe_shapes",
    "class_count": "check_class_count",
    "static_rules": "run_static_rules",
    "hyperparameters": "run_static_rules",
    "goal_alignment": "check_goal_alignment",
}
TOOL_ORDER = ["probe_shapes", "run_static_rules", "check_goal_alignment", "check_class_count"]

PSEUDO_TOOLS = [
    {
        "name": "ask_user",
        "description": "Stop investigating and ask the developer a question. Use when required information is missing "
                       "or when sources disagree and any fix would be a guess. question_id must be 'num_classes', "
                       "'input_shape', or a short snake_case id.",
        "input_schema": {"type": "object", "properties": {
            "question_id": {"type": "string"}, "prompt": {"type": "string"},
            "options": {"type": ["array", "null"], "items": {"type": "string"}}},
            "required": ["question_id", "prompt", "options"], "additionalProperties": False},
    },
    {
        "name": "finish",
        "description": "End the investigation when further tools would not change the outcome.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                         "required": ["summary"], "additionalProperties": False},
    },
]


def as_llm_tools() -> list[dict]:
    tools = [{"name": n, "description": t["description"], "input_schema": t["schema"]} for n, t in TOOLS.items()]
    return [{**t, "strict": True} for t in tools + PSEUDO_TOOLS]


def run_tool(name: str, ctx: ToolContext, **args) -> ToolResult:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(False, f"Unknown tool {name}")
    try:
        return tool["fn"](ctx, **args)
    except Exception as e:
        return ToolResult(False, f"{name} failed: {type(e).__name__}: {e}")


def tools_for_findings(evidence_tools: set[str]) -> list[str]:
    wanted = {EVIDENCE_TO_TOOL[t] for t in evidence_tools if t in EVIDENCE_TO_TOOL}
    if "check_class_count" in wanted:
        wanted.add("probe_shapes")
    return [t for t in TOOL_ORDER if t in wanted]
