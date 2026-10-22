"""EZ004: compare every statement of the class count in the code, the goal, and the measured head output.

The fix is only grounded when the sources clearly agree on one value. Otherwise the tool returns a question,
so the agent asks the developer instead of guessing.
"""

from __future__ import annotations

from collections import Counter

from app.schemas import Evidence
from app.tools.base import ToolContext, ToolResult, make_finding

TOOL = "class_count"
RELEVANT_TASKS = {"multiclass_classification", "multilabel_classification", "segmentation", "unknown"}
KIND_LABEL = {
    "constant": "constant", "keyword": "num_classes argument", "class_list": "class list",
    "loss_weight": "loss weight", "head": "model head", "goal": "stated goal", "user": "your answer",
}


def run(ctx: ToolContext) -> ToolResult:
    f, profile = ctx.facts, ctx.profile
    if profile.task not in RELEVANT_TASKS:
        return ToolResult(True, f"Class count check not relevant for {profile.task}")
    if f.head is None:
        return ToolResult(True, "No classification or segmentation head found")

    head_value = f.head_out
    head_detail = f"{f.head.name} = {f.head.type}(...) outputs {head_value}"
    measured = ctx.measured.get("output_shape")
    if measured and len(measured) >= 2 and isinstance(measured[1], int):
        head_value = measured[1]
        head_detail = f"{f.head.name} measured output {tuple(measured)}"
    if head_value is None:
        return ToolResult(True, "Head output size could not be determined")

    sources = [{"kind": s.kind, "value": s.value, "line": s.line, "detail": s.detail}
               for s in f.class_count_sources if s.kind != "head"]
    if profile.num_classes:
        sources.append({"kind": "goal", "value": profile.num_classes, "line": None, "detail": f"goal states {profile.num_classes} classes"})
    user = _int(ctx.answers.get("num_classes"))
    if user:
        sources.append({"kind": "user", "value": user, "line": None, "detail": f"you answered {user}"})
    head = {"kind": "head", "value": head_value, "line": f.head.line, "detail": head_detail}
    all_sources = [head] + sources
    data = {"sources": all_sources, "head_value": head_value}

    if not sources:
        return ToolResult(True, f"Head outputs {head_value}; no other class-count source to compare", data=data)

    if user:
        target, status = user, "grounded"
    else:
        counts = Counter(s["value"] for s in sources)
        goal_value = profile.num_classes
        (top, top_n), *rest = counts.most_common()
        clear_majority = not rest or top_n > rest[0][1]
        if goal_value and (counts.get(goal_value, 0) >= max(counts.values()) or len(counts) == 1):
            target, status = goal_value, "grounded"
        elif len(counts) == 1 and top_n >= 2:
            target, status = top, "grounded"
        elif clear_majority and top_n >= 2 and not goal_value:
            target, status = top, "grounded"
        else:
            target, status = None, "ambiguous"

    if status == "grounded":
        disagreeing = [s for s in all_sources if s["value"] != target and s["line"] is not None]
        if not disagreeing:
            return ToolResult(True, f"All {len(all_sources)} class-count sources agree on {target}", data=data)
        primary = next((s for s in disagreeing if s["kind"] == "head"), disagreeing[0])
        agreeing = [s for s in all_sources if s["value"] == target]
        crash = any(s["kind"] == "loss_weight" for s in disagreeing) or (primary["kind"] == "head" and any(s["kind"] == "loss_weight" for s in agreeing))
        agree_text = ", ".join(f"{KIND_LABEL[s['kind']]}" + (f" (line {s['line']})" if s["line"] else "") for s in agreeing)
        finding = make_finding(
            "EZ004", f"{KIND_LABEL[primary['kind']].capitalize()} uses {primary['value']} classes, the rest of the code uses {target}",
            f"{primary['detail']}, but {agree_text} state {target} classes. "
            f"The {KIND_LABEL[primary['kind']]} at line {primary['line']} should use {target}.",
            primary["line"],
            [Evidence(tool=TOOL, summary=f"{len(agreeing)} sources agree on {target}; {KIND_LABEL[primary['kind']]} says {primary['value']}",
                      lines=sorted({s["line"] for s in all_sources if s["line"]}), data={**data, "target_value": target, "fix_kind": primary["kind"]})]
            + ([Evidence(tool="shape_probe", summary=f"Measured model output {tuple(measured)}", data={"output_shape": measured})] if measured else []),
            impact="crash" if crash else "silent",
        )
        return ToolResult(True, f"Class count mismatch: {primary['kind']} says {primary['value']}, grounded value is {target}",
                          findings=[finding], data={**data, "target_value": target})

    listing = "; ".join(f"{s['detail']}" + (f" (line {s['line']})" if s["line"] else "") for s in all_sources)
    question = (f"The class count is inconsistent: {listing}. How many classes is this model meant to predict?")
    finding = make_finding(
        "EZ004", "Class count is inconsistent and the intended value is unclear",
        f"Sources disagree: {listing}. EpochZero will not guess which one is intended.",
        f.head.line,
        [Evidence(tool=TOOL, summary="Class-count sources conflict without a clear majority",
                  lines=sorted({s["line"] for s in all_sources if s["line"]}), data=data)],
        patchable=False,
    )
    options = sorted({str(s["value"]) for s in all_sources})
    return ToolResult(True, "Class-count sources conflict; developer input needed", findings=[finding], data={**data, "options": options},
                      needs="num_classes", question=question)


def _int(value) -> int | None:
    try:
        v = int(str(value).strip())
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None
