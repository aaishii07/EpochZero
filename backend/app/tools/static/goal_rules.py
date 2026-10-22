"""EZ003: loss function and output activation checked against the stated task."""

from __future__ import annotations

from app.goals.expectations import EXPECTATIONS
from app.schemas import Evidence, Finding
from app.tools.base import ToolContext, make_finding

TOOL = "goal_alignment"
TASK_LABEL = {
    "binary_classification": "binary classification",
    "multiclass_classification": "multi-class classification",
    "multilabel_classification": "multi-label classification",
    "regression": "regression",
    "segmentation": "segmentation",
}


def run(ctx: ToolContext) -> list[Finding]:
    task = ctx.profile.task
    exp = EXPECTATIONS.get(task)
    if not exp:
        return []
    f = ctx.facts
    label = TASK_LABEL[task]
    findings: list[Finding] = []

    loss = f.loss
    if loss and loss.type not in exp["losses"]:
        findings.append(make_finding(
            "EZ003", f"{loss.type} does not fit {label}",
            f"The goal is {label}, but the loss is {loss.type}. {label.capitalize()} normally uses {exp['preferred_loss']}.",
            loss.line,
            [Evidence(tool=TOOL, summary=f"Goal task {task}; loss {loss.type} at line {loss.line}", lines=[loss.line],
                      data={"task": task, "loss": loss.type, "expected": sorted(exp["losses"])})],
        ))

    act = f.final_activation
    double_covered = act and loss and (act.kind, loss.type) in {
        ("softmax", "CrossEntropyLoss"), ("log_softmax", "CrossEntropyLoss"), ("sigmoid", "BCEWithLogitsLoss")}
    if act and not act.in_no_grad and act.kind in exp["bad_activations"] and not double_covered:
        findings.append(make_finding(
            "EZ003", f"Output {act.kind} does not fit {label}",
            f"forward() ends with {act.source}. For {label} the model should output raw values without {act.kind}.",
            act.line,
            [Evidence(tool=TOOL, summary=f"Goal task {task}; final activation {act.kind} at line {act.line}", lines=[act.line],
                      data={"task": task, "activation": act.kind})],
        ))

    if task in {"binary_classification"} and loss and loss.type in {"BCEWithLogitsLoss", "BCELoss"} \
            and f.head_out is not None and f.head_out != 1 and f.head is not None:
        findings.append(make_finding(
            "EZ003", f"{loss.type} with {f.head_out} outputs",
            f"{loss.type} for binary classification expects one output per sample, but {f.head.name} outputs {f.head_out}.",
            f.head.line,
            [Evidence(tool=TOOL, summary=f"{f.head.name} outputs {f.head_out}; {loss.type} expects 1", lines=[f.head.line, loss.line])],
            impact="crash",
        ))
    if task == "binary_classification" and loss and loss.type == "CrossEntropyLoss" \
            and f.head_out is not None and f.head_out > 2 and f.head is not None:
        findings.append(make_finding(
            "EZ003", f"Binary task with a {f.head_out}-class head",
            f"The goal is binary classification, but {f.head.name} outputs {f.head_out} classes.",
            f.head.line,
            [Evidence(tool=TOOL, summary=f"{f.head.name} outputs {f.head_out} for a 2-class task", lines=[f.head.line])],
        ))
    return findings
