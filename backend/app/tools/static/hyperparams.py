"""EZ011: advisory hyperparameter heuristics. Warnings with reasoning, never patched."""

from __future__ import annotations

from app.schemas import Evidence, Finding
from app.tools.base import ToolContext, make_finding

TOOL = "hyperparameters"


def run(ctx: ToolContext) -> list[Finding]:
    f = ctx.facts
    findings: list[Finding] = []
    for opt in f.optimizers:
        if opt.lr is not None:
            if opt.type in {"Adam", "AdamW", "NAdam", "RAdam", "Adamax"} and opt.lr > 1e-2:
                findings.append(_advisory(
                    f"High learning rate for {opt.type}",
                    f"{opt.type} with lr={opt.lr:g} is above the usual range (1e-5 to 1e-3). Large adaptive steps often "
                    "cause loss spikes or NaN values early in training.", opt.line, {"lr": opt.lr}))
            elif opt.type == "SGD" and opt.lr > 1.0:
                findings.append(_advisory(
                    "High learning rate for SGD",
                    f"SGD with lr={opt.lr:g} is unusually large and likely to diverge.", opt.line, {"lr": opt.lr}))
            elif opt.lr < 1e-6:
                findings.append(_advisory(
                    "Very small learning rate",
                    f"lr={opt.lr:g} is small enough that weights will barely change in a typical run.", opt.line, {"lr": opt.lr}))
        if opt.weight_decay is not None and opt.weight_decay > 0.1:
            findings.append(_advisory(
                "High weight decay",
                f"weight_decay={opt.weight_decay:g} is large and can prevent the model from fitting.", opt.line,
                {"weight_decay": opt.weight_decay}))
    sizes = f.data_hints.get("batch_sizes", [])
    if f.data_hints.get("has_batchnorm") and sizes and min(sizes) < 2:
        findings.append(_advisory(
            "Batch size 1 with BatchNorm",
            "BatchNorm cannot compute batch statistics from a single sample and raises an error in training mode.",
            1, {"batch_size": min(sizes)}))
    return findings


def _advisory(title: str, message: str, line: int, data: dict) -> Finding:
    return make_finding("EZ011", title, message, line,
                        [Evidence(tool=TOOL, summary=message.split(".")[0], lines=[line], data=data)], patchable=False)
