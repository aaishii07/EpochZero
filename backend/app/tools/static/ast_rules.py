"""Static rules over CodeFacts: EZ002, EZ005, EZ006, EZ007, EZ008, EZ009, EZ010."""

from __future__ import annotations

import re

from app.analysis.facts import EVAL_HINT_RE, LoopInfo
from app.schemas import Evidence, Finding
from app.tools.base import ToolContext, make_finding

TOOL = "static_rules"
RANDOM_TRANSFORMS = {
    "RandomHorizontalFlip", "RandomVerticalFlip", "RandomCrop", "RandomResizedCrop", "RandomRotation",
    "RandomAffine", "RandomPerspective", "RandomErasing", "RandomGrayscale", "RandomApply", "RandomChoice",
    "RandomInvert", "RandomPosterize", "RandomSolarize", "RandomAdjustSharpness", "RandomAutocontrast",
    "RandomEqualize", "ColorJitter", "AutoAugment", "RandAugment", "TrivialAugmentWide", "AugMix", "GaussianBlur",
}


def double_activation(ctx: ToolContext) -> list[Finding]:
    f = ctx.facts
    act, loss = f.final_activation, f.loss
    if not act or not loss or act.in_no_grad:
        return []
    pairs = {
        ("softmax", "CrossEntropyLoss"): "CrossEntropyLoss already applies log_softmax",
        ("log_softmax", "CrossEntropyLoss"): "CrossEntropyLoss already applies log_softmax",
        ("sigmoid", "BCEWithLogitsLoss"): "BCEWithLogitsLoss already applies sigmoid",
    }
    reason = pairs.get((act.kind, loss.type))
    if not reason:
        return []
    return [make_finding(
        "EZ002", f"{act.kind} applied before {loss.type}",
        f"forward() returns {act.source}, and {reason}. The activation is applied twice, which compresses gradients.",
        act.line,
        [Evidence(tool=TOOL, summary=f"Model output passes through {act.kind} at line {act.line}; loss is {loss.type} at line {loss.line}",
                  lines=[act.line, loss.line], data={"activation": act.kind, "loss": loss.type})],
    )]


def eval_augmentation_leak(ctx: ToolContext) -> list[Finding]:
    f = ctx.facts
    findings = []
    by_var = {t.var: t for t in f.transforms if t.var}
    seen: set[int] = set()

    def check(items, label, use_line, reason_extra=""):
        random_items = [i for i in items if i["type"] in RANDOM_TRANSFORMS]
        if not random_items:
            return
        first, last = min(i["line"] for i in random_items), max(i["line"] for i in random_items)
        if first in seen:
            return
        seen.add(first)
        names = ", ".join(i["type"] for i in random_items)
        findings.append(make_finding(
            "EZ005", f"Random augmentation in {label}",
            f"{label} applies {names}. Evaluation data should be transformed deterministically, otherwise validation "
            f"metrics change between runs and do not reflect real performance.{reason_extra}",
            first,
            [Evidence(tool=TOOL, summary=f"{names} used by {label}", lines=sorted({i['line'] for i in random_items} | {use_line}),
                      data={"transforms": [i["type"] for i in random_items]})],
            end_line=last,
        ))

    for use in f.transform_uses:
        if use.split != "eval":
            continue
        if use.inline_items:
            check(use.inline_items, f"the {use.dataset_var} transform", use.line)
            continue
        t = by_var.get(use.transform_var)
        if not t:
            continue
        shared = any(u.transform_var == t.var and u.split == "train" for u in f.transform_uses)
        extra = f" The same transform object {t.var} is also used for training." if shared else ""
        check(t.items, f"{t.var} (used by {use.dataset_var})", use.line, extra)

    used_vars = {u.transform_var for u in f.transform_uses}
    for t in f.transforms:
        if t.var and t.var not in used_vars and EVAL_HINT_RE.search(t.var):
            check(t.items, t.var, t.line)
    return findings


def _train_loops(ctx: ToolContext) -> list[LoopInfo]:
    return [l for l in ctx.facts.loops if l.kind == "train"]


def zero_grad_rules(ctx: ToolContext) -> list[Finding]:
    findings = []
    for loop in _train_loops(ctx):
        backward, zero = loop.first("backward"), loop.first("zero_grad")
        steps = [c for c in loop.calls if c.name == "step" and c.receiver and "sched" not in c.receiver.lower()]
        step = steps[0] if steps else None
        if not backward or not step:
            continue
        if zero is None:
            accumulating = step.under_modulo_if or loop.loss_scaled
            if accumulating:
                findings.append(make_finding(
                    "EZ006", "Gradient accumulation without zero_grad",
                    "optimizer.step() is guarded by a modulo condition or the loss is scaled, which looks like intentional "
                    "gradient accumulation, but zero_grad() is never called, so gradients keep growing across steps.",
                    backward.line,
                    [Evidence(tool=TOOL, summary="Accumulation pattern detected; no zero_grad() in the loop",
                              lines=[loop.line, backward.line, step.line])],
                    severity="info", impact="risk", patchable=False,
                ))
                continue
            findings.append(make_finding(
                "EZ006", "optimizer.zero_grad() is never called",
                f"The training loop at line {loop.line} calls {backward.source} and {step.source} but never resets "
                "gradients. Each step adds to the previous gradients, so updates grow and training diverges or stalls.",
                backward.line,
                [Evidence(tool=TOOL, summary=f"Loop at line {loop.line}: backward() at {backward.line}, step() at {step.line}, no zero_grad()",
                          lines=[loop.line, backward.line, step.line], data={"loop_line": loop.line})],
            ))
        elif backward.line < zero.line < step.line:
            findings.append(make_finding(
                "EZ007", "zero_grad() runs between backward() and step()",
                f"Gradients computed at line {backward.line} are erased at line {zero.line} before {step.source} uses "
                "them, so the optimizer steps with zero gradients.",
                zero.line,
                [Evidence(tool=TOOL, summary=f"Order is backward ({backward.line}), zero_grad ({zero.line}), step ({step.line})",
                          lines=[backward.line, zero.line, step.line])],
            ))
    return findings


def loss_accumulation_leak(ctx: ToolContext) -> list[Finding]:
    findings = []
    for loop in _train_loops(ctx):
        backward = loop.first("backward")
        loss_var = backward.receiver if backward else None
        if not loss_var:
            continue
        pattern = re.compile(rf"(?<![\w.]){re.escape(loss_var)}(?![\w(])")
        for acc in loop.accumulations:
            if acc.detached or acc.target == loss_var or not pattern.search(acc.value_src):
                continue
            findings.append(make_finding(
                "EZ008", f"{acc.target} accumulates the loss tensor",
                f"`{acc.source}` adds the loss tensor itself, keeping every step's autograd graph in memory. "
                f"Use {loss_var}.item() to accumulate a number.",
                acc.line,
                [Evidence(tool=TOOL, summary=f"Line {acc.line} adds {loss_var} without .item() or .detach()", lines=[acc.line])],
            ))
    return findings


def eval_mode_missing(ctx: ToolContext) -> list[Finding]:
    f = ctx.facts
    findings = []
    has_eval_call = bool(f.eval_calls)
    for loop in f.loops:
        if loop.kind != "eval":
            continue
        missing = []
        if not (loop.in_no_grad or loop.contains_no_grad):
            missing.append("torch.no_grad()")
        if not has_eval_call:
            missing.append("model.eval()")
        if not missing:
            continue
        what = " and ".join(missing)
        findings.append(make_finding(
            "EZ009", f"Validation loop runs without {what}",
            f"The loop over {loop.iter_source} at line {loop.line} evaluates the model without {what}. "
            "Dropout and BatchNorm stay in training mode and autograd records every operation.",
            loop.line,
            [Evidence(tool=TOOL, summary=f"Evaluation loop at line {loop.line} missing {what}", lines=[loop.line],
                      data={"missing": missing})],
        ))
    return findings


def ce_float_targets(ctx: ToolContext) -> list[Finding]:
    f = ctx.facts
    if not f.loss or f.loss.type != "CrossEntropyLoss" or "label_smoothing" in (f.loss.kwargs or {}):
        return []
    findings = []
    for loop in _train_loops(ctx):
        for call in loop.loss_calls:
            if len(call["args"]) < 2:
                continue
            target = call["args"][1]
            cast_line = None
            if target.endswith(".float()"):
                cast_line = call["line"]
            elif target in loop.float_casts:
                cast_line = loop.float_casts[target]
            if cast_line is None:
                continue
            findings.append(make_finding(
                "EZ010", "Float targets passed to CrossEntropyLoss",
                f"{target} is cast to float before `{call['source']}`. CrossEntropyLoss expects class indices of dtype "
                "long; float targets are treated as probabilities and fail when shapes differ.",
                cast_line,
                [Evidence(tool=TOOL, summary=f"Target cast to float at line {cast_line}, used by loss at line {call['line']}",
                          lines=sorted({cast_line, call["line"]}))],
            ))
    return findings


RULE_FUNCS = [double_activation, eval_augmentation_leak, zero_grad_rules, loss_accumulation_leak,
              eval_mode_missing, ce_float_targets]


def run(ctx: ToolContext) -> list[Finding]:
    out: list[Finding] = []
    for rule in RULE_FUNCS:
        try:
            out += rule(ctx)
        except Exception:
            # A rule must never take down the analysis; malformed input simply yields no finding.
            continue
    return out
