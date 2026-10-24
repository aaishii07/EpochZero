"""EZ101-EZ103: two real training steps on CPU with a small synthetic batch.

Catches what a forward pass alone cannot: model outputs that do not fit the loss and targets, errors in backward,
non-finite losses, and layers that never receive gradients. Targets are synthetic but shaped the way a dataset for
the task would deliver them, and that assumption is stated in every finding.
"""

from __future__ import annotations

from app.sandbox.runner import run_job
from app.schemas import Evidence, SkippedCheck
from app.tools.base import ToolContext, ToolResult, make_finding
from app.tools.dynamic.shape_probe import ASSUMED, resolve_input_shape

TOOL = "dry_run"
MAX_PARAMS = 150_000_000  # larger models would make a CPU dry run slow; the meta-device shape probe still covers them

TARGET_TEXT = {
    "class": "class indices (one per sample, or one per input pixel for dense outputs)",
    "binary": "0/1 labels shaped like the model output",
    "like_input": "images shaped like the input",
    "like_output": "values shaped like the model output",
}


def _target_kind(ctx: ToolContext) -> str | None:
    loss = ctx.facts.loss
    if not loss:
        return None
    if loss.type in {"CrossEntropyLoss", "NLLLoss"}:
        return "class"
    if loss.type in {"BCEWithLogitsLoss", "BCELoss"}:
        return "binary"
    if loss.type in {"MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss"}:
        return "like_input" if ctx.profile.task == "image_to_image" else "like_output"
    return None


def _loss_sources(ctx: ToolContext) -> list[str]:
    loss = ctx.facts.loss
    if not loss:
        return []
    out = [loss.source] if loss.source and not loss.functional else []
    out.append(f"torch.nn.{loss.type}()")  # fallback when the original arguments reference local variables
    return out


def _loss_line(ctx: ToolContext) -> int:
    for loop in ctx.facts.loops:
        if loop.kind == "train" and loop.loss_calls:
            return loop.loss_calls[0]["line"]
    return ctx.facts.loss.line if ctx.facts.loss else 1


def _backward_line(ctx: ToolContext) -> int:
    for loop in ctx.facts.loops:
        call = loop.first("backward")
        if call:
            return call.line
    return _loss_line(ctx)


def run(ctx: ToolContext) -> ToolResult:
    f = ctx.facts
    if not f.model_classes:
        return ToolResult(False, "No nn.Module class found", skipped=[SkippedCheck(check="dry run", reason="no nn.Module class found")])
    if ctx.measured.get("forward_failed"):
        # The shape probe already reported the layer that fails; a training step cannot get further.
        return ToolResult(False, "Skipped: the forward pass already fails", skipped=[
            SkippedCheck(check="dry run", reason="the forward pass fails, see the shape finding")])
    shape, dtype, origin = resolve_input_shape(ctx, ctx.measured.get("input_shape"))
    if not shape:
        return ToolResult(False, "Input shape unknown; dry run skipped",
                          skipped=[SkippedCheck(check="dry run", reason="input shape unknown")])

    model_class = f.instantiations[0].cls if f.instantiations else f.model_classes[-1].name
    kind = _target_kind(ctx)
    job = {"kind": "dry_run", "input_shape": shape, "dtype": dtype, "model_class": model_class,
           "inst_sources": [i.source for i in f.instantiations if i.cls == model_class],
           "loss_sources": _loss_sources(ctx), "target_kind": kind, "max_params": MAX_PARAMS}
    res = run_job(ctx.code, job, timeout=30)
    data = {"input_shape": shape, "shape_origin": origin, "target_kind": kind}
    assumed = origin.startswith(ASSUMED)
    targets = TARGET_TEXT.get(kind or "", "no targets (the loss was not recognised, so the output mean was used)")
    setup = f"Two training steps on CPU with a random batch of shape {tuple(shape)} ({origin}) and synthetic targets: {targets}."

    stage, error = res.get("stage"), res.get("error", "")
    if not res.get("ok") and stage in {"construct", "too_large", "load", "timeout", "crash", "harness", "extract"}:
        reason = error if "EZ_EXTERNAL:" not in error else "the model loads pretrained weights while it is built"
        return ToolResult(False, f"Dry run could not run: {reason[:160]}", data={**data, "error": error},
                          skipped=[SkippedCheck(check="dry run", reason=reason[:200])])

    findings = []
    first_line = error.splitlines()[0][:240] if error else ""
    if stage == "loss":
        out, tgt = res.get("output"), res.get("target")
        detail = f"The model outputs {tuple(out) if out else '?'}" + (f" but the targets are {tuple(tgt)}" if tgt else "")
        findings.append(make_finding(
            "EZ101", "Loss cannot be computed from the model output",
            f"{detail}, and the loss fails: {first_line}. {setup}",
            _loss_line(ctx),
            [Evidence(tool=TOOL, summary=f"Loss raised {res.get('error_type')}: output {out}, targets {tgt}",
                      lines=[_loss_line(ctx)], data={**data, "output": out, "target": tgt, "error": error[:400]})],
            category="loss", **({"tier": "probed"} if assumed else {}),
        ))
    elif stage == "output":
        findings.append(make_finding(
            "EZ101", "Model output is not a tensor the loss can use", f"{error}. {setup}", _loss_line(ctx),
            [Evidence(tool=TOOL, summary=error, lines=[_loss_line(ctx)], data=data)], category="model",
        ))
    elif stage == "backward":
        line = _backward_line(ctx)
        findings.append(make_finding(
            "EZ101", "backward() fails", f"Gradients cannot be computed: {first_line}. {setup}", line,
            [Evidence(tool=TOOL, summary=f"backward raised {res.get('error_type')}", lines=[line], data={**data, "error": error[:400]})],
            category="training_loop",
        ))
    elif stage == "forward":
        line = f.model_classes[0].forward_line or f.model_classes[0].line
        findings.append(make_finding(
            "EZ101", "Forward pass fails on a real batch", f"{first_line}. {setup}", line,
            [Evidence(tool=TOOL, summary=f"forward raised {res.get('error_type')}", lines=[line], data={**data, "error": error[:400]})],
            category="model",
        ))
    elif res.get("ok"):
        losses = res.get("losses", [])
        if any(isinstance(v, str) for v in losses):
            findings.append(make_finding(
                "EZ102", "Loss is not a finite number", f"The loss was {losses} on the first steps. {setup}",
                _loss_line(ctx), [Evidence(tool=TOOL, summary=f"losses {losses}", lines=[_loss_line(ctx)], data=data)],
                category="loss",
            ))
        no_grad = res.get("no_grad", [])
        if no_grad and res.get("n_no_grad", 0) < res.get("n_trainable", 0):
            layer = no_grad[0].rsplit(".", 1)[0]
            line = next((l for name, l in f.layer_lines.items() if layer.endswith(name)), f.model_classes[0].line)
            names = ", ".join(n.rsplit(".", 1)[0] for n in no_grad[:5])
            findings.append(make_finding(
                "EZ103", "Some layers never receive gradients",
                f"After backward(), {res.get('n_no_grad')} trainable parameters still have no gradient ({names}). "
                f"These layers are defined but not used in the forward pass, or their output is detached. {setup}",
                line, [Evidence(tool=TOOL, summary=f"No gradient: {names}", lines=[line], data={**data, "no_grad": no_grad})],
                category="model",
            ))

    if findings:
        return ToolResult(True, f"Dry run found {len(findings)} problem{'s' if len(findings) != 1 else ''}", findings=findings, data=data)
    note = f" (loss built as {res.get('loss_note')})" if res.get("loss_note") else ""
    return ToolResult(True, f"Two training steps ran; losses {res.get('losses')}{note}", data={**data, "losses": res.get("losses")})
