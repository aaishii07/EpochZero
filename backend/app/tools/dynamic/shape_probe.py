"""EZ001: run the model's forward pass on meta tensors and report the first layer that rejects its input."""

from __future__ import annotations

import re

from app.analysis.facts import CodeFacts
from app.sandbox.runner import run_job
from app.schemas import Evidence, SkippedCheck
from app.tools.base import ToolContext, ToolResult, make_finding

TOOL = "shape_probe"
MATMUL_RE = re.compile(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)")
CHANNELS_RE = re.compile(r"expected input\[([\d, ]+)\] to have (\d+) channels, but got (\d+) channels")
DATA_DEPENDENT_HINTS = ("meta", "data-dependent", "item()", "Boolean value of Tensor", "bool value of Tensor")


def resolve_input_shape(ctx: ToolContext, override: list[int] | None = None) -> tuple[list[int] | None, str, str]:
    """Returns (shape, dtype, how it was obtained)."""
    for shape, origin in ((override, "given by the planner"), (ctx.input_shape, "provided by you"),
                          (_parse_shape(ctx.answers.get("input_shape")), "from your answer")):
        if shape:
            return _batch2(shape), _dtype(ctx.facts), origin
    return _infer(ctx.facts)


def _batch2(shape: list[int]) -> list[int]:
    # BatchNorm in training mode needs more than one sample, so a batch of 1 becomes 2.
    return [2] + list(shape[1:]) if shape and shape[0] == 1 else list(shape)


def _dtype(facts: CodeFacts) -> str:
    first = facts.data_hints.get("first_layer") or {}
    return "long" if first.get("type") == "Embedding" else "float"


def _parse_shape(text) -> list[int] | None:
    if not text:
        return None
    try:
        dims = [int(p) for p in re.split(r"[,\sx()\[\]]+", str(text)) if p]
        return dims if dims and all(d > 0 for d in dims) else None
    except ValueError:
        return None


def _infer(facts: CodeFacts) -> tuple[list[int] | None, str, str]:
    first = facts.data_hints.get("first_layer") or {}
    ltype, size_in = first.get("type"), first.get("in")
    image = facts.data_hints.get("image_size")
    if not isinstance(size_in, int):
        return None, "float", "unknown"
    if ltype in {"Conv2d", "LazyConv2d"}:
        if image:
            return [2, size_in, image[0], image[1]], "float", f"inferred from {ltype}({size_in}, ...) and image size {image[0]}x{image[1]}"
        return [2, size_in, 224, 224], "float", f"inferred from {ltype}({size_in}, ...), assuming 224x224 images"
    if ltype == "Conv1d":
        return [2, size_in, 128], "float", f"inferred from Conv1d({size_in}, ...), assuming length 128"
    if ltype == "Linear":
        return [2, size_in], "float", f"inferred from Linear({size_in}, ...)"
    if ltype == "Embedding":
        return [2, 16], "long", "inferred from Embedding, assuming sequences of 16 tokens"
    if ltype in {"LSTM", "GRU", "RNN"}:
        return [2, 16, size_in], "float", f"inferred from {ltype}({size_in}, ...), assuming batch_first sequences of 16"
    return None, "float", "unknown"


def _line_for(facts: CodeFacts, module_name: str | None) -> int | None:
    if not module_name:
        return None
    parts = module_name.split(".")
    for i in range(len(parts)):
        name = ".".join(parts[i:])
        if name in facts.layer_lines:
            return facts.layer_lines[name]
    return facts.layer_lines.get(parts[-1])


def _declared_in(facts: CodeFacts, module_name: str | None) -> tuple[str, int] | None:
    """The input size a layer was declared with, from the source: ("features", n) or ("channels", n)."""
    line = _line_for(facts, module_name)
    layer = next((l for l in facts.layers if l.line == line and l.type != "Sequential"), None)
    if not layer:
        return None
    if layer.type == "Linear":
        n = layer.kwargs.get("in_features", layer.args[0] if layer.args else None)
        return ("features", n) if isinstance(n, int) else None
    if layer.type.startswith("Conv") or layer.type.startswith("BatchNorm"):
        key = "num_features" if layer.type.startswith("BatchNorm") else "in_channels"
        n = layer.kwargs.get(key, layer.args[0] if layer.args else None)
        return ("channels", n) if isinstance(n, int) else None
    return None


def _mismatch(facts: CodeFacts, name: str | None, inp, error: str) -> tuple[int | None, int | None, str]:
    """Compare the measured input with the size the layer was declared with; fall back to parsing the error."""
    declared = _declared_in(facts, name)
    got_shape = inp[0] if inp and isinstance(inp, list) and isinstance(inp[0], list) else None
    if declared and got_shape:
        kind, expected = declared
        got = got_shape[-1] if kind == "features" else (got_shape[1] if len(got_shape) > 1 else None)
        if got is not None and got != expected:
            return expected, got, kind
    if m := MATMUL_RE.search(error):
        return int(m.group(3)), int(m.group(2)), "features"
    if c := CHANNELS_RE.search(error):
        return int(c.group(2)), int(c.group(3)), "channels"
    return None, None, ""


def run(ctx: ToolContext, input_shape: list[int] | None = None) -> ToolResult:
    f = ctx.facts
    if not f.model_classes:
        return ToolResult(False, "No nn.Module class found", skipped=[SkippedCheck(check="shape probe", reason="no nn.Module class found")])
    shape, dtype, origin = resolve_input_shape(ctx, input_shape)
    if not shape:
        return ToolResult(False, "Input shape unknown and could not be inferred", needs="input_shape",
                          question="What input shape does the model take? Give one example batch, for example 1,3,224,224.",
                          skipped=[SkippedCheck(check="shape probe", reason="input shape unknown")])

    model_class = (f.instantiations[0].cls if f.instantiations else f.model_classes[-1].name)
    job = {"kind": "shape_probe", "input_shape": shape, "dtype": dtype, "model_class": model_class,
           "inst_sources": [i.source for i in f.instantiations if i.cls == model_class]}
    res = run_job(ctx.code, job)
    data = {"input_shape": shape, "shape_origin": origin, "model_class": model_class}

    if res.get("ok"):
        out = res.get("output")
        ctx.measured["output_shape"] = out if isinstance(out, list) and all(isinstance(d, int) for d in out) else None
        ctx.measured["records"] = res.get("records", [])
        return ToolResult(True, f"Forward pass succeeded on input {tuple(shape)} ({origin}); output {out}",
                          data={**data, "output_shape": out, "records": res.get("records", [])})

    stage, error = res.get("stage"), res.get("error", "")
    if stage != "forward":
        return ToolResult(False, f"Shape probe could not run: {error}", data={**data, "error": error, "stage": stage},
                          skipped=[SkippedCheck(check="shape probe", reason=f"model could not be built ({error[:160]})")])

    if any(h in error for h in DATA_DEPENDENT_HINTS) and not MATMUL_RE.search(error) and not CHANNELS_RE.search(error):
        failing = (res.get("failing") or {}).get("name")
        line = _line_for(f, failing)
        where = f" at line {line}" if line else ""
        return ToolResult(False, "Forward pass depends on tensor values; shape check stopped",
                          data={**data, "error": error},
                          skipped=[SkippedCheck(check="shape probe", reason=f"forward pass branches on tensor values{where}")])

    failing = res.get("failing") or {}
    name, ltype, inp = failing.get("name"), failing.get("type"), failing.get("in")
    line = _line_for(f, name) or (f.model_classes[0].forward_line or f.model_classes[0].line)
    expected, got, kind = _mismatch(f, name, inp, error)
    if expected is not None:
        title = f"{name} expects {expected} input {kind} but receives {got}"
        detail = {"expected": expected, "received": got}
    else:
        title = f"Forward pass fails at {name or 'the model'}"
        detail = {}
    in_text = f" with input shape {tuple(inp[0])}" if inp and isinstance(inp, list) and inp and isinstance(inp[0], list) else ""
    finding = make_finding(
        "EZ001", title,
        f"Running the forward pass on an input of shape {tuple(shape)} ({origin}) fails at {name} ({ltype}){in_text}: {error.splitlines()[0][:240]}",
        line,
        [Evidence(tool=TOOL, summary=f"{ltype} {name} raised {res.get('error_type')} on input {tuple(shape)}",
                  lines=[line], data={**data, **detail, "failing_module": name, "failing_type": ltype,
                                      "failing_input": inp, "error": error[:400], "records": res.get("records", [])[-12:]})],
    )
    return ToolResult(True, f"Shape mismatch at {name}: {title}", findings=[finding], data={**data, "error": error})
