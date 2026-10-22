"""Turn a PyTorch training script into CodeFacts using only the ast module.

Every rule, the sandbox extractor, and the planner read CodeFacts instead of walking
the syntax tree themselves. Nothing here executes user code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from typing import Any

LAYER_TYPES = {
    "Linear", "Bilinear", "LazyLinear",
    "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "LazyConv2d", "Embedding", "LSTM", "GRU", "RNN",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
    "Dropout", "Dropout2d", "ReLU", "LeakyReLU", "GELU", "SiLU", "Tanh", "Sigmoid",
    "Softmax", "LogSoftmax", "Flatten", "MaxPool1d", "MaxPool2d", "AvgPool2d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "Upsample", "Identity",
    "Sequential", "ModuleList", "TransformerEncoder", "TransformerEncoderLayer", "MultiheadAttention",
}
HEAD_TYPES = {"Linear", "LazyLinear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d"}
ACTIVATION_FUNCS = {"softmax": "softmax", "log_softmax": "log_softmax", "sigmoid": "sigmoid", "relu": "relu"}
ACTIVATION_LAYERS = {"Softmax": "softmax", "LogSoftmax": "log_softmax", "Sigmoid": "sigmoid", "ReLU": "relu"}
OPTIMIZERS = {"SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Adamax", "NAdam", "RAdam", "LBFGS"}
FUNCTIONAL_LOSSES = {
    "cross_entropy": "CrossEntropyLoss", "nll_loss": "NLLLoss", "mse_loss": "MSELoss",
    "l1_loss": "L1Loss", "binary_cross_entropy": "BCELoss",
    "binary_cross_entropy_with_logits": "BCEWithLogitsLoss", "smooth_l1_loss": "SmoothL1Loss",
    "huber_loss": "HuberLoss",
}
CLASS_CONST_RE = re.compile(r"^(num|n|nb)_?(classes|labels|class)$|^classes_?num$", re.IGNORECASE)
CLASS_LIST_RE = re.compile(r"^(class(es)?(_names)?|labels?(_names)?|categories|label_names|class_list)$", re.IGNORECASE)
EVAL_HINT_RE = re.compile(r"val|valid|test|eval|dev", re.IGNORECASE)


@dataclass
class Layer:
    name: str
    type: str
    args: list[Any]
    kwargs: dict[str, Any]
    line: int
    owner: str


@dataclass
class ModelClass:
    name: str
    line: int
    end_line: int
    init_params: dict[str, Any]
    forward_line: int | None = None
    forward_end_line: int | None = None


@dataclass
class Activation:
    kind: str
    line: int
    source: str
    in_no_grad: bool = False


@dataclass
class LossInfo:
    var: str | None
    type: str
    kwargs: dict[str, str]
    line: int
    functional: bool = False
    weight_len: int | None = None


@dataclass
class OptimizerInfo:
    var: str | None
    type: str
    lr: float | None
    weight_decay: float | None
    momentum: float | None
    line: int


@dataclass
class Call:
    name: str
    receiver: str | None
    line: int
    under_modulo_if: bool
    source: str


@dataclass
class Accumulation:
    line: int
    target: str
    detached: bool
    source: str
    value_src: str


@dataclass
class LoopInfo:
    line: int
    end_line: int
    iter_source: str
    function: str | None
    function_line: int
    calls: list[Call] = field(default_factory=list)
    in_no_grad: bool = False
    contains_no_grad: bool = False
    loss_scaled: bool = False
    accumulations: list[Accumulation] = field(default_factory=list)
    float_casts: dict[str, int] = field(default_factory=dict)
    loss_calls: list[dict] = field(default_factory=list)
    kind: str = "other"

    def first(self, name: str) -> Call | None:
        return next((c for c in self.calls if c.name == name), None)


@dataclass
class TransformInfo:
    var: str | None
    line: int
    end_line: int
    items: list[dict]


@dataclass
class TransformUse:
    transform_var: str | None
    dataset_var: str | None
    split: str
    line: int
    inline_items: list[dict] | None = None


@dataclass
class ClassCountSource:
    kind: str
    value: int
    line: int
    detail: str


@dataclass
class Instantiation:
    var: str
    cls: str
    source: str
    line: int
    kwargs: dict[str, Any]


@dataclass
class CodeFacts:
    lines: list[str]
    imports: list[str] = field(default_factory=list)
    constants: dict[str, Any] = field(default_factory=dict)
    model_classes: list[ModelClass] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    instantiations: list[Instantiation] = field(default_factory=list)
    final_activation: Activation | None = None
    head: Layer | None = None
    head_out: int | None = None
    losses: list[LossInfo] = field(default_factory=list)
    optimizers: list[OptimizerInfo] = field(default_factory=list)
    loops: list[LoopInfo] = field(default_factory=list)
    eval_calls: list[int] = field(default_factory=list)
    transforms: list[TransformInfo] = field(default_factory=list)
    transform_uses: list[TransformUse] = field(default_factory=list)
    class_count_sources: list[ClassCountSource] = field(default_factory=list)
    data_hints: dict[str, Any] = field(default_factory=dict)
    layer_lines: dict[str, int] = field(default_factory=dict)

    @property
    def loss(self) -> LossInfo | None:
        return self.losses[0] if self.losses else None

    @property
    def optimizer(self) -> OptimizerInfo | None:
        return self.optimizers[0] if self.optimizers else None

    def summary(self) -> dict:
        """Compact view for the planner prompt."""
        d = asdict(self)
        d.pop("lines")
        d["line_count"] = len(self.lines)
        for loop in d["loops"]:
            loop["calls"] = [f"{c['receiver'] + '.' if c['receiver'] else ''}{c['name']}@{c['line']}" for c in loop["calls"]]
        return d


def callee_name(node: ast.AST) -> str | None:
    """Last component of a call target: nn.Linear -> Linear, F.softmax -> softmax."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def receiver_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        try:
            return ast.unparse(call.func.value)
        except Exception:
            return None
    return None


def literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _is_no_grad_ctx(expr: ast.AST) -> bool:
    return callee_name(expr) in {"no_grad", "inference_mode"}


def _contains_mod(node: ast.AST) -> bool:
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) for n in ast.walk(node))


def _tensor_literal_len(node: ast.AST) -> int | None:
    """Length of torch.tensor([...]) / torch.ones(k) / torch.FloatTensor([...])."""
    if not isinstance(node, ast.Call):
        return None
    name = callee_name(node)
    if name in {"tensor", "Tensor", "FloatTensor", "as_tensor"} and node.args:
        val = literal(node.args[0])
        if isinstance(val, (list, tuple)):
            return len(val)
    if name in {"ones", "zeros", "full", "rand", "randn"} and node.args:
        val = literal(node.args[0])
        if isinstance(val, int):
            return val
        if isinstance(val, (list, tuple)) and len(val) == 1 and isinstance(val[0], int):
            return val[0]
    return None


class _Extractor(ast.NodeVisitor):
    def __init__(self, facts: CodeFacts):
        self.f = facts
        self.loop_stack: list[LoopInfo] = []
        self.func_stack: list[ast.FunctionDef] = []
        self.no_grad_depth = 0
        self.mod_if_depth = 0
        self.model_names: set[str] = set()

    # Imports and module constants

    def visit_Import(self, node: ast.Import):
        self.f.imports += [a.name for a in node.names]

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.f.imports += [f"{node.module}.{a.name}" for a in node.names]

    # Functions and no_grad scopes

    def visit_FunctionDef(self, node: ast.FunctionDef):
        decorated = any(_is_no_grad_ctx(d) or callee_name(d) in {"no_grad", "inference_mode"} for d in node.decorator_list)
        self.func_stack.append(node)
        self.no_grad_depth += decorated
        self.generic_visit(node)
        self.no_grad_depth -= decorated
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_With(self, node: ast.With):
        ng = any(_is_no_grad_ctx(item.context_expr) for item in node.items)
        if ng and self.loop_stack:
            self.loop_stack[-1].contains_no_grad = True
        self.no_grad_depth += ng
        self.generic_visit(node)
        self.no_grad_depth -= ng

    def visit_If(self, node: ast.If):
        mod = _contains_mod(node.test)
        self.visit(node.test)
        self.mod_if_depth += mod
        for stmt in node.body:
            self.visit(stmt)
        self.mod_if_depth -= mod
        for stmt in node.orelse:
            self.visit(stmt)

    # Loops

    def _visit_loop(self, node: ast.For | ast.While):
        fn = self.func_stack[-1] if self.func_stack else None
        it = ast.unparse(node.iter) if isinstance(node, (ast.For, ast.AsyncFor)) else ast.unparse(node.test)
        info = LoopInfo(
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            iter_source=it,
            function=fn.name if fn else None,
            function_line=fn.lineno if fn else 1,
            in_no_grad=self.no_grad_depth > 0,
        )
        self.f.loops.append(info)
        self.loop_stack.append(info)
        self.generic_visit(node)
        self.loop_stack.pop()

    visit_For = visit_While = visit_AsyncFor = _visit_loop

    # Assignments

    def visit_Assign(self, node: ast.Assign):
        target = node.targets[0]
        tname = ast.unparse(target)
        value = node.value

        if not self.func_stack and isinstance(target, ast.Name):
            val = literal(value)
            if isinstance(val, (int, float, str, list, tuple)):
                self.f.constants[target.id] = val

        if isinstance(target, ast.Name):
            self._record_class_sources(target.id, value, node.lineno)

        if isinstance(value, ast.Call):
            name = callee_name(value)
            if name and name.endswith("Loss"):
                self._record_loss(tname, value, node.lineno)
            elif name in OPTIMIZERS:
                self._record_optimizer(tname, value, node.lineno)
            elif name == "Compose":
                self.f.transforms.append(TransformInfo(tname, node.lineno, node.end_lineno or node.lineno, _compose_items(value)))
            elif name in self.model_names and isinstance(target, ast.Name):
                kwargs = {k.arg: literal(k.value) for k in value.keywords if k.arg}
                self.f.instantiations.append(Instantiation(target.id, name, ast.unparse(value), node.lineno, kwargs))
            elif name in {"to", "cuda", "cpu"} and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Call):
                inner = value.func.value
                iname = callee_name(inner)
                if iname in self.model_names and isinstance(target, ast.Name):
                    kwargs = {k.arg: literal(k.value) for k in inner.keywords if k.arg}
                    self.f.instantiations.append(Instantiation(target.id, iname, ast.unparse(inner), node.lineno, kwargs))
            self._record_transform_use(tname, value, node.lineno)

            if self.loop_stack and isinstance(target, ast.Name) and name == "float":
                self.loop_stack[-1].float_casts[target.id] = node.lineno

        if self.loop_stack:
            loop = self.loop_stack[-1]
            # loss = loss / k  (gradient accumulation scaling)
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div) and isinstance(target, ast.Name) \
                    and isinstance(value.left, ast.Name) and value.left.id == target.id:
                loop.loss_scaled = True
            # total = total + loss
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add) and isinstance(target, ast.Name) \
                    and isinstance(value.left, ast.Name) and value.left.id == target.id:
                loop.accumulations.append(Accumulation(node.lineno, target.id, _is_detached(value.right),
                                                       ast.unparse(node), ast.unparse(value.right)))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if self.loop_stack:
            loop = self.loop_stack[-1]
            tname = ast.unparse(node.target)
            if isinstance(node.op, ast.Div):
                loop.loss_scaled = True
            if isinstance(node.op, ast.Add):
                loop.accumulations.append(Accumulation(node.lineno, tname, _is_detached(node.value),
                                                       ast.unparse(node), ast.unparse(node.value)))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = callee_name(node)
        if name == "eval" and isinstance(node.func, ast.Attribute):
            self.f.eval_calls.append(node.lineno)
        if name in FUNCTIONAL_LOSSES and receiver_name(node) in {"F", "functional", "torch.nn.functional", "nn.functional", "torch"}:
            if not any(l.functional and l.type == FUNCTIONAL_LOSSES[name] for l in self.f.losses):
                self.f.losses.append(LossInfo(None, FUNCTIONAL_LOSSES[name], {}, node.lineno, functional=True))
        if name == "DataLoader":
            for kw in node.keywords:
                if kw.arg == "batch_size" and isinstance(literal(kw.value), int):
                    self.f.data_hints.setdefault("batch_sizes", []).append(literal(kw.value))
        for kw in node.keywords:
            if kw.arg in {"num_classes", "n_classes", "num_labels"}:
                # Only literal values are independent sources; a name just repeats a constant.
                val = literal(kw.value)
                if isinstance(val, int) and not isinstance(val, bool):
                    self.f.class_count_sources.append(ClassCountSource("keyword", val, node.lineno, f"{kw.arg}={ast.unparse(kw.value)} in {name}(...)"))
        if self.loop_stack:
            loop = self.loop_stack[-1]
            loop.calls.append(Call(name or "?", receiver_name(node), node.lineno, self.mod_if_depth > 0, ast.unparse(node)))
            loss_vars = {l.var for l in self.f.losses if l.var}
            if (isinstance(node.func, ast.Name) and node.func.id in loss_vars) or \
                    (name in FUNCTIONAL_LOSSES and receiver_name(node) in {"F", "functional", "torch.nn.functional"}):
                loop.loss_calls.append({"line": node.lineno, "args": [ast.unparse(a) for a in node.args], "source": ast.unparse(node)})
        self.generic_visit(node)

    # Recorders

    def _record_class_sources(self, name: str, value: ast.AST, line: int):
        if CLASS_CONST_RE.match(name):
            val = literal(value)
            if isinstance(val, int):
                self.f.class_count_sources.append(ClassCountSource("constant", val, line, f"{name} = {val}"))
        if CLASS_LIST_RE.match(name) and isinstance(value, (ast.List, ast.Tuple)):
            items = [literal(e) for e in value.elts]
            if items and all(isinstance(i, str) for i in items):
                self.f.class_count_sources.append(ClassCountSource("class_list", len(items), line, f"{name} has {len(items)} entries"))

    def _record_loss(self, var: str, call: ast.Call, line: int):
        kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
        info = LossInfo(var, callee_name(call), kwargs, line)
        for k in call.keywords:
            if k.arg in {"weight", "pos_weight"}:
                info.weight_len = _tensor_literal_len(k.value)
                if info.weight_len and k.arg == "weight":
                    self.f.class_count_sources.append(ClassCountSource("loss_weight", info.weight_len, line, f"{info.type} weight has {info.weight_len} values"))
        self.f.losses.append(info)

    def _record_optimizer(self, var: str, call: ast.Call, line: int):
        kw = {k.arg: literal(k.value) for k in call.keywords if k.arg}
        lr = kw.get("lr")
        if lr is None and len(call.args) > 1:
            lr = literal(call.args[1])
        num = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        self.f.optimizers.append(OptimizerInfo(var, callee_name(call), num(lr), num(kw.get("weight_decay")), num(kw.get("momentum")), line))

    def _record_transform_use(self, target: str, call: ast.Call, line: int):
        for kw in call.keywords:
            if kw.arg not in {"transform", "transforms"}:
                continue
            split = _split_of(target, call)
            if isinstance(kw.value, ast.Name):
                self.f.transform_uses.append(TransformUse(kw.value.id, target, split, line))
            elif isinstance(kw.value, ast.Call) and callee_name(kw.value) == "Compose":
                self.f.transform_uses.append(TransformUse(None, target, split, line, _compose_items(kw.value)))


def _is_detached(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and callee_name(n) in {"item", "detach", "tolist", "float"} and isinstance(n.func, ast.Attribute)
               for n in ast.walk(node)) or not isinstance(node, (ast.Name, ast.BinOp, ast.Call, ast.Attribute))


def _compose_items(call: ast.Call) -> list[dict]:
    items = []
    if call.args and isinstance(call.args[0], (ast.List, ast.Tuple)):
        for e in call.args[0].elts:
            items.append({"type": callee_name(e) or ast.unparse(e), "line": e.lineno, "source": ast.unparse(e),
                          "args": [literal(a) for a in e.args] if isinstance(e, ast.Call) else []})
    return items


def _split_of(target: str, call: ast.Call) -> str:
    for kw in call.keywords:
        val = literal(kw.value)
        if kw.arg == "train" and val is False:
            return "eval"
        if kw.arg == "train" and val is True:
            return "train"
        if kw.arg == "split" and isinstance(val, str):
            return "eval" if EVAL_HINT_RE.search(val) else "train"
    if EVAL_HINT_RE.search(target):
        return "eval"
    if "train" in target.lower():
        return "train"
    return "unknown"


def _model_classes(tree: ast.Module) -> list[ast.ClassDef]:
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for c in classes:
            if c.name in names:
                continue
            for b in c.bases:
                bsrc = ast.unparse(b)
                if bsrc.endswith("Module") or bsrc in names:
                    names.add(c.name)
                    changed = True
    return [c for c in classes if c.name in names]


def _layer_from_call(name: str, call: ast.Call, owner: str, out: list[Layer], resolve) -> None:
    ltype = callee_name(call) or "?"
    args = [resolve(a) for a in call.args]
    kwargs = {k.arg: resolve(k.value) for k in call.keywords if k.arg}
    out.append(Layer(name, ltype, args, kwargs, call.lineno, owner))
    if ltype == "Sequential":
        idx = 0
        for a in call.args:
            if isinstance(a, ast.Call):
                _layer_from_call(f"{name}.{idx}", a, owner, out, resolve)
            idx += 1


def _out_dim(layer: Layer) -> Any:
    keys = {"Linear": ("out_features", 1), "LazyLinear": ("out_features", 0)}
    key, pos = keys.get(layer.type, ("out_channels", 1))
    if key in layer.kwargs:
        return layer.kwargs[key]
    return layer.args[pos] if len(layer.args) > pos else None


def _analyze_model(cls: ast.ClassDef, facts: CodeFacts, resolve_factory):
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    fwd = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
    params: dict[str, Any] = {}
    if init:
        a = init.args
        defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
        for arg, d in zip(a.args, defaults):
            if arg.arg != "self":
                params[arg.arg] = literal(d) if d is not None else None
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            params[arg.arg] = literal(d) if d is not None else None
    mc = ModelClass(cls.name, cls.lineno, cls.end_lineno or cls.lineno, params,
                    fwd.lineno if fwd else None, (fwd.end_lineno if fwd else None))
    facts.model_classes.append(mc)
    resolve = resolve_factory(cls.name, params)

    own_layers: list[Layer] = []
    if init:
        for node in ast.walk(init):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                t = node.targets[0]
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    ltype = callee_name(node.value)
                    if ltype in LAYER_TYPES or (isinstance(node.value.func, ast.Attribute) and ast.unparse(node.value.func.value) in {"nn", "torch.nn"}):
                        _layer_from_call(t.attr, node.value, cls.name, own_layers, resolve)
    facts.layers += own_layers
    for l in own_layers:
        facts.layer_lines[l.name] = l.line

    if not fwd:
        return
    by_name = {l.name: l for l in own_layers}
    no_grad_lines = set()
    for node in ast.walk(fwd):
        if isinstance(node, ast.With) and any(_is_no_grad_ctx(i.context_expr) for i in node.items):
            no_grad_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    # Head: last layer called in forward that produces the output dimension.
    called = []
    for node in ast.walk(fwd):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "self" and node.func.attr in by_name:
            called.append((node.lineno, node.col_offset, node.func.attr))
    called.sort()
    head = None
    for _, _, attr in reversed(called):
        layer = by_name[attr]
        if layer.type in HEAD_TYPES:
            head = layer
            break
        if layer.type == "Sequential":
            subs = [l for l in own_layers if l.name.startswith(attr + ".") and l.type in HEAD_TYPES]
            if subs:
                head = subs[-1]
                break
    if head is None:
        heads = [l for l in own_layers if l.type in HEAD_TYPES]
        head = heads[-1] if heads else None
    if head and facts.head is None:
        facts.head = head
        out = _out_dim(head)
        facts.head_out = out if isinstance(out, int) else None

    # Final activation: what the returned value went through last.
    act = _final_activation(fwd, by_name, own_layers, no_grad_lines)
    if act and facts.final_activation is None:
        facts.final_activation = act


def _activation_of_expr(expr: ast.AST, by_name: dict, layers: list[Layer]) -> str | None:
    if not isinstance(expr, ast.Call):
        return None
    name = callee_name(expr)
    if name in ACTIVATION_FUNCS:
        return ACTIVATION_FUNCS[name]
    if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name) and expr.func.value.id == "self":
        layer = by_name.get(expr.func.attr)
        if layer and layer.type in ACTIVATION_LAYERS:
            return ACTIVATION_LAYERS[layer.type]
        if layer and layer.type == "Sequential":
            subs = [l for l in layers if l.name.startswith(layer.name + ".")]
            if subs and subs[-1].type in ACTIVATION_LAYERS:
                return ACTIVATION_LAYERS[subs[-1].type]
    return None


def _final_activation(fwd: ast.FunctionDef, by_name, layers, no_grad_lines) -> Activation | None:
    returns = [n for n in ast.walk(fwd) if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return None
    ret = returns[-1]
    kind = _activation_of_expr(ret.value, by_name, layers)
    if kind:
        return Activation(kind, ret.lineno, ast.unparse(ret.value), ret.lineno in no_grad_lines)
    if isinstance(ret.value, ast.Name):
        var = ret.value.id
        assigns = [n for n in ast.walk(fwd) if isinstance(n, ast.Assign) and n.lineno < ret.lineno
                   and any(isinstance(t, ast.Name) and t.id == var for t in n.targets)]
        if assigns:
            last = max(assigns, key=lambda n: n.lineno)
            kind = _activation_of_expr(last.value, by_name, layers)
            if kind:
                return Activation(kind, last.lineno, ast.unparse(last.value), last.lineno in no_grad_lines)
    return None


def _classify_loops(facts: CodeFacts):
    opt_vars = {o.var for o in facts.optimizers if o.var}
    for loop in facts.loops:
        names = {c.name for c in loop.calls}
        is_step = any(c.name == "step" and (c.receiver in opt_vars or (c.receiver and "opt" in c.receiver.lower())) for c in loop.calls)
        if "backward" in names and is_step:
            loop.kind = "train"
        elif "backward" not in names and (loop.in_no_grad or loop.contains_no_grad or EVAL_HINT_RE.search(loop.iter_source)
                                          or (loop.function and EVAL_HINT_RE.search(loop.function))):
            loop.kind = "eval"
    # A loop that only contains other loops is not itself train/eval: keep innermost classification.


def _data_hints(facts: CodeFacts):
    for t in facts.transforms + [TransformInfo(None, u.line, u.line, u.inline_items) for u in facts.transform_uses if u.inline_items]:
        for item in t.items:
            if item["type"] in {"Resize", "CenterCrop", "RandomResizedCrop", "RandomCrop"} and item["args"]:
                size = item["args"][0]
                if isinstance(size, int):
                    facts.data_hints.setdefault("image_size", (size, size))
                elif isinstance(size, (list, tuple)) and len(size) == 2 and all(isinstance(s, int) for s in size):
                    facts.data_hints.setdefault("image_size", tuple(size))
    if facts.model_classes:
        first_cls = facts.model_classes[0].name
        own = [l for l in facts.layers if l.owner == first_cls and l.type not in {"Sequential", "ModuleList"}]
        if own:
            first = own[0]
            facts.data_hints["first_layer"] = {"name": first.name, "type": first.type,
                                               "in": first.args[0] if first.args else first.kwargs.get("in_channels", first.kwargs.get("in_features"))}
    facts.data_hints["has_batchnorm"] = any(l.type.startswith("BatchNorm") for l in facts.layers)


def build_facts(code: str) -> CodeFacts:
    """Raises SyntaxError if the code does not parse."""
    tree = ast.parse(code)
    facts = CodeFacts(lines=code.splitlines())
    ex = _Extractor(facts)

    # Constants and model class names first, so later passes can resolve them.
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            val = literal(node.value)
            if isinstance(val, (int, float, str, list, tuple)):
                facts.constants[node.targets[0].id] = val
    classes = _model_classes(tree)
    ex.model_names = {c.name for c in classes}

    ex.visit(tree)

    def resolve_factory(cls_name: str, params: dict):
        def resolve(node: ast.AST) -> Any:
            val = literal(node)
            if isinstance(val, str):
                if val in facts.constants and isinstance(facts.constants[val], (int, float)):
                    return facts.constants[val]
                inst = next((i for i in facts.instantiations if i.cls == cls_name), None)
                if inst and val in inst.kwargs and isinstance(inst.kwargs[val], (int, float)):
                    return inst.kwargs[val]
                if inst and val in inst.kwargs and isinstance(inst.kwargs[val], str) and inst.kwargs[val] in facts.constants:
                    return facts.constants[inst.kwargs[val]]
                if params.get(val) is not None and isinstance(params[val], (int, float)):
                    return params[val]
            return val
        return resolve

    for cls in classes:
        _analyze_model(cls, facts, resolve_factory)

    if facts.head is not None and facts.head_out is not None:
        facts.class_count_sources.append(ClassCountSource(
            "head", facts.head_out, facts.head.line, f"{facts.head.name} = {facts.head.type}(...) outputs {facts.head_out}"))

    _classify_loops(facts)
    _data_hints(facts)
    return facts
