"""Code appended to the extracted user code and executed inside the sandbox subprocess.

It reads the job from the __EZ_JOB__ global, runs it, and prints one JSON result between markers so that
anything the user code prints does not interfere.
"""

HARNESS = r'''

def __ez_shapes(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [__ez_shapes(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): __ez_shapes(v) for k, v in obj.items()}
    return None


def __ez_block_downloads():
    # Building the model must not download or read weights: it is slow, needs network or files the sandbox does not
    # have, and is not needed to measure shapes. Each blocked call raises a marker the backend turns into a clear reason.
    import torch

    def blocked(name):
        def raise_marker(*args, **kwargs):
            raise RuntimeError(f"EZ_EXTERNAL:{name}")
        return raise_marker

    torch.hub.load = blocked("torch.hub.load")
    torch.hub.load_state_dict_from_url = blocked("torch.hub.load_state_dict_from_url")
    torch.load = blocked("torch.load")
    import sys
    tf = sys.modules.get("transformers")
    if tf is not None:
        for cls_name in ("PreTrainedModel", "AutoModel", "AutoModelForImageClassification", "AutoModelForCausalLM",
                         "AutoModelForSequenceClassification", "AutoBackbone"):
            cls = getattr(tf, cls_name, None)
            if cls is not None:
                cls.from_pretrained = blocked(f"transformers.{cls_name}.from_pretrained")


def __ez_build_model(job):
    __ez_block_downloads()
    errors = []
    for src in job.get("inst_sources", []):
        try:
            return eval(src, globals()), None
        except Exception as e:
            errors.append(f"{src}: {type(e).__name__}: {e}")
    cls = globals().get(job.get("model_class") or "")
    if cls is None:
        return None, "model class not found"
    try:
        return cls(), None
    except Exception as e:
        errors.append(f"{cls.__name__}(): {type(e).__name__}: {e}")
    return None, " | ".join(errors)[:600]


def __ez_shape_probe(job):
    import torch
    model, err = __ez_build_model(job)
    if model is None:
        return {"ok": False, "stage": "construct", "error": err}
    try:
        model = model.to("meta")
    except Exception as e:
        return {"ok": False, "stage": "construct", "error": f"could not move model to meta device: {e}"}
    model.train()
    names = {m: n for n, m in model.named_modules()}
    records, stack = [], []

    def pre(m, inp):
        stack.append({"name": names.get(m, "?"), "type": type(m).__name__, "in": __ez_shapes(inp)})

    def post(m, inp, out):
        entry = stack.pop() if stack else {"name": names.get(m, "?"), "type": type(m).__name__, "in": __ez_shapes(inp)}
        entry["out"] = __ez_shapes(out)
        records.append(entry)

    for m, n in names.items():
        if n:
            m.register_forward_pre_hook(pre)
            m.register_forward_hook(post)
    dtype = torch.long if job.get("dtype") == "long" else torch.float32
    x = torch.empty(job["input_shape"], device="meta", dtype=dtype)
    try:
        out = model(x)
    except Exception as e:
        failing = stack[-1] if stack else None
        return {"ok": False, "stage": "forward", "error_type": type(e).__name__, "error": str(e)[:800],
                "failing": failing, "records": records[-40:]}
    return {"ok": True, "output": __ez_shapes(out), "records": records[-80:]}


def __ez_first_tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for o in out:
            t = __ez_first_tensor(o)
            if t is not None:
                return t
    if isinstance(out, dict):
        for key in ("out", "logits", "pred", "output"):
            if isinstance(out.get(key), torch.Tensor):
                return out[key]
        for v in out.values():
            t = __ez_first_tensor(v)
            if t is not None:
                return t
    if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
        return out.logits
    return None


def __ez_target(kind, logits, x):
    # Synthetic targets shaped the way a real dataset for this task would deliver them.
    import torch
    if kind == "class":
        classes = max(int(logits.shape[1]), 1) if logits.dim() >= 2 else 2
        if logits.dim() >= 3:  # dense prediction: one label per input pixel, at the input's spatial size
            return torch.randint(0, classes, (x.shape[0],) + tuple(x.shape[2:]))
        return torch.randint(0, classes, (x.shape[0],))
    if kind == "binary":
        return torch.randint(0, 2, tuple(logits.shape)).float()
    if kind == "like_input":
        return torch.rand_like(x.float())
    if kind == "like_output":
        return torch.rand_like(logits.float())
    return None


def __ez_dry_run(job):
    import math
    import torch
    torch.manual_seed(0)
    model, err = __ez_build_model(job)
    if model is None:
        return {"ok": False, "stage": "construct", "error": err}
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > job["max_params"]:
        return {"ok": False, "stage": "too_large", "error": f"{n_params:,} parameters is too large for a CPU dry run"}
    model.train()
    shape = job["input_shape"]
    x = torch.randint(0, 100, shape) if job.get("dtype") == "long" else torch.randn(shape)
    criterion, loss_note = None, None
    for src in job.get("loss_sources", []):
        try:
            criterion = eval(src, globals())
            break
        except Exception as e:
            loss_note = f"{src}: {type(e).__name__}: {e}"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params or [torch.zeros(1, requires_grad=True)], lr=1e-4)
    losses, out_shape, target_shape = [], None, None
    for step in range(2):
        opt.zero_grad()
        try:
            out = model(x)
        except Exception as e:
            return {"ok": False, "stage": "forward", "error_type": type(e).__name__, "error": str(e)[:800]}
        logits = __ez_first_tensor(out)
        if logits is None:
            return {"ok": False, "stage": "output", "error": f"forward returned {type(out).__name__}, not a tensor"}
        out_shape = list(logits.shape)
        target = __ez_target(job.get("target_kind"), logits, x) if criterion is not None else None
        target_shape = list(target.shape) if target is not None else None
        try:
            loss = criterion(logits, target) if criterion is not None and target is not None else logits.float().mean()
        except Exception as e:
            return {"ok": False, "stage": "loss", "error_type": type(e).__name__, "error": str(e)[:800],
                    "output": out_shape, "target": target_shape,
                    "target_dtype": str(target.dtype).replace("torch.", "") if target is not None else None}
        if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
            return {"ok": False, "stage": "loss", "error_type": "ValueError",
                    "error": "the loss is not a single number; it needs a reduction such as mean",
                    "output": out_shape, "target": target_shape}
        try:
            loss.backward()
        except Exception as e:
            return {"ok": False, "stage": "backward", "error_type": type(e).__name__, "error": str(e)[:800]}
        opt.step()
        value = float(loss.detach())
        losses.append(value if math.isfinite(value) else str(value))
    names = {p: n for n, p in model.named_parameters()}
    no_grad = [names[p] for p in params if p.grad is None]
    return {"ok": True, "losses": losses, "output": out_shape, "target": target_shape, "n_params": n_params,
            "no_grad": no_grad[:20], "n_no_grad": len(no_grad), "n_trainable": len(params), "loss_note": loss_note}


def __ez_run():
    import json
    job = __EZ_JOB__
    try:
        if job["kind"] == "shape_probe":
            result = __ez_shape_probe(job)
        elif job["kind"] == "dry_run":
            result = __ez_dry_run(job)
        else:
            result = {"ok": False, "stage": "harness", "error": "unknown job"}
    except BaseException as e:
        result = {"ok": False, "stage": "harness", "error": f"{type(e).__name__}: {e}"[:800]}
    print("__EZ_RESULT_BEGIN__" + json.dumps(result) + "__EZ_RESULT_END__")


__ez_run()
'''
