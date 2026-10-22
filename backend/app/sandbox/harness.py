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


def __ez_build_model(job):
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


def __ez_run():
    import json
    job = __EZ_JOB__
    try:
        if job["kind"] == "shape_probe":
            result = __ez_shape_probe(job)
        else:
            result = {"ok": False, "stage": "harness", "error": "unknown job"}
    except BaseException as e:
        result = {"ok": False, "stage": "harness", "error": f"{type(e).__name__}: {e}"[:800]}
    print("__EZ_RESULT_BEGIN__" + json.dumps(result) + "__EZ_RESULT_END__")


__ez_run()
'''
