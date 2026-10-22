"""Run extracted user code plus the harness in a separate Python process with a timeout."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import SANDBOX_TIMEOUT_SECONDS
from app.sandbox.extract import extract
from app.sandbox.harness import HARNESS


# The verifier reruns the same probe on the original code; identical scripts reuse the earlier result.
_cache: dict[str, dict] = {}


def run_job(code: str, job: dict, timeout: int = SANDBOX_TIMEOUT_SECONDS) -> dict:
    try:
        script = extract(code) + f"\n__EZ_JOB__ = {json.dumps(job)!r}\nimport json as __ez_json\n__EZ_JOB__ = __ez_json.loads(__EZ_JOB__)\n" + HARNESS
    except SyntaxError as e:
        return {"ok": False, "stage": "extract", "error": f"SyntaxError: {e}"}
    key = hashlib.sha256(script.encode("utf-8")).hexdigest()
    if key in _cache:
        return _cache[key]
    result = _execute(script, timeout)
    if result.get("stage") != "timeout":
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = result
    return result


def _execute(script: str, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="epochzero_") as tmp:
        path = Path(tmp) / "job.py"
        path.write_text(script, encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, env=env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "timeout", "error": f"model check did not finish within {timeout} seconds"}
    out = proc.stdout or ""
    start, end = out.find("__EZ_RESULT_BEGIN__"), out.find("__EZ_RESULT_END__")
    if start == -1 or end == -1:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"ok": False, "stage": "load", "error": " ".join(tail)[:600] or "sandbox produced no result"}
    return json.loads(out[start + len("__EZ_RESULT_BEGIN__"):end])
