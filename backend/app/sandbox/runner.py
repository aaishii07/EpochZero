"""Run extracted user code plus the harness in a separate, long-lived Python process with a timeout.

The worker (sandbox/worker.py) imports torch once, so each job costs milliseconds instead of a fresh
process start (about 6 seconds on Windows). A job that hangs or crashes the worker gets it restarted.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from app.config import SANDBOX_STARTUP_SECONDS, SANDBOX_TIMEOUT_SECONDS
from app.sandbox.extract import extract
from app.sandbox.harness import HARNESS

WORKER = Path(__file__).with_name("worker.py")
STARTUP_TIMEOUT = SANDBOX_STARTUP_SECONDS

# The verifier reruns the same probe on the original code; identical scripts reuse the earlier result.
_cache: dict[str, dict] = {}


class _Worker:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.ready = False  # torch imported and the worker answered
        self.lines: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.cwd = tempfile.mkdtemp(prefix="epochzero_")

    def _start(self):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(WORKER)], cwd=self.cwd, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()
        self.ready = False
        ready = self._read(STARTUP_TIMEOUT)
        if not ready or not ready.get("ready"):
            self._kill()
            raise RuntimeError("sandbox worker did not start")
        self.ready = True

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: queue.Queue):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _read(self, timeout: float) -> dict | None:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        return json.loads(line) if line else None

    def _kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def warm(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()

    def run(self, script: str, timeout: int) -> dict:
        with self.lock:
            for attempt in range(2):
                if self.proc is None or self.proc.poll() is not None:
                    self._start()
                try:
                    self.proc.stdin.write(json.dumps({"script": script}) + "\n")
                    self.proc.stdin.flush()
                except OSError:
                    self._kill()
                    continue
                reply = self._read(timeout)
                if reply is None:
                    died = self.proc.poll() is not None
                    self._kill()
                    if died and attempt == 0:
                        continue  # the worker crashed on this job; retry once in a fresh worker
                    stage = "crash" if died else "timeout"
                    return {"stage": stage, "error": "the model check crashed the sandbox" if died
                            else f"model check did not finish within {timeout} seconds"}
                return reply
            return {"stage": "crash", "error": "the model check crashed the sandbox"}


_worker = _Worker()


def worker_ready() -> bool:
    proc = _worker.proc
    return _worker.ready and proc is not None and proc.poll() is None


def warm_up():
    """Start the worker ahead of the first request."""
    try:
        _worker.warm()
    except Exception:
        pass


def run_job(code: str, job: dict, timeout: int = SANDBOX_TIMEOUT_SECONDS) -> dict:
    try:
        script = extract(code) + f"\n__EZ_JOB__ = {json.dumps(job)!r}\nimport json as __ez_json\n__EZ_JOB__ = __ez_json.loads(__EZ_JOB__)\n" + HARNESS
    except SyntaxError as e:
        return {"ok": False, "stage": "extract", "error": f"SyntaxError: {e}"}
    key = hashlib.sha256(script.encode("utf-8")).hexdigest()
    if key in _cache:
        return _cache[key]
    result = _execute(script, timeout)
    if result.get("stage") not in {"timeout", "crash"}:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = result
    return result


def _execute(script: str, timeout: int) -> dict:
    try:
        reply = _worker.run(script, timeout)
    except RuntimeError as e:
        return {"ok": False, "stage": "load", "error": str(e)}
    if "stage" in reply:
        return {"ok": False, **reply}
    out = reply.get("stdout") or ""
    start, end = out.find("__EZ_RESULT_BEGIN__"), out.find("__EZ_RESULT_END__")
    if start == -1 or end == -1:
        return {"ok": False, "stage": "load", "error": reply.get("error") or "sandbox produced no result"}
    return json.loads(out[start + len("__EZ_RESULT_BEGIN__"):end])
