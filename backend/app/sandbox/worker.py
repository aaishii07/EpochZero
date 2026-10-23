"""Long-lived sandbox process. Started once by runner.py so torch is imported only once.

Protocol: one JSON object per line on stdin ({"script": "..."}), one JSON object per line on stdout
({"stdout": "...", "error": null}). Each script runs in a fresh namespace. If a script hangs or kills the
process, the runner restarts the worker.
"""

import contextlib
import io
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402

# Warm up the meta device so the first real job does not pay the one-time initialisation cost.
torch.nn.Linear(4, 2).to("meta")(torch.empty(2, 4, device="meta"))

out = sys.stdout
out.write(json.dumps({"ready": True}) + "\n")
out.flush()

for line in sys.stdin:
    try:
        job = json.loads(line)
    except json.JSONDecodeError:
        continue
    buffer = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(job["script"], "<epochzero-job>", "exec"), {"__name__": "__epochzero_job__"})
    except BaseException as e:  # user code may raise anything, including SystemExit
        error = f"{type(e).__name__}: {e}"[:600]
    out.write(json.dumps({"stdout": buffer.getvalue()[-20000:], "error": error}) + "\n")
    out.flush()
