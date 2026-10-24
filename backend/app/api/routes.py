import threading
import time
from collections import defaultdict, deque
from importlib import metadata
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config
from app.agent import orchestrator
from app.agent.verifier import verify_combination
from app.analysis.facts import build_facts
from app.knowledge.docs import rule_catalogue
from app.llm import mock
from app.sandbox import runner
from app.schemas import AnalyzeRequest, GoalProfile, Patch

router = APIRouter()

# ----- Per-visitor rate limit (in memory; one server process) -----
_WINDOW_SECONDS = 600
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the visitor's address is the first entry of X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


def _check_rate(request: Request) -> None:
    limit = config.RATE_LIMIT_PER_10_MIN
    if limit <= 0:
        return
    now, ip = time.monotonic(), _client_ip(request)
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > _WINDOW_SECONDS:
            q.popleft()
        if len(q) >= limit:
            wait = int(_WINDOW_SECONDS - (now - q[0])) + 1
            raise HTTPException(status_code=429, detail=f"Too many checks from this address. Try again in {wait} seconds.")
        q.append(now)


# Streaming through a reverse proxy: ask it not to buffer, so events reach the browser as they happen.
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/analyze")
def analyze(req: AnalyzeRequest, request: Request):
    if len(req.code.encode("utf-8")) > config.MAX_CODE_BYTES:
        raise HTTPException(status_code=413, detail="Script is larger than 200 KB")
    _check_rate(request)

    def stream():
        for event in orchestrator.run(req):
            yield event.model_dump_json() + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers=STREAM_HEADERS)


class PatchItem(BaseModel):
    id: str
    patch: Patch


class VerifyRequest(BaseModel):
    code: str  # the analyzed code every patch refers to
    goal: str = ""
    input_shape: list[int] | None = None
    answers: dict[str, str] = {}
    patches: list[PatchItem]  # in the order to try them
    applied_ids: list[str] = []  # already applied by the developer; kept as the starting point


@router.post("/verify")
def verify(req: VerifyRequest):
    """Check fixes together: each is kept only if the combined script gets no worse (no LLM involved)."""
    if len(req.code.encode("utf-8")) > config.MAX_CODE_BYTES:
        raise HTTPException(status_code=413, detail="Script is larger than 200 KB")
    # Not rate limited: it uses no LLM and runs on every Apply click.
    code = req.code.replace("\r\n", "\n")
    try:
        facts = build_facts(code)
    except SyntaxError:
        facts = None
    if req.goal.strip():
        profile = mock.read_goal(req.goal)
    else:
        profile = mock.infer_goal(facts) if facts else GoalProfile()
    return verify_combination(code, profile, req.input_shape, req.answers,
                              [(p.id, p.patch) for p in req.patches], req.applied_ids)


@router.get("/health")
def health():
    # torch is only imported in the sandbox worker; reading its version here does not load it into this process.
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_version = None
    return {"status": "ok", "llm": config.LLM_MODE, "provider": "groq", "model": config.MODEL,
            "torch": torch_version, "sandbox": "ready" if runner.worker_ready() else "starting"}


@router.get("/rules")
def rules():
    return rule_catalogue()


# The demo scripts live at the repository root; a deployment that only ships backend/ can place them in backend/demo.
_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = next((d for d in (_ROOT / "demo", _ROOT / "backend" / "demo") if d.is_dir()), _ROOT / "demo")


@router.get("/examples")
def examples():
    """Demo scripts for the example picker. The first line of each file is '# Goal: ...'."""
    out = []
    for path in sorted(DEMO_DIR.glob("*.py")):
        code = path.read_text(encoding="utf-8")
        first, _, rest = code.partition("\n")
        goal = first[len("# Goal:"):].strip() if first.startswith("# Goal:") else ""
        title = path.stem.split("_", 1)[-1].replace("_", " ").capitalize()
        out.append({"id": path.stem, "title": title, "goal": goal, "code": rest if goal else code})
    return out
