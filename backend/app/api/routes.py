from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from app import config
from app.agent import orchestrator
from app.knowledge.docs import rule_catalogue
from app.agent.verifier import verify_combination
from app.analysis.facts import build_facts
from app.llm import mock
from app.schemas import AnalyzeRequest, GoalProfile, Patch

router = APIRouter()


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    if len(req.code.encode("utf-8")) > config.MAX_CODE_BYTES:
        raise HTTPException(status_code=413, detail="Script is larger than 200 KB")

    def stream():
        for event in orchestrator.run(req):
            yield event.model_dump_json() + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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
    import torch

    return {"status": "ok", "llm": config.LLM_MODE, "provider": "groq", "model": config.MODEL, "torch": torch.__version__}


@router.get("/rules")
def rules():
    return rule_catalogue()


DEMO_DIR = Path(__file__).resolve().parents[3] / "demo"


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
