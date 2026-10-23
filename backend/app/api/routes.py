from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app import config
from app.agent import orchestrator
from app.knowledge.docs import rule_catalogue
from app.schemas import AnalyzeRequest

router = APIRouter()


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    if len(req.code.encode("utf-8")) > config.MAX_CODE_BYTES:
        raise HTTPException(status_code=413, detail="Script is larger than 200 KB")

    def stream():
        for event in orchestrator.run(req):
            yield event.model_dump_json() + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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
