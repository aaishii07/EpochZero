"""Log history sent with each check: what the reviewer is told, and which repeats are dropped."""

from app.agent import memory
from app.schemas import AnalyzeRequest, HistoryItem

CODE = "import os\nos.makedirs('checkpoints', exist_ok=True)\ntorch.save(model.state_dict(), 'checkpoints/m.pt')\n"

FIX = HistoryItem(title="Checkpoint directory may not exist", rule_id="EZ100", check=1, line=2,
                  before=[], after=["os.makedirs('checkpoints', exist_ok=True)"])


def test_fix_still_in_code_is_detected():
    assert memory.still_in_code(FIX, CODE)
    assert not memory.still_in_code(FIX, CODE.replace("os.makedirs('checkpoints', exist_ok=True)\n", ""))


def test_history_text_marks_state():
    assert "still in the code" in memory.history_text([FIX], CODE)
    assert "no longer in the code" in memory.history_text([FIX], "torch.save(model, 'm.pt')\n")


def test_repeat_of_a_fixed_problem_is_dropped():
    assert memory.is_rereport("Checkpoint directory may not exist before saving", "torch.save fails", [FIX], CODE)


def test_problem_caused_by_the_fix_is_kept():
    assert not memory.is_rereport("Checkpoint directory may not exist", "This was introduced by the earlier fix.", [FIX], CODE)


def test_repeat_is_kept_when_the_fix_was_undone():
    assert not memory.is_rereport("Checkpoint directory may not exist", "", [FIX], "torch.save(model, 'm.pt')\n")


def test_request_accepts_history():
    req = AnalyzeRequest(code=CODE, history=[FIX.model_dump()], previous_open=[{"rule_id": "EZ006", "title": "x", "line": 3}])
    assert req.history[0].title == FIX.title and req.previous_open[0].line == 3


def test_agent_keeps_recent_results_and_shortens_older_ones():
    from app.agent.agent import KEEP_FULL_RESULTS, _compact
    messages = [{"role": "user", "content": "start"}] + [
        {"role": "tool", "tool_call_id": str(i), "content": "x" * 2000} for i in range(6)]
    _compact(messages)
    tools = [m for m in messages if m["role"] == "tool"]
    assert all(m["content"].endswith("[shortened]") for m in tools[:-KEEP_FULL_RESULTS])
    assert all(len(m["content"]) == 2000 for m in tools[-KEEP_FULL_RESULTS:])


def test_offline_agent_reports_and_finishes():
    import os
    os.environ["EPOCHZERO_LLM"] = "mock"
    from app import config
    config.LLM_MODE = "mock"
    from pathlib import Path
    from app.agent import orchestrator
    code = (Path(__file__).resolve().parents[2] / "demo" / "01_shape_mismatch.py").read_text(encoding="utf-8")
    events = list(orchestrator.run(AnalyzeRequest(code=code, goal="10-class image classification")))
    kinds = [e.type for e in events]
    assert "agent" in kinds and "report" in kinds and kinds[-1] == "done"
    report = next(e for e in events if e.type == "report")
    assert report.mode == "offline" and "verified" in report.text
