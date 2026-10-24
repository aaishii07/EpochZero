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
