"""Full agent loop on the demo scripts in offline mode. Slower: each runs the sandbox."""

import os
from pathlib import Path

os.environ["EPOCHZERO_LLM"] = "mock"

from app import config  # noqa: E402

config.LLM_MODE = "mock"

from app.agent import orchestrator  # noqa: E402
from app.schemas import AnalyzeRequest  # noqa: E402

DEMO = Path(__file__).resolve().parents[2] / "demo"


def run(name, goal, answers=None):
    events = list(orchestrator.run(AnalyzeRequest(code=(DEMO / name).read_text(encoding="utf-8"), goal=goal,
                                                  answers=answers or {})))
    return events, [e.suggestion for e in events if e.type == "suggestion"]


def test_shape_mismatch_resolved():
    _, s = run("01_shape_mismatch.py", "10-class image classification on CIFAR-10")
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ001", "resolved")]
    assert "2048" in s[0].patch.replacement


def test_segmentation_grounded_resolved():
    _, s = run("02_segmentation_classes_grounded.py", "5-class road scene segmentation")
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ004", "resolved")]


def test_segmentation_ambiguous_asks_then_resolves():
    events, s = run("02b_segmentation_classes_ambiguous.py", "road scene segmentation")
    assert events[-1].type == "needs_input" and events[-1].question.id == "num_classes" and not s
    _, s = run("02b_segmentation_classes_ambiguous.py", "road scene segmentation", {"num_classes": "7"})
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ004", "resolved")]


def test_val_leak_and_advisory_lr():
    _, s = run("03_val_leak_and_lr.py", "10-class image classification of product photos")
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ005", "resolved"), ("EZ011", "not_attempted")]


FLAWED = """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from mnist_data import loader

class FlawedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)

train_set = datasets.MNIST("data", train=True, download=True)
model = FlawedNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for x, y in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
"""


def test_user_given_shape_is_used_by_verifier():
    events = list(orchestrator.run(AnalyzeRequest(code=FLAWED, goal="10-class digit classification", input_shape=[16, 1, 28, 28])))
    s = [e.suggestion for e in events if e.type == "suggestion"]
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ001", "resolved")]
    assert "3136" in s[0].patch.replacement


def test_goal_inferred_from_code_when_missing():
    events = list(orchestrator.run(AnalyzeRequest(code=FLAWED)))
    goal = next(e for e in events if e.type == "goal")
    assert goal.inferred and goal.profile.task == "multiclass_classification" and goal.profile.num_classes == 10
    s = [e.suggestion for e in events if e.type == "suggestion"]
    # MNIST in the code gives 28x28 inputs, so the probe measures the real 3136 without a user-given shape.
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ001", "resolved")]


def test_inferred_goal_does_not_settle_ambiguous_class_count():
    events, s = run("02b_segmentation_classes_ambiguous.py", "")
    assert events[-1].type == "needs_input" and events[-1].question.id == "num_classes"


def test_agreeing_single_source_is_not_ambiguous():
    _, s = run("03_val_leak_and_lr.py", "")
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ005", "resolved"), ("EZ011", "not_attempted")]


def test_dont_know_class_count_does_not_loop():
    events, s = run("02b_segmentation_classes_ambiguous.py", "road scene segmentation", {"num_classes": "Don't know"})
    assert events[-1].type == "done"
    assert [(x.finding.rule_id, x.verify_status) for x in s] == [("EZ004", "not_attempted")]


def test_single_channel_conv_assumes_28x28_and_marks_probed():
    code = FLAWED.replace('train_set = datasets.MNIST("data", train=True, download=True)', "")
    events = list(orchestrator.run(AnalyzeRequest(code=code, goal="", answers={"input_shape": "dont know"})))
    s = [e.suggestion for e in events if e.type == "suggestion"]
    assert [(x.finding.rule_id, x.finding.tier, x.verify_status) for x in s] == [("EZ001", "probed", "resolved")]
    assert "3136" in s[0].patch.replacement


def test_all_syntax_errors_are_listed():
    from app.agent.orchestrator import collect_syntax_errors
    code = "import torch\ndef f(:\n    return 1\nx = (1,\ny = 2\nprint 'hi'\n"
    lines = [line for line, _ in collect_syntax_errors(code)]
    assert len(lines) >= 2 and lines[0] == 2


def test_syntax_error_is_a_finding_not_a_crash():
    events = list(orchestrator.run(AnalyzeRequest(code="import torch\nmodel = torch.nn.Linear(3, 2\nprint(model)\n")))
    findings = [e.finding for e in events if e.type == "finding"]
    assert findings and findings[0].rule_id == "EZ000"
    assert events[-1].type == "done"


def test_pretrained_download_skips_shape_probe_with_reason():
    code = '''
import torch
import torch.nn as nn

class Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16")
        self.decoder = nn.Conv2d(768, 5, 3)

    def forward(self, x):
        return self.decoder(self.backbone(x))

model = Seg()
'''
    events = list(orchestrator.run(AnalyzeRequest(code=code, goal="5-class segmentation", input_shape=[1, 3, 224, 224])))
    cov = next(e.coverage for e in events if e.type == "coverage")
    reason = next(s.reason for s in cov.skipped if s.check == "shape probe")
    assert "torch.hub.load" in reason


def test_review_findings_flow_through_repair(monkeypatch):
    from app.agent import reviewer
    from app.schemas import Evidence, Patch
    from app.tools.base import make_finding

    target = FLAWED.splitlines().index("model = FlawedNet()") + 1

    def fake_review(code, goal, existing, *_):
        f = make_finding("EZ100", "Weights never loaded", "The checkpoint is created but never loaded.", target,
                         [Evidence(tool="llm_review", summary="reviewer", lines=[target])], impact="silent")
        return [(f, Patch(start_line=target, end_line=target, replacement="model = FlawedNet()  # load weights here"))]

    monkeypatch.setattr(config, "LLM_MODE", "live")
    monkeypatch.setattr(reviewer, "review", fake_review)
    from app.agent import agent, fixer, goal_reader
    from app.llm import mock
    monkeypatch.setattr(goal_reader, "structured", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(agent, "tool_turn", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(fixer, "structured", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    events = list(orchestrator.run(AnalyzeRequest(code=FLAWED, goal="10-class digit classification")))
    s = {x.finding.rule_id: x for x in (e.suggestion for e in events if e.type == "suggestion")}
    assert s["EZ100"].finding.tier == "advisory" and s["EZ100"].verify_status == "inconclusive"
    assert s["EZ001"].verify_status == "resolved"
