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
