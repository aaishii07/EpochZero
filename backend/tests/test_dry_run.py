"""Dry run (two real training steps) and the combined-fix check."""

from app.agent.verifier import verify_combination
from app.analysis.facts import build_facts
from app.schemas import GoalProfile, Patch
from app.tools.base import ToolContext
from app.tools.dynamic import dry_run, shape_probe

SEG = '''
import torch
import torch.nn as nn

NUM_CLASSES = 5

class Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 16, 3, stride=4, padding=1)
        self.head = nn.Conv2d(16, NUM_CLASSES, 1)
        self.unused = nn.Linear(4, 4)

    def forward(self, x):
        return self.head(self.encoder(x))

model = Seg()
criterion = nn.CrossEntropyLoss()
'''


def run(code, task="segmentation", shape=(2, 3, 64, 64)):
    ctx = ToolContext(code=code, facts=build_facts(code), profile=GoalProfile(task=task), input_shape=list(shape))
    shape_probe.run(ctx)
    return dry_run.run(ctx)


def test_output_smaller_than_masks_is_caught_in_the_loss():
    result = run(SEG)
    rules = [f.rule_id for f in result.findings]
    assert rules == ["EZ101"], result.summary
    f = result.findings[0]
    assert f.category == "loss" and "(2, 5, 16, 16)" in f.message and "(2, 64, 64)" in f.message


def test_unused_layer_has_no_gradients():
    fixed = SEG.replace("stride=4, padding=1", "padding=1")
    result = run(fixed)
    assert [f.rule_id for f in result.findings] == ["EZ103"], result.summary
    assert "unused" in result.findings[0].message


def test_clean_model_trains():
    fixed = SEG.replace("stride=4, padding=1", "padding=1").replace("        self.unused = nn.Linear(4, 4)\n", "")
    result = run(fixed)
    assert result.ok and not result.findings, result.summary


def test_combination_rejects_a_patch_that_breaks_the_others():
    code = SEG.replace("stride=4, padding=1", "padding=1").replace("        self.unused = nn.Linear(4, 4)\n", "")
    lines = code.splitlines()
    head = lines.index("        self.head = nn.Conv2d(16, NUM_CLASSES, 1)") + 1
    const = lines.index("NUM_CLASSES = 5") + 1
    good = Patch(start_line=const, end_line=const, replacement="NUM_CLASSES = 5  # road, car, person, building, sky")
    # Shrinks the head input to 8 channels while the encoder still produces 16: the forward pass breaks.
    bad = Patch(start_line=head, end_line=head, replacement="        self.head = nn.Conv2d(8, NUM_CLASSES, 1)")
    result = verify_combination(code, GoalProfile(task="segmentation"), [2, 3, 64, 64], {}, [("good", good), ("bad", bad)])
    assert result["accepted"] == ["good"]
    assert [r["id"] for r in result["rejected"]] == ["bad"]
