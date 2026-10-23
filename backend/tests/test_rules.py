"""Static rules, goal rules, class count, and patching. Each rule: buggy fires, fixed does not."""

import textwrap

import pytest

from app.agent.patching import PatchError, apply_patch
from app.analysis.facts import build_facts
from app.schemas import GoalProfile, Patch
from app.tools.base import ToolContext
from app.tools.static import ast_rules, class_count, goal_rules, hyperparams

LOOP = """
import torch
import torch.nn as nn
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(3):
    for x, y in train_loader:
{body}
"""


def ctx(code, task="unknown", num_classes=None, answers=None):
    code = textwrap.dedent(code)
    return ToolContext(code=code, facts=build_facts(code), profile=GoalProfile(task=task, num_classes=num_classes),
                       answers=answers or {})


def rules(c):
    return sorted(f.rule_id for f in ast_rules.run(c) + hyperparams.run(c))


def loop(*lines):
    return LOOP.format(body="\n".join("        " + l for l in lines))


def test_missing_zero_grad():
    assert rules(ctx(loop("loss = criterion(model(x), y)", "loss.backward()", "optimizer.step()"))) == ["EZ006"]


def test_zero_grad_present_is_clean():
    assert rules(ctx(loop("optimizer.zero_grad()", "loss = criterion(model(x), y)", "loss.backward()", "optimizer.step()"))) == []


def test_accumulation_is_info_only():
    c = ctx(loop("loss = criterion(model(x), y) / 4", "loss.backward()", "if (epoch + 1) % 4 == 0:", "    optimizer.step()"))
    found = ast_rules.run(c)
    assert [f.rule_id for f in found] == ["EZ006"]
    assert found[0].severity == "info" and not found[0].patchable


def test_misplaced_zero_grad():
    assert rules(ctx(loop("loss = criterion(model(x), y)", "loss.backward()", "optimizer.zero_grad()", "optimizer.step()"))) == ["EZ007"]


def test_loss_accumulation_leak():
    base = ["optimizer.zero_grad()", "loss = criterion(model(x), y)", "loss.backward()", "optimizer.step()"]
    assert rules(ctx(loop(*base, "total += loss"))) == ["EZ008"]
    assert rules(ctx(loop(*base, "total += loss.item()"))) == []


def test_float_targets_to_cross_entropy():
    base = ["optimizer.zero_grad()", "y = y.float()", "loss = criterion(model(x), y)", "loss.backward()", "optimizer.step()"]
    assert rules(ctx(loop(*base))) == ["EZ010"]


def test_scheduler_step_is_not_optimizer_step():
    c = ctx(loop("optimizer.zero_grad()", "loss = criterion(model(x), y)", "loss.backward()", "optimizer.step()", "scheduler.step()"))
    assert rules(c) == []


DOUBLE = """
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)
criterion = nn.{loss}()
"""


def test_double_softmax():
    assert rules(ctx(DOUBLE.format(loss="CrossEntropyLoss"))) == ["EZ002"]
    assert rules(ctx(DOUBLE.format(loss="NLLLoss"))) == []


def test_goal_mismatch_regression_with_cross_entropy():
    found = goal_rules.run(ctx(DOUBLE.replace("F.softmax(self.fc(x), dim=1)", "self.fc(x)").format(loss="CrossEntropyLoss"),
                               task="regression"))
    assert [f.rule_id for f in found] == ["EZ003"]


VAL = """
from torchvision import transforms, datasets
train_tf = transforms.Compose([transforms.RandomCrop(32), transforms.ToTensor()])
val_tf = transforms.Compose([transforms.{aug}, transforms.ToTensor()])
val_set = datasets.ImageFolder("val", transform=val_tf)
"""


def test_eval_augmentation_leak():
    assert rules(ctx(VAL.format(aug="RandomHorizontalFlip()"))) == ["EZ005"]
    assert rules(ctx(VAL.format(aug="CenterCrop(32)"))) == []


def test_eval_mode_missing():
    code = """
    import torch
    model = Net()
    for x, y in val_loader:
        out = model(x)
    """
    assert rules(ctx(code)) == ["EZ009"]
    fixed = """
    import torch
    model = Net()
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
    """
    assert rules(ctx(fixed)) == []


def test_high_learning_rate_is_advisory():
    found = hyperparams.run(ctx("import torch\nopt = torch.optim.Adam(params, lr=0.1)\n"))
    assert [f.rule_id for f in found] == ["EZ011"] and found[0].tier == "advisory" and not found[0].patchable


HEAD = """
import torch.nn as nn
NUM_CLASSES = 5
CLASSES = {classes}
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(32, 7, 1)
    def forward(self, x):
        return self.head(x)
"""


def test_class_count_grounded():
    r = class_count.run(ctx(HEAD.format(classes='["a", "b", "c", "d", "e"]'), task="segmentation", num_classes=5))
    assert [f.rule_id for f in r.findings] == ["EZ004"] and r.findings[0].patchable and r.needs is None


def test_class_count_ambiguous_asks():
    r = class_count.run(ctx(HEAD.format(classes='["a", "b", "c", "d", "e", "f", "g"]'), task="segmentation"))
    assert r.needs == "num_classes" and not r.findings[0].patchable


def test_class_count_answer_grounds_it():
    r = class_count.run(ctx(HEAD.format(classes='["a", "b", "c", "d", "e", "f", "g"]'), task="segmentation",
                            answers={"num_classes": "7"}))
    assert r.needs is None and r.findings[0].line == 3  # NUM_CLASSES line is the one to change


def test_apply_patch():
    code = "a\nb\nc\n"
    assert apply_patch(code, Patch(start_line=2, end_line=2, replacement="B")) == "a\nB\nc\n"
    assert apply_patch(code, Patch(start_line=2, end_line=2, replacement="")) == "a\nc\n"
    with pytest.raises(PatchError):
        apply_patch(code, Patch(start_line=5, end_line=6, replacement="x"))


def test_is_unknown():
    from app.tools.base import is_unknown
    assert all(is_unknown(v) for v in ["Don't know", "dont know", "I don't know.", "unknown", "?", "Not sure"])
    assert not any(is_unknown(v) for v in ["5", "1,3,224,224", None])
