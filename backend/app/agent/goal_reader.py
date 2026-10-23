"""Goal text -> GoalProfile, or, when no goal is given, the task inferred from the code."""

from __future__ import annotations

import json

from app import config
from app.agent import prompts
from app.analysis.facts import CodeFacts
from app.llm import mock
from app.llm.client import structured
from app.schemas import GoalProfile

TASK_TEXT = {
    "binary_classification": "binary classification",
    "multiclass_classification": "multi-class classification",
    "multilabel_classification": "multi-label classification",
    "regression": "regression",
    "segmentation": "segmentation",
    "unknown": "unknown task",
}


def summarize(profile: GoalProfile) -> str:
    """One line that also works as a goal text, for example '10-class image classification'."""
    task = TASK_TEXT[profile.task]
    if profile.num_classes and profile.task in {"multiclass_classification", "multilabel_classification", "segmentation"}:
        task = f"{profile.num_classes}-class {task.replace('multi-class ', '')}"
    if profile.modality == "unknown" or profile.task == "unknown":
        return task
    # Put the modality right before the last word: "10-class image classification", "5-class image segmentation".
    head, _, tail = task.rpartition(" ")
    return f"{head} {profile.modality} {tail}".strip() if head else f"{profile.modality} {tail}"


def _clean(profile: GoalProfile) -> GoalProfile:
    if profile.num_classes is not None and profile.num_classes < 1:
        profile.num_classes = None
    return profile


def read_goal(goal: str, facts: CodeFacts, code: str) -> tuple[GoalProfile, str, bool]:
    """Returns (profile, source, inferred). source is "llm" or "offline"."""
    if goal.strip():
        if config.LLM_MODE == "mock":
            return mock.read_goal(goal), "offline", False
        try:
            return _clean(structured(prompts.GOAL_READER, f"Goal:\n{goal}", GoalProfile, effort="low", role="goal",
                                  model=config.FAST_MODEL)), "llm", False
        except Exception:
            return mock.read_goal(goal), "offline", False

    if config.LLM_MODE == "mock":
        return mock.infer_goal(facts), "offline", True
    lines = code.splitlines()
    excerpt = "\n".join(lines[:120]) + ("\n# ... (truncated)" if len(lines) > 120 else "")
    hints = {"loss": facts.loss.type if facts.loss else None, "head": facts.head.type if facts.head else None,
             "head_out": facts.head_out, "data_hints": facts.data_hints}
    try:
        profile = structured(prompts.GOAL_FROM_CODE, f"Extracted hints: {json.dumps(hints, default=str)}\n\nScript:\n{excerpt}",
                             GoalProfile, effort="low", role="goal", model=config.FAST_MODEL)
        return _clean(profile), "llm", True
    except Exception:
        return mock.infer_goal(facts), "offline", True
