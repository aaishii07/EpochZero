"""Goal text -> GoalProfile."""

from __future__ import annotations

from app import config
from app.agent import prompts
from app.llm import mock
from app.llm.client import LLMError, structured
from app.schemas import GoalProfile


def read_goal(goal: str) -> tuple[GoalProfile, str]:
    """Returns the profile and where it came from ("llm" or "offline")."""
    if config.LLM_MODE == "mock" or not goal.strip():
        return mock.read_goal(goal), "offline"
    try:
        profile = structured(prompts.GOAL_READER, f"Goal:\n{goal}", GoalProfile, effort="low")
        if profile.num_classes is not None and profile.num_classes < 1:
            profile.num_classes = None
        return profile, "llm"
    except (LLMError, Exception):
        return mock.read_goal(goal), "offline"
