"""Finding + evidence -> FixProposal (explanation and line-range patch)."""

from __future__ import annotations

import json

from app import config
from app.agent import prompts
from app.llm import mock
from app.llm.client import structured
from app.schemas import Finding, FixProposal, GoalProfile, Patch


def numbered(code: str) -> str:
    return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(code.splitlines(), 1))


def _finding_payload(f: Finding) -> dict:
    d = f.model_dump(exclude={"docs"})
    for ev in d["evidence"]:
        ev["data"].pop("records", None)
    return d


def propose(code: str, goal: str, profile: GoalProfile, finding: Finding,
            feedback: str | None = None, previous: Patch | None = None) -> FixProposal:
    if config.LLM_MODE == "mock":
        return mock.fix(code, finding)
    parts = [
        f"Developer goal: {goal}",
        f"Task profile: {profile.model_dump_json()}",
        f"Finding:\n{json.dumps(_finding_payload(finding), indent=1, default=str)[:6000]}",
    ]
    if finding.docs:
        parts.append(f"Documentation ({finding.docs.title}): {finding.docs.snippet}")
    if feedback:
        parts.append(f"The previous patch failed verification: {feedback}")
        if previous:
            parts.append(f"Previous patch: {previous.model_dump_json()}")
    parts.append(f"Script:\n{numbered(code)}")
    try:
        return structured(prompts.FIXER, "\n\n".join(parts), FixProposal, effort="medium")
    except Exception:
        # Live repair failed; fall back to the offline repair for the rules it covers.
        return mock.fix(code, finding)
