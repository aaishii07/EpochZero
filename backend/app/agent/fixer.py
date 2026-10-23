"""Finding + evidence -> FixProposal (explanation and line-range patch)."""

from __future__ import annotations

import json

from app import config
from app.agent import prompts
from app.llm import mock
from app.llm.client import structured
from app.schemas import Finding, FixProposal, GoalProfile, Patch


WINDOW = 15  # lines of context on each side of the finding
SHORT_SCRIPT = 120  # scripts up to this length are sent whole


def numbered(code: str) -> str:
    return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(code.splitlines(), 1))


def excerpt(code: str, finding: Finding) -> str:
    """Numbered source the fixer needs: the whole script if short, otherwise the lines around the finding and its
    evidence, plus module-level assignments (constants such as NUM_CLASSES). Line numbers stay those of the file."""
    lines = code.splitlines()
    if len(lines) <= SHORT_SCRIPT:
        return numbered(code)
    keep: set[int] = set()
    for n in {finding.line, finding.end_line, *(l for ev in finding.evidence for l in ev.lines)}:
        keep.update(range(max(1, n - WINDOW), min(len(lines), n + WINDOW) + 1))
    keep.update(i for i, l in enumerate(lines, 1) if l and not l[0].isspace() and "=" in l and not l.startswith(("#", "def ", "class ")))
    out, prev = [], 0
    for i in sorted(keep):
        if i != prev + 1:
            out.append("     | ...")
        out.append(f"{i:4d} | {lines[i - 1]}")
        prev = i
    return "\n".join(out)


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
        f"Finding:\n{json.dumps(_finding_payload(finding), default=str)[:3000]}",
    ]
    if finding.docs:
        parts.append(f"Documentation ({finding.docs.title}): {finding.docs.snippet}")
    if feedback:
        parts.append(f"The previous patch failed verification: {feedback}")
        if previous:
            parts.append(f"Previous patch: {previous.model_dump_json()}")
    parts.append(f"Script (excerpt, original line numbers):\n{excerpt(code, finding)}")
    try:
        return structured(prompts.FIXER, "\n\n".join(parts), FixProposal, effort="medium", role="fixer")
    except Exception:
        # Live repair failed; fall back to the offline repair for the rules it covers.
        return mock.fix(code, finding)
