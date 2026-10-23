"""Shared types for tools: the context every tool receives and the result it returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.analysis.facts import CodeFacts
from app.knowledge.docs import RULES
from app.schemas import Evidence, Finding, GoalProfile, SkippedCheck


@dataclass
class ToolContext:
    code: str
    facts: CodeFacts
    profile: GoalProfile
    input_shape: list[int] | None = None
    answers: dict[str, str] = field(default_factory=dict)
    # True when the profile was inferred from the code; then it is not independent evidence about the code.
    goal_inferred: bool = False
    # Filled by shape_probe so later tools (class count) can use measured output shapes.
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    summary: str
    findings: list[Finding] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    skipped: list[SkippedCheck] = field(default_factory=list)
    # When set, the tool could not run without this information (for example "input_shape").
    needs: str | None = None
    question: str | None = None


_UNKNOWN = {"", "dont know", "do not know", "unknown", "not sure", "idk", "no idea", "none", "na", "n/a", "skip", "?"}


def is_unknown(answer: str | None) -> bool:
    """True when the developer answered that they do not know (for example "Don't know")."""
    if answer is None:
        return False
    text = str(answer).strip().lower().replace("'", "").replace("’", "").rstrip(".!")
    return text in _UNKNOWN or text.startswith(("i dont know", "dont know", "i do not know"))


def make_finding(rule_id: str, title: str, message: str, line: int, evidence: list[Evidence],
                 end_line: int | None = None, patchable: bool = True, **overrides) -> Finding:
    rule = RULES[rule_id]
    return Finding(
        rule_id=rule_id,
        title=title,
        message=message,
        severity=overrides.get("severity", rule["severity"]),
        tier=overrides.get("tier", rule["tier"]),
        impact=overrides.get("impact", rule["impact"]),
        line=line,
        end_line=end_line or line,
        evidence=evidence,
        patchable=patchable,
        docs=rule["docs"],
    )
