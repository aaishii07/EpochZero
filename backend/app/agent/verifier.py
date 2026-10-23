"""Apply a patch, rerun the tool(s) that produced the finding, and decide resolved / unresolved / inconclusive.

No LLM is involved here.
"""

from __future__ import annotations

import ast
from collections import Counter

from app.agent.patching import PatchError, apply_patch
from app.analysis.facts import build_facts
from app.schemas import Finding, GoalProfile, Patch, VerifyStatus
from app.tools.base import ToolContext
from app.tools.registry import run_tool, tools_for_findings


def run_checks(code: str, profile: GoalProfile, input_shape, answers, tools: list[str]) -> tuple[Counter, list[str]]:
    """Returns rule-id counts and the list of reasons any tool could not run.

    input_shape must be the shape the finding was measured with, so before and after are compared on the same input.
    """
    ctx = ToolContext(code=code, facts=build_facts(code), profile=profile, input_shape=input_shape, answers=answers)
    counts: Counter = Counter()
    problems: list[str] = []
    for name in tools:
        result = run_tool(name, ctx)
        if not result.ok:
            problems.append(f"{name}: {result.summary}")
        counts.update(f.rule_id for f in result.findings)
    return counts, problems


def measured_shape(finding: Finding) -> list[int] | None:
    """The input shape the shape probe used when it produced (or supported) this finding."""
    for ev in finding.evidence:
        shape = ev.data.get("input_shape")
        if isinstance(shape, list) and shape and all(isinstance(d, int) for d in shape):
            return shape
    return None


class Verifier:
    def __init__(self, original: str, profile: GoalProfile, input_shape, answers):
        self.original = original
        self.profile = profile
        self.input_shape = input_shape
        self.answers = answers
        self._baseline: dict[tuple, tuple[Counter, list[str]]] = {}

    def _before(self, tools: list[str], shape):
        key = (tuple(tools), tuple(shape or ()))
        if key not in self._baseline:
            self._baseline[key] = run_checks(self.original, self.profile, shape, self.answers, tools)
        return self._baseline[key]

    def verify(self, patch: Patch, finding: Finding) -> tuple[VerifyStatus, str]:
        try:
            patched = apply_patch(self.original, patch)
        except PatchError as e:
            return "unresolved", f"Patch could not be applied: {e}"
        try:
            ast.parse(patched)
        except SyntaxError as e:
            return "unresolved", f"Patched code does not parse: {e.msg} at line {e.lineno}"

        tools = tools_for_findings({ev.tool for ev in finding.evidence})
        if not tools:
            return "inconclusive", "No check can reproduce this finding"
        shape = measured_shape(finding) or self.input_shape
        before, _ = self._before(tools, shape)
        after, problems = run_checks(patched, self.profile, shape, self.answers, tools)

        if after[finding.rule_id] >= before[finding.rule_id]:
            if problems:
                return "inconclusive", f"Check could not run on the patched code ({problems[0]})"
            return "unresolved", f"{finding.rule_id} is still reported after the patch"
        introduced = sorted(r for r in after if after[r] > before[r])
        if introduced:
            return "unresolved", f"The patch introduced new findings: {', '.join(introduced)}"
        if problems:
            return "inconclusive", f"Finding is gone, but a check could not run on the patched code ({problems[0]})"
        names = ", ".join(t.replace("_", " ") for t in tools)
        return "resolved", f"Reran {names} on the patched code: {finding.rule_id} no longer reported, nothing new introduced"
