"""Apply patches, rerun the checks, and decide resolved / unresolved / inconclusive.

No LLM is involved here. Two entry points:
- Verifier.verify: one patch for one finding, against the analyzed code.
- verify_combination: several patches applied together, added one at a time; any patch that makes the combined
  script worse than before is left out, with the reason.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field

from app.agent.patching import PatchError, apply_patch
from app.analysis.facts import build_facts
from app.schemas import Finding, GoalProfile, Patch, VerifyStatus
from app.tools.base import ToolContext
from app.tools.registry import run_tool, tools_for_findings


@dataclass
class CheckState:
    """What a set of checks reported on one version of the script."""
    counts: Counter = field(default_factory=Counter)
    findings: list[Finding] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # tool -> why it could not run
    ran: set[str] = field(default_factory=set)

    @property
    def problems(self) -> list[str]:
        return [f"{name}: {why}" for name, why in self.failed.items()]


def run_state(code: str, profile: GoalProfile, input_shape, answers, tools: list[str]) -> CheckState:
    """input_shape must be the shape the finding was measured with, so before and after use the same input."""
    state = CheckState()
    try:
        facts = build_facts(code)
    except SyntaxError as e:
        state.failed["parse"] = f"{e.msg} at line {e.lineno}"
        return state
    ctx = ToolContext(code=code, facts=facts, profile=profile, input_shape=input_shape, answers=answers)
    for name in tools:
        result = run_tool(name, ctx)
        if result.ok:
            state.ran.add(name)
        else:
            state.failed[name] = result.summary
        state.counts.update(f.rule_id for f in result.findings)
        state.findings += result.findings
    return state


def run_checks(code: str, profile: GoalProfile, input_shape, answers, tools: list[str]) -> tuple[Counter, list[str]]:
    state = run_state(code, profile, input_shape, answers, tools)
    return state.counts, state.problems


def regressions(before: CheckState, after: CheckState) -> list[str]:
    """Ways the after state is worse: a check that ran before can no longer run, or a rule fires more often."""
    out = []
    # New findings first: they name the actual cause.
    for rule in sorted(r for r in after.counts if after.counts[r] > before.counts[r]):
        remaining = Counter(f.title for f in before.findings if f.rule_id == rule)
        fresh = []
        for f in after.findings:
            if f.rule_id != rule:
                continue
            if remaining[f.title] > 0:
                remaining[f.title] -= 1
            else:
                fresh.append(f)
        title = fresh[0].title if fresh else rule
        line = f" at line {fresh[0].line}" if fresh else ""
        out.append(f"{rule}: {title}{line}")
    # Then checks that ran before and no longer can. A check skipped because an earlier one failed is a consequence,
    # not a separate problem.
    out += [f"{name.replace('_', ' ')} can no longer run ({why[:120]})" for name, why in after.failed.items()
            if (name in before.ran or name == "parse") and not why.startswith("Skipped:")]
    return out


def measured_shape(finding: Finding) -> list[int] | None:
    """The input shape the shape probe used when it produced (or supported) this finding."""
    for ev in finding.evidence:
        shape = ev.data.get("input_shape")
        if isinstance(shape, list) and shape and all(isinstance(d, int) for d in shape):
            return shape
    return None


def safety_tools(code: str) -> list[str]:
    """Checks that can tell whether a patch broke something, even when no check covers the finding itself."""
    try:
        has_model = bool(build_facts(code).model_classes)
    except SyntaxError:
        has_model = False
    return ["run_static_rules", *(["probe_shapes", "dry_run"] if has_model else [])]


class Verifier:
    def __init__(self, original: str, profile: GoalProfile, input_shape, answers):
        self.original = original
        self.profile = profile
        self.input_shape = input_shape
        self.answers = answers
        self._baseline: dict[tuple, CheckState] = {}

    def _before(self, tools: list[str], shape) -> CheckState:
        key = (tuple(tools), tuple(shape or ()))
        if key not in self._baseline:
            self._baseline[key] = run_state(self.original, self.profile, shape, self.answers, tools)
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
            # Code-review findings: nothing reproduces them, so check that the patch breaks nothing the rules, the
            # shape probe, or a real training step can see.
            tools = safety_tools(self.original)
            before = self._before(tools, self.input_shape)
            after = run_state(patched, self.profile, self.input_shape, self.answers, tools)
            worse = regressions(before, after)
            if worse:
                return "unresolved", f"The patch makes the script worse: {worse[0]}"
            ran = ", ".join(t.replace("_", " ") for t in tools if t in after.ran)
            return "inconclusive", (f"Unverified: no automated check covers this kind of issue. The patched script "
                                    f"parses and nothing new was found ({ran}).")
        shape = measured_shape(finding) or self.input_shape
        before = self._before(tools, shape)
        after = run_state(patched, self.profile, shape, self.answers, tools)

        if after.counts[finding.rule_id] >= before.counts[finding.rule_id]:
            if after.problems:
                return "inconclusive", f"Check could not run on the patched code ({after.problems[0]})"
            return "unresolved", f"{finding.rule_id} is still reported after the patch"
        worse = [w for w in regressions(before, after) if not w.startswith(finding.rule_id)]
        if worse:
            return "unresolved", f"The patch introduced a new problem: {worse[0]}"
        if after.problems:
            return "inconclusive", f"Finding is gone, but a check could not run on the patched code ({after.problems[0]})"
        names = ", ".join(t.replace("_", " ") for t in tools)
        return "resolved", f"Reran {names} on the patched code: {finding.rule_id} no longer reported, nothing new introduced"


def verify_combination(original: str, profile: GoalProfile, input_shape, answers,
                       patches: list[tuple[str, Patch]], baseline_ids: list[str] | None = None) -> dict:
    """Add patches one at a time onto `original` and keep each one only if the combined script gets no worse.

    `baseline_ids` are patches already applied and accepted by the developer; they are always kept and form the
    starting point. Patches must not overlap. Returns accepted ids, rejected ids with a reason, and a summary.
    """
    tools = safety_tools(original)
    by_id = dict(patches)
    kept = [pid for pid in (baseline_ids or []) if pid in by_id]

    def compose(ids: list[str]) -> str:
        code = original
        for pid in sorted(ids, key=lambda i: by_id[i].start_line, reverse=True):
            code = apply_patch(code, by_id[pid])
        return code

    state = run_state(compose(kept), profile, input_shape, answers, tools)
    accepted, rejected = [], []
    for pid, patch in patches:
        if pid in kept:
            continue
        trial_ids = kept + [pid]
        try:
            code = compose(trial_ids)
        except PatchError as e:
            rejected.append({"id": pid, "reason": f"could not be applied together with the other fixes ({e})"})
            continue
        trial = run_state(code, profile, input_shape, answers, tools)
        worse = regressions(state, trial)
        if worse:
            rejected.append({"id": pid, "reason": f"together with the other fixes it causes: {worse[0]}"})
            continue
        kept, state = trial_ids, trial
        accepted.append(pid)
    checks = ", ".join(t.replace("_", " ") for t in tools if t in state.ran) or "parse"
    summary = (f"Checked the fixes together ({checks}): {len(accepted)} kept"
               + (f", {len(rejected)} left out" if rejected else ""))
    return {"accepted": accepted, "rejected": rejected, "summary": summary}
