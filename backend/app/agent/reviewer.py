"""LLM code review: problems the deterministic rules do not cover, each with an optional patch.

These findings are advisory. No deterministic check proves them, so their patches can only be verified to parse and
to introduce no new problems; the UI labels them as unverified.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from app import config
from app.agent import prompts
from app.agent.fixer import numbered
from app.agent.patching import normalize_indent
from app.llm.client import structured
from app.llm.text import plain
from app.agent import memory
from app.schemas import Category, Evidence, Finding, HistoryItem, OpenFinding, Patch
from app.tools.base import make_finding

log = logging.getLogger("uvicorn.error")
TOOL = "llm_review"
MAX_LINES = 600
# The review has no offline fallback and finds most semantic problems, so it waits out rate limits for longer.
REVIEW_MAX_WAIT = 30
REVIEW_HISTORY_RULES = (
    "These changes were made on purpose. Do not report the problems they fixed again, and do not suggest undoing "
    "them. Do check whether any of them introduced a new problem or no longer fits the rest of the script; if so, "
    "report it and say in the explanation that it was introduced by the earlier fix."
)
# Entries where the reviewer talked itself out of the problem ("No issue; os is already imported"). Matched only on
# the title and on how the explanation starts: a real explanation can say "make sure the shape is correct".
_NOT_A_PROBLEM = r"(no issue|not an issue|no problem|keep (it )?as is|nothing to (fix|change)|no change needed)"
NOT_A_PROBLEM = re.compile(rf"\b{_NOT_A_PROBLEM}\b", re.IGNORECASE)
NOT_A_PROBLEM_START = re.compile(rf"^\W*{_NOT_A_PROBLEM}\b", re.IGNORECASE)


def talked_out_of_it(title: str, explanation: str) -> bool:
    return bool(NOT_A_PROBLEM.search(title) or NOT_A_PROBLEM_START.search(explanation))


class ReviewFix(BaseModel):
    start_line: int
    end_line: int
    replacement: str


class ReviewIssue(BaseModel):
    # False when, after checking, the code turns out to be fine. Such entries are dropped.
    is_real_problem: bool = True
    title: str
    line: int
    end_line: int | None = None
    impact: Literal["crash", "silent", "risk"]
    category: Category = "other"
    explanation: str
    fix: ReviewFix | None = None


class Review(BaseModel):
    issues: list[ReviewIssue] = Field(default_factory=list)


def review(code: str, goal: str, existing: list[Finding], history: list[HistoryItem] | None = None,
           previous_open: list[OpenFinding] | None = None) -> list[tuple[Finding, Patch | None]]:
    """Raises LLMError (or any exception) if the review cannot run; the caller records it as skipped."""
    history = history or []
    lines = code.splitlines()
    shown = "\n".join(lines[:MAX_LINES])
    note = f"\n(The script continues for {len(lines) - MAX_LINES} more lines that are not shown.)" if len(lines) > MAX_LINES else ""
    already = [{"line": f.line, "title": f.title} for f in existing]
    parts = [
        f"Developer goal: {goal}",
        f"Already reported in this check (do not repeat): {json.dumps(already)}",
    ]
    if history:
        parts.append(memory.history_text(history, code) + "\n" + REVIEW_HISTORY_RULES)
    if previous_open:
        parts.append(memory.open_text(previous_open))
    parts.append(f"Script:\n{numbered(shown)}{note}")
    user = "\n\n".join(parts)
    result = structured(prompts.REVIEWER, user, Review, effort="medium", role="review", model=config.MODEL,
                        max_tokens=8192, max_wait=REVIEW_MAX_WAIT)

    taken = {l for f in existing for l in range(f.line, f.end_line + 1)}
    out: list[tuple[Finding, Patch | None]] = []
    for issue in result.issues[:25]:
        if not issue.is_real_problem or talked_out_of_it(issue.title, issue.explanation):
            log.info("[review] dropped (not a problem): %s", issue.title)
            continue
        if memory.is_rereport(issue.title, issue.explanation, history, code):
            log.info("[review] dropped (already fixed in this log): %s", issue.title)
            continue  # an applied fix (still in the code) already addressed this
        start = min(max(1, issue.line), len(lines))
        end = min(max(start, issue.end_line or start), len(lines))
        if start in taken:
            continue  # a deterministic check already covers this line, and its evidence is stronger
        taken.update(range(start, end + 1))
        patch = None
        if issue.fix and 1 <= issue.fix.start_line <= issue.fix.end_line <= len(lines):
            raw = Patch(start_line=issue.fix.start_line, end_line=issue.fix.end_line, replacement=issue.fix.replacement)
            current = "\n".join(lines[raw.start_line - 1:raw.end_line])
            if raw.replacement.rstrip("\n") != current:  # a patch that changes nothing is not a fix
                patch = normalize_indent(code, raw)
        finding = make_finding(
            "EZ100", plain(issue.title)[:140], plain(issue.explanation), start,
            [Evidence(tool=TOOL, summary="Found by the code reviewer; not proven by a deterministic check", lines=[start])],
            end_line=end, patchable=patch is not None, impact=issue.impact, category=issue.category,
            severity="error" if issue.impact == "crash" else "warning",
        )
        out.append((finding, patch))
    return out
