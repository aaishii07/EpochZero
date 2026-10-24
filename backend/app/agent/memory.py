"""What earlier checks of the same log did, sent by the frontend with every request.

The backend itself keeps no state; the log's history lives in the browser. With it, the reviewer and fixer know
which changes were made on purpose, so they do not re-report or undo them, and can check whether they broke
something.
"""

from __future__ import annotations

import re

from app.schemas import HistoryItem, OpenFinding

MAX_ITEMS = 25
MAX_LINES = 4
# An issue that says it was caused by an earlier change is a real report about that change, not a repeat.
CAUSED_BY_FIX = re.compile(r"\b(introduc|caused by|because of the (earlier|previous)|earlier fix|previous fix|new code)",
                           re.IGNORECASE)


def still_in_code(item: HistoryItem, code: str) -> bool:
    """True when the fix's new lines are still present, so it has not been changed or undone since."""
    added = [l.strip() for l in item.after if l.strip()]
    if not added:
        removed = [l.strip() for l in item.before if l.strip()]
        return bool(removed) and not all(r in code for r in removed)  # a deletion is in place if the text is gone
    return all(a in code for a in added)


def _clip(lines: list[str]) -> str:
    shown = [l.strip() for l in lines if l.strip()][:MAX_LINES]
    more = " ..." if len([l for l in lines if l.strip()]) > MAX_LINES else ""
    return " | ".join(shown) + more if shown else "(nothing)"


def history_text(history: list[HistoryItem], code: str) -> str:
    if not history:
        return ""
    rows = []
    for h in history[-MAX_ITEMS:]:
        state = "still in the code" if still_in_code(h, code) else "no longer in the code (changed or undone since)"
        rows.append(f"- check {h.check}, {h.rule_id} '{h.title}': replaced [{_clip(h.before)}] with [{_clip(h.after)}]; {state}")
    return "Fixes the developer already applied in this log, oldest first:\n" + "\n".join(rows)


def open_text(previous: list[OpenFinding]) -> str:
    if not previous:
        return ""
    rows = [f"- {p.rule_id} at line {p.line}: {p.title}" for p in previous[:MAX_ITEMS]]
    return "Problems the previous check reported that were not fixed (report them again if still present, with the " \
           "same wording):\n" + "\n".join(rows)


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9_]+", text.lower()) if len(w) > 3}


STOPWORDS = {"error", "issue", "problem", "missing", "wrong", "incorrect", "never", "with", "from", "that", "this",
              "does", "should", "instead", "causing", "cause", "causes", "line", "code", "when", "before", "after"}


def _key_words(text: str) -> set[str]:
    return _words(text) - STOPWORDS


def similar(a: str, b: str) -> bool:
    """Titles about the same thing: at least two shared content words and most of the words in common.
    Generic words ("error", "missing", "wrong") do not count, so "Loss error" does not match "Syntax error"."""
    wa, wb = _key_words(a), _key_words(b)
    common = wa & wb
    return len(common) >= 2 and len(common) / len(wa | wb) >= 0.5


def is_rereport(title: str, explanation: str, history: list[HistoryItem], code: str) -> bool:
    """A reviewer issue that repeats a problem a code-review fix (still in place) already addressed.

    Only code-review fixes count: the reviewer never reports syntax errors or rule findings, so those can not be
    repeated by it."""
    if CAUSED_BY_FIX.search(f"{title} {explanation}"):
        return False
    return any(h.rule_id == "EZ100" and similar(title, h.title) and still_in_code(h, code) for h in history)
