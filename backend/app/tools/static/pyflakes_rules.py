"""EZ012-EZ013: deterministic Python errors found by pyflakes (undefined names, misuse that fails at runtime)."""

from __future__ import annotations

import ast

from pyflakes import checker, messages

from app.schemas import Evidence, Finding
from app.tools.base import ToolContext, make_finding

TOOL = "static_rules"

# Messages that mean the script will fail when that line runs.
CRASH = {
    messages.UndefinedName: "Undefined name",
    messages.UndefinedLocal: "Local variable used before assignment",
    messages.UndefinedExport: "Undefined name in __all__",
    messages.DuplicateArgument: "Duplicate argument",
    messages.ReturnOutsideFunction: "return outside a function",
    messages.YieldOutsideFunction: "yield outside a function",
    messages.ContinueOutsideLoop: "continue outside a loop",
    messages.BreakOutsideLoop: "break outside a loop",
    messages.TooManyExpressionsInStarredAssignment: "Too many expressions in starred assignment",
    messages.TwoStarredExpressions: "Two starred expressions in assignment",
    messages.FStringMissingPlaceholders: None,  # harmless; listed so it is explicitly ignored
}
# Messages that point at a likely mistake but do not crash.
RISK = {
    messages.RedefinedWhileUnused: "Name redefined before it was used",
    messages.ImportShadowedByLoopVar: "Import shadowed by a loop variable",
    messages.IsLiteral: "'is' used to compare with a literal",
    messages.MultiValueRepeatedKeyLiteral: "Dictionary key repeated with different values",
    messages.PercentFormatMissingArgument: "Format string is missing an argument",
    messages.StringDotFormatMissingArgument: "Format string is missing an argument",
}


def run(ctx: ToolContext) -> list[Finding]:
    try:
        tree = ast.parse(ctx.code)
    except SyntaxError:
        return []
    found: list[Finding] = []
    seen: set[tuple[int, str]] = set()
    for msg in checker.Checker(tree, "script.py").messages:
        text = msg.message % msg.message_args
        key = (msg.lineno, text)
        if key in seen:
            continue
        seen.add(key)
        kind = type(msg)
        if kind in CRASH and CRASH[kind]:
            found.append(make_finding(
                "EZ012", f"{CRASH[kind]}: {text}",
                f"Line {msg.lineno}: {text}. Python raises an error when this line runs, so training stops here.",
                msg.lineno,
                [Evidence(tool=TOOL, summary=f"pyflakes: {text}", lines=[msg.lineno], data={"check": kind.__name__})],
            ))
        elif kind in RISK:
            found.append(make_finding(
                "EZ013", f"{RISK[kind]}: {text}",
                f"Line {msg.lineno}: {text}. This usually means a typo or a leftover line.",
                msg.lineno,
                [Evidence(tool=TOOL, summary=f"pyflakes: {text}", lines=[msg.lineno], data={"check": kind.__name__})],
            ))
    return found
