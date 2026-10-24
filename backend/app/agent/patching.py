"""Line-range patch application. frontend/lib/patch.ts implements the same logic."""

from __future__ import annotations

from app.schemas import Patch


class PatchError(ValueError):
    pass


def apply_patch(code: str, patch: Patch) -> str:
    newline = "\r\n" if "\r\n" in code else "\n"
    lines = code.split(newline)
    if patch.start_line < 1 or patch.end_line < patch.start_line or patch.end_line > len(lines):
        raise PatchError(f"patch range {patch.start_line}-{patch.end_line} is outside the file (1-{len(lines)})")
    replacement = patch.replacement.replace("\r\n", "\n")
    if replacement.endswith("\n"):
        replacement = replacement[:-1]
    new = [] if replacement == "" and patch.replacement == "" else replacement.split("\n")
    return newline.join(lines[: patch.start_line - 1] + new + lines[patch.end_line:])


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _shift(lines: list[str], delta: int) -> list[str]:
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        elif delta >= 0:
            out.append(" " * delta + line)
        else:
            out.append(line[min(-delta, _indent(line)):])
    return out


def normalize_indent(code: str, patch: Patch) -> Patch:
    """Fix the indentation mistakes LLMs make in line-range patches.

    Typical mistakes: the first line loses its indentation while the rest keep theirs, or the whole block is shifted
    left or right. Candidates are built from the indentation of the lines being replaced, and the first one that
    keeps the file parseable wins; if none parses, the patch is returned unchanged and verification reports it.
    """
    import ast

    lines = code.replace("\r\n", "\n").split("\n")
    if not (1 <= patch.start_line <= patch.end_line <= len(lines)) or not patch.replacement.strip():
        return patch
    original = [l for l in lines[patch.start_line - 1:patch.end_line] if l.strip()]
    if not original:
        return patch
    base = _indent(original[0].expandtabs(4))
    repl = patch.replacement.replace("\r\n", "\n").expandtabs(4).rstrip("\n").split("\n")
    body = [l for l in repl if l.strip()]
    first = _indent(body[0])
    rest_min = min((_indent(l) for l in body[1:]), default=None)

    candidates = [repl]
    if first != base:
        candidates.append(_shift(repl, base - first))  # whole block shifted
        if rest_min is not None and rest_min >= base:
            # Only the first line lost (or gained) its indentation.
            i = next(k for k, l in enumerate(repl) if l.strip())
            fixed = list(repl)
            fixed[i] = " " * base + fixed[i].lstrip(" ")
            candidates.insert(1, fixed)
    # Prefer a candidate whose first line matches the indentation of the code it replaces.
    candidates.sort(key=lambda c: _indent(next(l for l in c if l.strip())) != base)
    for cand in candidates:
        trial = Patch(start_line=patch.start_line, end_line=patch.end_line, replacement="\n".join(cand))
        try:
            ast.parse(apply_patch(code, trial))
            return trial
        except (SyntaxError, PatchError):
            continue
    return patch
