"""Line-range patch application. frontend/lib/patch.ts implements the same logic."""

from __future__ import annotations

from app.config import MAX_PATCH_LINES
from app.schemas import Patch


class PatchError(ValueError):
    pass


def apply_patch(code: str, patch: Patch) -> str:
    newline = "\r\n" if "\r\n" in code else "\n"
    lines = code.split(newline)
    if patch.start_line < 1 or patch.end_line < patch.start_line or patch.end_line > len(lines):
        raise PatchError(f"patch range {patch.start_line}-{patch.end_line} is outside the file (1-{len(lines)})")
    if patch.end_line - patch.start_line + 1 > MAX_PATCH_LINES:
        raise PatchError(f"patch spans more than {MAX_PATCH_LINES} lines")
    replacement = patch.replacement.replace("\r\n", "\n")
    if replacement.endswith("\n"):
        replacement = replacement[:-1]
    new = [] if replacement == "" and patch.replacement == "" else replacement.split("\n")
    return newline.join(lines[: patch.start_line - 1] + new + lines[patch.end_line:])
