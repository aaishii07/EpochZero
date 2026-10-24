"""Indentation repair for LLM patches, and the reviewer's "no issue" filter."""

import ast

from app.agent.patching import apply_patch, normalize_indent
from app.agent.reviewer import NOT_A_PROBLEM
from app.schemas import Patch

CODE = "\n".join([
    "class Data:",
    "    def __getitem__(self, idx):",
    "        image = load(idx)",
    "        mask = load_mask(idx)",
    "        return image, mask",
    "",
])


def patched(replacement: str) -> str:
    p = normalize_indent(CODE, Patch(start_line=4, end_line=4, replacement=replacement))
    out = apply_patch(CODE, p)
    ast.parse(out)
    return out


def test_first_line_lost_its_indentation():
    out = patched("mask = load_mask(idx)\n        mask = mask.long()")
    assert "        mask = load_mask(idx)\n        mask = mask.long()" in out


def test_whole_block_shifted_left():
    out = patched("mask = load_mask(idx)\nmask = mask.long()")
    assert "        mask = load_mask(idx)\n        mask = mask.long()" in out


def test_whole_block_shifted_right():
    out = patched("            mask = load_mask(idx)\n            mask = mask.long()")
    assert "\n        mask = load_mask(idx)\n        mask = mask.long()" in out


def test_relative_indentation_is_kept():
    out = patched("if idx >= 0:\n    mask = load_mask(idx)")
    assert "        if idx >= 0:\n            mask = load_mask(idx)" in out


def test_correct_patch_is_untouched():
    p = Patch(start_line=4, end_line=4, replacement="        mask = load_mask(idx).long()")
    assert normalize_indent(CODE, p) == p


def test_tabs_become_spaces():
    out = patched("\t\tmask = load_mask(idx).long()")
    assert "        mask = load_mask(idx).long()" in out


def test_large_patches_apply():
    code = "\n".join(f"x{i} = {i}" for i in range(200))
    new = apply_patch(code, Patch(start_line=1, end_line=150, replacement="y = 1"))
    assert new.startswith("y = 1\nx150 = 150")


def test_no_issue_entries_are_recognised():
    assert NOT_A_PROBLEM.search("No issue; os is already imported.")
    assert NOT_A_PROBLEM.search("Missing import for os (already imported) but no error; keep as is")
    assert not NOT_A_PROBLEM.search("Masks are loaded as RGB floats, which CrossEntropyLoss cannot accept.")
