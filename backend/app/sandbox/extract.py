"""Build the reduced script that the sandbox executes.

Kept: imports (each guarded), module-level constant assignments, class and function definitions.
Dropped: everything else at module level, including training loops, data loading, file and network I/O,
and `if __name__ == "__main__":` blocks. Definitions are not called, so their bodies do not run.
"""

from __future__ import annotations

import ast

SAFE_CALLS = {"device", "dtype"}


def _is_constant_assign(node: ast.Assign) -> bool:
    try:
        ast.literal_eval(node.value)
        return True
    except Exception:
        pass
    if isinstance(node.value, ast.Call):
        func = node.value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        return name in SAFE_CALLS
    return False


def extract(code: str) -> str:
    tree = ast.parse(code)
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append("try:\n    " + ast.unparse(node) + "\nexcept Exception:\n    pass")
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(ast.unparse(node))
        elif isinstance(node, ast.Assign) and _is_constant_assign(node):
            out.append(ast.unparse(node))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                ast.literal_eval(node.value)
                out.append(ast.unparse(node))
            except Exception:
                pass
    return "\n\n".join(out) + "\n"
