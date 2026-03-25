"""fetch_file, fetch_files_batch, get_code_file_structure, analyze_code_structure."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from pydantic_ai import RunContext

from poc.managers.deps import PoCDeepDeps


def _root(ctx: RunContext[PoCDeepDeps]) -> Path:
    return Path(ctx.deps.poc_run.project_root)


async def fetch_file(
    ctx: RunContext[PoCDeepDeps],
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    p = _root(ctx) / path
    if not p.exists():
        return f"Error: not found: {path}"
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if start_line is not None or end_line is not None:
        s = (start_line or 1) - 1
        e = end_line if end_line is not None else len(lines)
        lines = lines[s:e]
    return "\n".join(lines)


async def fetch_files_batch(
    ctx: RunContext[PoCDeepDeps], paths: list[str]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        out[path] = await fetch_file(ctx, path)
    return out


async def get_code_file_structure(
    ctx: RunContext[PoCDeepDeps], subdir: str = "", max_entries: int = 500
) -> str:
    base = _root(ctx) / subdir
    if not base.exists():
        return f"Error: {subdir} not found"
    lines: list[str] = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv")]
        for fn in filenames:
            if n >= max_entries:
                lines.append("...(truncated)")
                return "\n".join(lines)
            rel = os.path.relpath(os.path.join(dirpath, fn), _root(ctx))
            lines.append(rel)
            n += 1
    return "\n".join(lines)


async def analyze_code_structure(ctx: RunContext[PoCDeepDeps], path: str) -> str:
    p = _root(ctx) / path
    if not p.exists():
        return "not found"
    text = p.read_text(encoding="utf-8", errors="replace")
    if not path.endswith(".py"):
        return f"non-python file, {len(text)} chars"
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return f"parse error: {e}"
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out.append(f"class {node.name} L{node.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {node.name} L{node.lineno}")
    return "\n".join(out[:200])
