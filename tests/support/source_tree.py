"""Parse each source once for repository architecture contracts."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def python_trees(root: Path) -> dict[str, ast.Module]:
    return {
        path.relative_to(root).as_posix(): ast.parse(
            path.read_text(), filename=str(path)
        )
        for directory in ("potpie", "tests", "scripts")
        for path in (root / directory).rglob("*.py")
        if not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(root).parts
        )
    }


def imports_namespace(tree: ast.AST, namespace: str) -> bool:
    for node in ast.walk(tree):
        names = (
            (node.module or "",)
            if isinstance(node, ast.ImportFrom)
            else tuple(alias.name for alias in node.names)
            if isinstance(node, ast.Import)
            else ()
        )
        if any(name == namespace or name.startswith(f"{namespace}.") for name in names):
            return True
    return False


def has_literal(tree: ast.AST, value: str) -> bool:
    return any(
        isinstance(node, ast.Constant) and node.value == value
        for node in ast.walk(tree)
    )


def definitions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def nested_strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from nested_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_strings(item)
    elif isinstance(value, str):
        yield value
