"""The temporary package contains re-exports, never a second implementation."""

from __future__ import annotations

# ruff: noqa: S101 - pytest compatibility tests use assertions intentionally.

import ast
from pathlib import Path


def test_every_compatibility_module_contains_only_imports_and_all_assignment() -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "potpie_context_core"
    modules = sorted(package.rglob("*.py"))

    assert modules
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        implementation_nodes = [
            node
            for node in tree.body
            if not isinstance(
                node,
                (
                    ast.Expr,
                    ast.ImportFrom,
                    ast.Assign,
                ),
            )
        ]
        assert implementation_nodes == [], module
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        assert any(
            (node.module or "").startswith("potpie_context_engine.core")
            for node in imports
        ), module
