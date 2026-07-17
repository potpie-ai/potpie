"""Static ownership locks for the product/engine split.

The root ``potpie`` distribution owns every delivery surface (CLI, daemon,
MCP, host-app services); ``potpie-context-engine`` is the runtime underneath.
The dependency arrow points strictly downward: the engine must never import
the ``potpie`` product namespace.
"""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "potpie" / "context-engine"

_SKIP_PARTS = {".venv", "__pycache__", "node_modules"}


def _python_files(search_root: Path):
    for path in search_root.rglob("*.py"):
        if _SKIP_PARTS.isdisjoint(path.parts):
            yield path


def _imports_namespace(path: Path, namespace: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == namespace or node.module.startswith(f"{namespace}."):
                return True
        if isinstance(node, ast.Import):
            if any(
                alias.name == namespace or alias.name.startswith(f"{namespace}.")
                for alias in node.names
            ):
                return True
    return False


def test_legacy_cli_namespace_is_not_imported() -> None:
    legacy_namespace = "potpie_context_engine." + ".".join(
        ("adapters", "inbound", "cli")
    )
    offenders = {
        path.relative_to(ROOT).as_posix()
        for search_root in (ROOT / "potpie", ROOT / "tests")
        for path in _python_files(search_root)
        if _imports_namespace(path, legacy_namespace)
    }
    assert offenders == set()


def test_engine_never_imports_the_potpie_product_namespace() -> None:
    engine_src = ENGINE_ROOT / "src"
    offenders = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for path in _python_files(engine_src)
        if _imports_namespace(path, "potpie")
    }
    assert offenders == set()


def test_engine_metadata_does_not_depend_on_root_potpie() -> None:
    engine_metadata = tomllib.loads(
        (ENGINE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = engine_metadata["project"]["dependencies"]
    dependency_names = {
        re.split(r"[\s<>=!~;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in dependencies
    }
    assert "potpie" not in dependency_names


def test_root_console_scripts_target_relocated_product_modules() -> None:
    root_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = root_metadata["project"]["scripts"]
    assert scripts["potpie"] == "potpie.cli.main:main"
    assert scripts["potpie-daemon"] == "potpie.daemon.main:main"
    assert scripts["potpie-mcp"] == "potpie.mcp.server:main"
