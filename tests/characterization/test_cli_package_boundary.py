"""Static ownership locks for the Phase 1 CLI relocation."""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "potpie" / "context-engine"

EXPECTED_ENGINE_CLI_IMPORTERS = {
    "adapters/outbound/daemon_process/launcher.py",
    "adapters/outbound/skills/agent_installer.py",
    "adapters/outbound/skills/bundle_catalog.py",
    "bootstrap/sentry_metrics_runtime.py",
    "host/daemon_main.py",
}


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


def _references_namespace(path: Path, namespace: str) -> bool:
    if _imports_namespace(path, namespace):
        return True
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, ast.Constant) and node.value == namespace
        for node in ast.walk(tree)
    )


def test_legacy_cli_namespace_is_not_imported() -> None:
    legacy_namespace = ".".join(("adapters", "inbound", "cli"))
    offenders = {
        path.relative_to(ROOT).as_posix()
        for search_root in (ROOT / "potpie", ROOT / "tests")
        for path in search_root.rglob("*.py")
        if _imports_namespace(path, legacy_namespace)
    }
    assert offenders == set()


def test_temporary_engine_to_cli_imports_are_explicitly_bounded() -> None:
    importers = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for path in ENGINE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(ENGINE_ROOT).parts
        and _references_namespace(path, "potpie.cli")
    }
    assert importers == EXPECTED_ENGINE_CLI_IMPORTERS


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


def test_root_console_script_targets_relocated_cli() -> None:
    root_metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert root_metadata["project"]["scripts"]["potpie"] == "potpie.cli.main:main"
