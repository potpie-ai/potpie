"""Static ownership locks for the daemon relocation and MCP removal."""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "potpie" / "context-engine"

EXPECTED_ENGINE_DAEMON_IMPORTERS = {
    "src/potpie_context_engine/application/services/setup_orchestrator.py",
    "src/potpie_context_engine/bootstrap/host_wiring.py",
    "src/potpie_context_engine/host/__init__.py",
    "src/potpie_context_engine/host/shell.py",
}

LEGACY_DAEMON_NAMESPACES = {
    "potpie_context_engine.adapters.inbound.daemon_http",
    "potpie_context_engine.adapters.inbound.http.ui",
    "potpie_context_engine.adapters.outbound.daemon_process",
    "potpie_context_engine.adapters.outbound.managed_services",
    "potpie_context_engine.application.services.managed_service_manager",
    "potpie_context_engine.domain.ports.daemon",
    "potpie_context_engine.host.daemon",
    "potpie_context_engine.host.daemon_client",
    "potpie_context_engine.host.daemon_main",
    "potpie_context_engine.host.daemon_rpc",
    "potpie_context_engine.host.daemon_runtime",
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


def _dependency_names(metadata: dict) -> set[str]:
    return {
        re.split(r"[\s<>=!~;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in metadata["project"]["dependencies"]
    }


def test_legacy_daemon_namespaces_are_not_imported() -> None:
    offenders = {
        (path.relative_to(ROOT).as_posix(), namespace)
        for search_root in (ROOT / "potpie", ROOT / "tests")
        for path in search_root.rglob("*.py")
        for namespace in LEGACY_DAEMON_NAMESPACES
        if _imports_namespace(path, namespace)
    }
    assert offenders == set()


def test_temporary_engine_to_daemon_imports_are_explicitly_bounded() -> None:
    importers = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for path in ENGINE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(ENGINE_ROOT).parts
        and _imports_namespace(path, "potpie.daemon")
    }
    assert importers == EXPECTED_ENGINE_DAEMON_IMPORTERS


def test_root_daemon_entrypoint_targets_relocated_package() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"]["potpie-daemon"] == (
        "potpie.daemon.__main__:main"
    )


def test_mcp_process_and_dependency_are_removed() -> None:
    root_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    engine_metadata = tomllib.loads(
        (ENGINE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "potpie-mcp" not in root_metadata["project"]["scripts"]
    assert "mcp" not in _dependency_names(root_metadata)
    assert "mcp" not in _dependency_names(engine_metadata)
    assert not (
        ENGINE_ROOT / "src" / "potpie_context_engine" / "adapters" / "inbound" / "mcp"
    ).exists()
