"""Exact temporary caller allowlists for the Context Runtime migration.

These observations make existing legacy adoption visible and prevent new callers.
They shrink during the migration and are deleted with the final compatibility
seams; none of the listed modules is approved as target architecture.
"""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import ast
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
THIS_TEST = Path(__file__).resolve().relative_to(ROOT).as_posix()

EXPECTED_LEGACY_PATHS: dict[str, frozenset[str]] = {
    "context_core_importers": frozenset({
        "potpie/cli/commands/cloud.py",
        "potpie/cli/read_presenter.py",
        "potpie/cli/telemetry/onboarding_events.py",
        "potpie/cli/ui/setup_ux.py",
        "potpie/daemon/client.py",
        "potpie/daemon/main.py",
        "tests/integration/test_context_runtime_baseline.py",
        "tests/unit/test_agent_templates_v15.py",
        "tests/unit/test_cli_ergonomics.py",
        "tests/unit/test_daemon_rpc.py",
        "tests/unit/test_graph_cli_contract.py",
        "tests/unit/test_onboarding_analytics.py",
        "tests/unit/test_read_presenter.py",
        "tests/unit/test_repo_baseline_skill.py",
        "tests/unit/test_sentry_cli.py",
        "tests/unit/test_sentry_daemon.py",
        "tests/unit/test_setup_live_ux.py",
        "tests/unit/test_ui_router.py",
    }),
    "context_core_dependencies": frozenset({
        "pyproject.toml",
    }),
    "host_shell": frozenset({
        "potpie/cli/commands/_common.py",
        "potpie/cli/commands/bootstrap.py",
        "potpie/context-engine/src/potpie_context_engine/bootstrap/host_wiring.py",
        "potpie/context-engine/src/potpie_context_engine/host/__init__.py",
        "potpie/context-engine/src/potpie_context_engine/host/shell.py",
        "potpie/context-engine/tests/conformance/test_host_shell_end_to_end.py",
        "potpie/context-engine/tests/conformance/test_local_profile_completion.py",
        "potpie/context-engine/tests/unit/test_falkordb_backend.py",
        "potpie/context-engine/tests/unit/test_observability.py",
        "potpie/context-engine/tests/unit/test_setup_defer_pot.py",
        "potpie/context-engine/tests/unit/test_setup_defer_skills.py",
        "potpie/context-engine/tests/unit/test_skill_manager_global_targets.py",
        "potpie/daemon/main.py",
        "potpie/daemon/runtime/__main__.py",
        "tests/unit/test_onboarding_analytics.py",
        "tests/unit/test_setup_first_pot.py",
    }),
    "cli_host_acquisition": frozenset({
        "potpie/cli/commands/_common.py",
        "potpie/cli/commands/bootstrap.py",
        "tests/integration/test_context_runtime_baseline.py",
        "tests/unit/test_audit8_audit22.py",
        "tests/unit/test_cli_bootstrap_status.py",
        "tests/unit/test_cli_daemon_service.py",
        "tests/unit/test_cli_ergonomics.py",
        "tests/unit/test_empty_pot_guidance.py",
        "tests/unit/test_graph_cli_contract.py",
        "tests/unit/test_pot_create_repo.py",
        "tests/unit/test_sentry_daemon.py",
        "tests/unit/test_setup_first_pot.py",
        "tests/unit/test_skills_cli.py",
        "tests/unit/test_source_cli_contract.py",
    }),
    "remote_host": frozenset({
        "potpie/cli/commands/_common.py",
        "potpie/daemon/client.py",
        "tests/integration/test_context_runtime_baseline.py",
    }),
    "candidate_runtime": frozenset({
        "potpie/daemon/http/errors.py",
        "potpie/daemon/http/transport.py",
        "potpie/daemon/managed_services/__init__.py",
        "potpie/daemon/managed_services/container_backend.py",
        "potpie/daemon/managed_services/external_backend.py",
        "potpie/daemon/managed_services/subprocess_backend.py",
        "potpie/daemon/ports/__init__.py",
        "potpie/daemon/ports/operations.py",
        "potpie/daemon/ports/service.py",
        "potpie/daemon/ports/shell.py",
        "potpie/daemon/runtime/__init__.py",
        "potpie/daemon/runtime/__main__.py",
        "potpie/daemon/runtime/config.py",
        "potpie/daemon/runtime/context.py",
        "potpie/daemon/runtime/health.py",
        "potpie/daemon/runtime/ipc_auth.py",
        "potpie/daemon/runtime/registry.py",
        "potpie/daemon/runtime/service_manager.py",
        "potpie/daemon/runtime/shell.py",
        "tests/conftest.py",
        "tests/integration/test_daemon_external_backend.py",
        "tests/integration/test_daemon_http_transport.py",
        "tests/integration/test_daemon_http_transport_admin.py",
        "tests/integration/test_daemon_http_transport_extra.py",
        "tests/integration/test_daemon_http_transport_socket_edges.py",
        "tests/integration/test_daemon_runtime_run.py",
        "tests/integration/test_daemon_subprocess_backend.py",
        "tests/integration/test_daemon_subprocess_backend_extra.py",
        "tests/unit/test_daemon_config.py",
        "tests/unit/test_daemon_container_backend.py",
        "tests/unit/test_daemon_service_manager.py",
    }),
    "discovery": frozenset({
        "potpie/daemon/lifecycle.py",
        "potpie/daemon/main.py",
        "potpie/daemon/process/ipc_client.py",
        "potpie/daemon/process/launcher.py",
        "potpie/daemon/process/pidfile.py",
        "potpie/daemon/runtime/__main__.py",
        "tests/integration/test_context_runtime_baseline.py",
        "tests/integration/test_daemon_ipc_client.py",
        "tests/unit/test_daemon_launcher.py",
        "tests/unit/test_daemon_pidfile.py",
        "tests/unit/test_daemon_seam.py",
    }),
    "reflective_rpc": frozenset({
        "potpie/daemon/client.py",
        "potpie/daemon/main.py",
    }),
}

SUPPORTED_PRODUCT_ENTRYPOINTS = frozenset(
    {
        "pyproject.toml",
        "potpie/cli/commands/_common.py",
        "potpie/daemon/main.py",
    }
)
POTENTIALLY_EXTERNAL_UNCONTRACTED = frozenset(
    {
        "potpie/context-engine/pyproject.toml",
        "potpie/context-engine/src/potpie_context_engine/__init__.py",
        "potpie/context-engine/src/potpie_context_engine/api.py",
        "potpie/context-engine/src/potpie_context_engine/bootstrap/host_wiring.py",
        "potpie/context-engine/src/potpie_context_engine/host/__init__.py",
        "potpie/daemon/client.py",
    }
)


class CallerClass(StrEnum):
    SUPPORTED_PRODUCT_ENTRYPOINT = "supported_product_entrypoint"
    REPOSITORY_INTERNAL = "repository_internal"
    POTENTIALLY_EXTERNAL_UNCONTRACTED = "potentially_external_uncontracted"


def _python_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for search_root in (ROOT / "potpie", ROOT / "tests"):
        for path in search_root.rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative != THIS_TEST:
                files.append((path, relative))
    return files


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _contains_literal(node: ast.AST | None, values: set[str]) -> bool:
    if node is None:
        return False
    return any(
        isinstance(child, ast.Constant) and child.value in values
        for child in ast.walk(node)
    )


def _uses_discovery_path(node: ast.AST) -> bool:
    discovery_names = {"discovery.json", "daemon.json"}
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and _contains_literal(node.right, discovery_names)
    ):
        return True
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        return any(
            isinstance(target, ast.Name) and target.id == "_DISCOVERY_FILES"
            for target in targets
        ) and _contains_literal(value, discovery_names)
    return False


def _uses_reflective_route(node: ast.AST) -> bool:
    routes = {"/rpc", "/attr"}
    if isinstance(node, ast.Call) and _call_name(node) in {"get", "post"}:
        return bool(node.args) and _contains_literal(node.args[0], routes)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        return any(
            isinstance(target, ast.Name) and target.id == "url" for target in targets
        ) and _contains_literal(value, routes)
    return False


@lru_cache(maxsize=1)
def _legacy_paths() -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = {
        name: set() for name in EXPECTED_LEGACY_PATHS
    }

    for path, relative in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imports: list[str] = []
        imported_names: list[tuple[str, set[str]]] = []

        for node in ast.walk(tree):
            if _uses_discovery_path(node):
                groups["discovery"].add(relative)
            if _uses_reflective_route(node):
                groups["reflective_rpc"].add(relative)
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
                imported_names.append(
                    (module, {alias.name for alias in node.names})
                )
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                if _call_name(node) in {"get_host", "set_host"}:
                    groups["cli_host_acquisition"].add(relative)
                if _call_name(node) in {
                    "write_discovery",
                    "read_discovery",
                    "load_discovery",
                }:
                    groups["discovery"].add(relative)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in {
                    "write_discovery",
                    "read_discovery",
                    "load_discovery",
                }:
                    groups["discovery"].add(relative)

        if not relative.startswith("potpie/context-core/") and any(
            module == "potpie_context_core"
            or module.startswith("potpie_context_core.")
            for module in imports
        ):
            groups["context_core_importers"].add(relative)

        for module, names in imported_names:
            if module in {
                "potpie_context_engine.host",
                "potpie_context_engine.host.shell",
                "potpie_context_engine.bootstrap",
                "potpie_context_engine.bootstrap.host_wiring",
            } and names & {"HostShell", "build_host_shell"}:
                groups["host_shell"].add(relative)
            if module == "potpie.cli.commands._common" and names & {
                "get_host",
                "set_host",
            }:
                groups["cli_host_acquisition"].add(relative)
            if module == "potpie.daemon.client":
                groups["remote_host"].add(relative)

        if any(
            module == "potpie.daemon.client"
            or module.startswith("potpie.daemon.client.")
            for module in imports
        ):
            groups["remote_host"].add(relative)

        candidate_prefixes = (
            "potpie/daemon/runtime/",
            "potpie/daemon/ports/",
            "potpie/daemon/managed_services/",
        )
        if relative.startswith(candidate_prefixes) or relative in {
            "potpie/daemon/http/errors.py",
            "potpie/daemon/http/transport.py",
        }:
            groups["candidate_runtime"].add(relative)
        if any(
            module.startswith(
                (
                    "potpie.daemon.runtime",
                    "potpie.daemon.ports",
                    "potpie.daemon.managed_services",
                )
            )
            or module
            in {"potpie.daemon.http.errors", "potpie.daemon.http.transport"}
            for module in imports
        ):
            groups["candidate_runtime"].add(relative)

    groups["host_shell"].update(
        {
            "potpie/context-engine/src/potpie_context_engine/host/shell.py",
            "potpie/context-engine/src/potpie_context_engine/bootstrap/host_wiring.py",
        }
    )
    groups["cli_host_acquisition"].add("potpie/cli/commands/_common.py")
    groups["remote_host"].add("potpie/daemon/client.py")

    for metadata_path in (
        ROOT / "pyproject.toml",
        ROOT / "potpie" / "context-engine" / "pyproject.toml",
    ):
        metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
        dependency_names = {
            re.split(r"[\s<>=!~;\[]", dependency, maxsplit=1)[0].lower()
            for dependency in metadata["project"]["dependencies"]
        }
        if "potpie-context-core" in dependency_names:
            groups["context_core_dependencies"].add(
                metadata_path.relative_to(ROOT).as_posix()
            )

    return {name: frozenset(paths) for name, paths in groups.items()}


def _classification(path: str) -> CallerClass:
    if path in SUPPORTED_PRODUCT_ENTRYPOINTS:
        return CallerClass.SUPPORTED_PRODUCT_ENTRYPOINT
    if path in POTENTIALLY_EXTERNAL_UNCONTRACTED:
        return CallerClass.POTENTIALLY_EXTERNAL_UNCONTRACTED
    return CallerClass.REPOSITORY_INTERNAL


@pytest.mark.parametrize("surface", sorted(EXPECTED_LEGACY_PATHS))
def test_legacy_surface_paths_match_temporary_allowlist(surface: str) -> None:
    assert _legacy_paths()[surface] == EXPECTED_LEGACY_PATHS[surface]


def test_every_allowlisted_path_has_a_migration_classification() -> None:
    classifications = {
        path: _classification(path)
        for paths in EXPECTED_LEGACY_PATHS.values()
        for path in paths
    }

    assert classifications
    assert set(classifications.values()) == set(CallerClass)
