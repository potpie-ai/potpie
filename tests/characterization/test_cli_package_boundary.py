"""Static ownership locks for the Phase 1 CLI relocation."""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "potpie" / "context-engine"

EXPECTED_ENGINE_CLI_IMPORTERS: set[str] = set()

EXPECTED_ENGINE_ROOT_IMPORTERS: set[str] = set()


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
    legacy_namespace = "potpie_context_engine." + ".".join(
        ("adapters", "inbound", "cli")
    )
    offenders = {
        path.relative_to(ROOT).as_posix()
        for search_root in (ROOT / "potpie", ROOT / "tests")
        for path in search_root.rglob("*.py")
        if _imports_namespace(path, legacy_namespace)
    }
    assert offenders == set()


def test_engine_does_not_import_cli() -> None:
    importers = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for path in ENGINE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(ENGINE_ROOT).parts
        and _references_namespace(path, "potpie.cli")
    }
    assert importers == EXPECTED_ENGINE_CLI_IMPORTERS


def test_engine_does_not_import_root_potpie() -> None:
    importers = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for path in ENGINE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(ENGINE_ROOT).parts
        and _imports_namespace(path, "potpie")
    }
    assert importers == EXPECTED_ENGINE_ROOT_IMPORTERS


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


def test_product_authentication_is_owned_by_root_potpie() -> None:
    legacy_auth_directories = (
        ENGINE_ROOT
        / "src"
        / "potpie_context_engine"
        / "adapters"
        / "outbound"
        / "cli_auth",
        ENGINE_ROOT / "src" / "potpie_context_engine" / "domain" / "ports" / "cli_auth",
    )
    assert all(
        not any(path.rglob("*.py")) for path in legacy_auth_directories if path.is_dir()
    )
    assert not (
        ENGINE_ROOT
        / "src"
        / "potpie_context_engine"
        / "bootstrap"
        / "cli_auth_wiring.py"
    ).exists()

    engine_auth_importers = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for path in ENGINE_ROOT.rglob("*.py")
        if _imports_namespace(path, "potpie.auth")
    }
    assert engine_auth_importers == set()

    engine_metadata = tomllib.loads(
        (ENGINE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "cli-auth" not in engine_metadata["project"]["optional-dependencies"]


def test_product_lifecycle_sources_are_owned_by_root_potpie() -> None:
    engine_source = ENGINE_ROOT / "src" / "potpie_context_engine"
    removed_files = (
        engine_source / "domain" / "ports" / "install.py",
        engine_source / "application" / "services" / "agent_context.py",
        engine_source / "application" / "services" / "auth_service.py",
        engine_source / "application" / "services" / "config_service.py",
        engine_source / "application" / "services" / "pot_management.py",
        engine_source / "application" / "services" / "setup_orchestrator.py",
        engine_source / "application" / "services" / "skill_manager.py",
    )
    assert all(not path.exists() for path in removed_files)

    removed_directories = (
        engine_source / "domain" / "ports" / "services",
        engine_source / "adapters" / "outbound" / "install",
        engine_source / "adapters" / "outbound" / "pots",
        engine_source / "adapters" / "outbound" / "skills",
    )
    assert all(
        not any(path.rglob("*.py")) for path in removed_directories if path.is_dir()
    )


def test_engine_adapters_and_ports_do_not_depend_on_product_setup_types() -> None:
    engine_source = ENGINE_ROOT / "src" / "potpie_context_engine"
    offenders = {
        path.relative_to(ENGINE_ROOT).as_posix()
        for search_root in (
            engine_source / "adapters",
            engine_source / "domain" / "ports",
        )
        for path in search_root.rglob("*.py")
        if _imports_namespace(path, "potpie_context_engine.core.lifecycle")
    }
    assert offenders == set()


def test_product_runtime_configuration_is_owned_by_root_potpie() -> None:
    engine_bootstrap = ENGINE_ROOT / "src" / "potpie_context_engine" / "bootstrap"
    assert not (engine_bootstrap / "runtime_settings.py").exists()
    assert not (engine_bootstrap / "env_bootstrap.py").exists()
    assert (ROOT / "potpie" / "runtime" / "settings.py").is_file()
    assert (ROOT / "potpie" / "runtime" / "env_bootstrap.py").is_file()

    root_runtime_importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "potpie").rglob("*.py")
        if _references_namespace(
            path, "potpie_context_engine.bootstrap.runtime_settings"
        )
        or _references_namespace(path, "potpie_context_engine.bootstrap.env_bootstrap")
    }
    assert root_runtime_importers == set()


def test_product_build_defaults_are_not_packaged_by_context_engine() -> None:
    engine_config = (ENGINE_ROOT / "sentry_defaults_hook.py").read_text(
        encoding="utf-8"
    )
    for product_field in (
        "posthog_api_key",
        "posthog_host",
        "linear_client_id",
        "github_client_id",
    ):
        assert product_field not in engine_config

    assert (ROOT / "build_config_values.py").is_file()
    assert (ROOT / "distribution_defaults_hook.py").is_file()


def test_root_console_script_targets_relocated_cli() -> None:
    root_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert root_metadata["project"]["scripts"]["potpie"] == "potpie.cli.main:main"
