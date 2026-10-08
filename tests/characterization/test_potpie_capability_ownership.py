"""Permanent ownership and packaging locks for root Potpie capabilities."""

# ruff: noqa: S101 - pytest architecture tests use assertions intentionally.

from __future__ import annotations

import ast
import tomllib
from collections import defaultdict
from pathlib import Path

from potpie.runtime.local_engine import LocalEngineServices
from potpie.runtime.root_services import RootRuntimeServices

ROOT = Path(__file__).resolve().parents[2]
THIS_TEST = Path(__file__).resolve()
FORBIDDEN_NAMESPACE = "potpie." + "product"

ROOT_SOURCE_PATHS = (
    ROOT / "potpie" / "agent_context.py",
    ROOT / "potpie" / "auth",
    ROOT / "potpie" / "cli",
    ROOT / "potpie" / "config",
    ROOT / "potpie" / "daemon",
    ROOT / "potpie" / "pots",
    ROOT / "potpie" / "runtime",
    ROOT / "potpie" / "setup",
    ROOT / "potpie" / "skills",
)

EXPECTED_CAPABILITY_SYMBOLS = {
    "potpie/agent_context.py": {"AgentContextService"},
    "potpie/auth/ports/identity.py": {"AuthIdentity", "AuthService"},
    "potpie/auth/adapters/local_identity.py": {"LocalAuthService"},
    "potpie/config/contracts.py": {"ConfigService"},
    "potpie/config/local.py": {"LocalConfigService"},
    "potpie/config/local_paths.py": {"default_home"},
    "potpie/pots/contracts.py": {
        "PotAggregateStatus",
        "PotInfo",
        "PotManagementService",
        "SourceInfo",
    },
    "potpie/pots/local_service.py": {"LocalPotManagementService"},
    "potpie/pots/local_store.py": {"LocalPotStore"},
    "potpie/setup/contracts.py": {
        "NoOpSetupObserver",
        "SetupObserver",
        "SetupOrchestrator",
    },
    "potpie/setup/installation.py": {"Installer"},
    "potpie/setup/local_installer.py": {"LocalInstaller"},
    "potpie/setup/state.py": {"MigrationPort", "StateStorePort"},
    "potpie/setup/flat_file_state.py": {
        "FlatFileMigrator",
        "FlatFileStateStore",
    },
    "potpie/setup/orchestrator.py": {"DefaultSetupOrchestrator"},
    "potpie/skills/contracts.py": {
        "AgentTargetPort",
        "SkillInfo",
        "SkillManager",
        "SkillOperationResult",
        "SkillStatus",
    },
    "potpie/skills/catalog.py": {"catalog_by_id", "load_bundle_skills"},
    "potpie/skills/installer.py": {
        "InstallResult",
        "install_agent_bundle",
        "validate_packaged_skill_command_snippets",
    },
    "potpie/skills/manager.py": {"DefaultSkillManager"},
    "potpie/skills/targets.py": {
        "ClaudeAgentTarget",
        "CodexAgentTarget",
        "CursorAgentTarget",
        "FileBackedAgentTarget",
        "OpenCodeAgentTarget",
        "ProjectAgentTarget",
    },
}

DOCUMENTATION_ONLY_INITIALIZERS = (
    "potpie/config/__init__.py",
    "potpie/pots/__init__.py",
    "potpie/setup/__init__.py",
    "potpie/skills/__init__.py",
)

FORBIDDEN_SOURCE_UMBRELLAS = (
    "app",
    "application",
    "capabilities",
    "control_plane",
    "product",
)

CONCRETE_ASSEMBLY_TYPES = frozenset(
    {
        "AgentContextService",
        "ClaudeAgentTarget",
        "CodexAgentTarget",
        "CursorAgentTarget",
        "DefaultSetupOrchestrator",
        "DefaultSkillManager",
        "FlatFileMigrator",
        "FlatFileStateStore",
        "LocalAuthService",
        "LocalConfigService",
        "LocalInstaller",
        "LocalPotManagementService",
        "LocalPotStore",
        "OpenCodeAgentTarget",
    }
)

EXPECTED_SDIST_CAPABILITIES = {
    "/potpie/agent_context.py",
    "/potpie/auth",
    "/potpie/cli",
    "/potpie/config",
    "/potpie/daemon",
    "/potpie/pots",
    "/potpie/runtime",
    "/potpie/setup",
    "/potpie/skills",
}


def _root_python_files() -> list[Path]:
    files: list[Path] = []
    for search_path in ROOT_SOURCE_PATHS:
        if search_path.is_file():
            files.append(search_path)
        elif search_path.is_dir():
            files.extend(search_path.rglob("*.py"))
    return files


def _imports_namespace(path: Path, namespace: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == namespace or module.startswith(f"{namespace}."):
                return True
        if isinstance(node, ast.Import) and any(
            alias.name == namespace or alias.name.startswith(f"{namespace}.")
            for alias in node.names
        ):
            return True
    return False


def _definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _sys_modules_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if not (
        isinstance(value, ast.Attribute)
        and value.attr == "modules"
        and isinstance(value.value, ast.Name)
        and value.value.id == "sys"
    ):
        return None
    key = node.slice
    return (
        key.value
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
        else None
    )


def _writes_forbidden_module_alias(node: ast.AST) -> bool:
    targets: tuple[ast.AST, ...] = ()
    if isinstance(node, ast.Assign):
        targets = tuple(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    elif isinstance(node, ast.AugAssign):
        targets = (node.target,)
    return any(
        (key := _sys_modules_key(target)) is not None
        and (key == FORBIDDEN_NAMESPACE or key.startswith(f"{FORBIDDEN_NAMESPACE}."))
        for target in targets
    )


def test_product_namespace_has_no_source_imports_aliases_or_reexports() -> None:
    product_path = ROOT / "potpie" / "product"
    remaining_source = (
        {path.relative_to(ROOT).as_posix() for path in product_path.rglob("*.py")}
        if product_path.exists()
        else set()
    )
    importers = {
        path.relative_to(ROOT).as_posix()
        for search_root in (ROOT / "potpie", ROOT / "tests")
        for path in search_root.rglob("*.py")
        if path != THIS_TEST and _imports_namespace(path, FORBIDDEN_NAMESPACE)
    }
    alias_writers = {
        path.relative_to(ROOT).as_posix()
        for path in _root_python_files()
        if any(
            _writes_forbidden_module_alias(node)
            for node in ast.walk(
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            )
        )
    }
    namespace_literals = {
        path.relative_to(ROOT).as_posix()
        for path in _root_python_files()
        if any(
            isinstance(node, ast.Constant) and node.value == FORBIDDEN_NAMESPACE
            for node in ast.walk(
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            )
        )
    }

    assert remaining_source == set()
    assert importers == set()
    assert alias_writers == set()
    assert namespace_literals == set()


def test_no_replacement_source_umbrella_exists() -> None:
    offenders = {
        path.relative_to(ROOT).as_posix()
        for name in FORBIDDEN_SOURCE_UMBRELLAS
        for path in (ROOT / "potpie" / name).rglob("*.py")
        if (ROOT / "potpie" / name).exists()
    }

    assert offenders == set()


def test_capability_modules_exist_and_own_their_expected_symbols() -> None:
    actual = {
        relative: _definitions(ROOT / relative)
        for relative in EXPECTED_CAPABILITY_SYMBOLS
    }

    assert all((ROOT / relative).is_file() for relative in EXPECTED_CAPABILITY_SYMBOLS)
    for relative, expected in EXPECTED_CAPABILITY_SYMBOLS.items():
        assert expected <= actual[relative], (
            f"{relative} does not own {expected - actual[relative]}"
        )


def test_capability_initializers_are_documentation_only() -> None:
    for relative in DOCUMENTATION_ONLY_INITIALIZERS:
        tree = ast.parse(
            (ROOT / relative).read_text(encoding="utf-8"), filename=relative
        )
        assert len(tree.body) == 1, relative
        statement = tree.body[0]
        assert isinstance(statement, ast.Expr), relative
        assert isinstance(statement.value, ast.Constant), relative
        assert isinstance(statement.value.value, str), relative


def test_runtime_composition_is_the_only_concrete_cross_capability_assembly() -> None:
    callers: dict[str, set[str]] = defaultdict(set)
    for path in _root_python_files():
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name in CONCRETE_ASSEMBLY_TYPES:
                callers[name].add(relative)

    assert callers == {
        name: {"potpie/runtime/composition.py"} for name in CONCRETE_ASSEMBLY_TYPES
    }


def test_root_and_engine_service_groups_remain_separated() -> None:
    root_fields = set(RootRuntimeServices.__dataclass_fields__)
    engine_fields = set(LocalEngineServices.__dataclass_fields__)

    assert root_fields == {
        "auth",
        "backend",
        "config",
        "daemon",
        "installer",
        "ledger",
        "pots",
        "profile",
        "setup",
        "skills",
    }
    assert engine_fields == {
        "agent_context",
        "backend",
        "graph",
        "graph_workbench",
        "ingestion",
        "ingestion_events",
        "nudge",
        "pots",
        # Document chunk payload plane (potpie document/resource commands) —
        # engine-owned: it writes through the graph service and feeds DocsReader.
        "resources",
    }
    assert root_fields.isdisjoint(
        {"agent_context", "graph", "graph_workbench", "nudge"}
    )
    assert engine_fields.isdisjoint(
        {"auth", "config", "daemon", "installer", "ledger", "setup", "skills"}
    )


def test_root_distribution_metadata_owns_capability_packages() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build = metadata["tool"]["hatch"]["build"]
    wheel = build["targets"]["wheel"]
    sdist_includes = set(build["targets"]["sdist"]["include"])

    assert wheel["packages"] == ["potpie"]
    assert EXPECTED_SDIST_CAPABILITIES <= sdist_includes
    assert not any("product" in entry.split("/") for entry in sdist_includes)
    assert metadata["project"]["scripts"] == {
        "potpie": "potpie.cli.main:main",
        "potpie-daemon": "potpie.daemon.__main__:main",
    }


def test_independent_package_installation_lane_remains_enforced() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_context_package_isolation.py").read_text(
        encoding="utf-8"
    )

    assert (
        "uv build --project potpie/context-engine --out-dir dist/context-engine"
        in workflow
    )
    assert "Verify isolated Context Engine installation" in workflow
    assert "Verify isolated root installation" in workflow
    assert "scripts/verify_context_package_isolation.py --expect-root" in workflow
    assert 'find_spec("potpie_context_engine")' in verifier
    assert 'find_spec("potpie")' in verifier
    assert "if args.expect_root:" in verifier
