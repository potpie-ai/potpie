"""Product ownership, package isolation, and runtime composition contracts."""

# ruff: noqa: S101 - pytest contracts use assertions.

from __future__ import annotations

import ast
import re
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest

from tests.cases.package_ownership import (
    CANONICAL_RUNTIME_DEFINITIONS,
    CONCRETE_ASSEMBLY_TYPES,
    DOCUMENTATION_ONLY_INITIALIZERS,
    EXPECTED_CAPABILITY_SYMBOLS,
    EXPECTED_SDIST_CAPABILITIES,
    FORBIDDEN_IDENTIFIERS,
    FORBIDDEN_SOURCE_UMBRELLAS,
    LEGACY_DAEMON_NAMESPACES,
)
from tests.support.source_tree import (
    definitions,
    has_literal,
    imports_namespace,
    nested_strings,
    python_trees,
)

pytestmark = pytest.mark.source_contract
ROOT = Path(__file__).resolve().parents[2]
ENGINE = "potpie/context-engine/src/potpie_context_engine/"
PRODUCT = (
    "potpie/agent_context.py",
    "potpie/auth/",
    "potpie/cli/",
    "potpie/config/",
    "potpie/daemon/",
    "potpie/pots/",
    "potpie/runtime/",
    "potpie/setup/",
    "potpie/skills/",
)


@pytest.fixture(scope="module")
def trees():
    return python_trees(ROOT)


@pytest.fixture(scope="module")
def metadata():
    return {
        owner: tomllib.loads((ROOT / relative).read_text())
        for owner, relative in (
            ("root", "pyproject.toml"),
            ("engine", "potpie/context-engine/pyproject.toml"),
        )
    }


@pytest.mark.parametrize(
    "namespace",
    sorted(
        LEGACY_DAEMON_NAMESPACES
        | {
            "potpie_context_engine.adapters.inbound.cli",
            "potpie_context_core",
            "potpie.product",
            "potpie_context_engine.bootstrap.runtime_settings",
            "potpie_context_engine.bootstrap.env_bootstrap",
        }
    ),
)
def test_sources_use_current_package_names(trees, namespace):
    assert not {
        path for path, tree in trees.items() if imports_namespace(tree, namespace)
    }


def test_engine_is_independent_of_product(trees, metadata):
    assert not {
        path
        for path, tree in trees.items()
        if path.startswith(ENGINE)
        and (imports_namespace(tree, "potpie") or has_literal(tree, "potpie.cli"))
    }
    assert not {
        path
        for path, tree in trees.items()
        if path.startswith(ENGINE + "adapters/")
        or path.startswith(ENGINE + "domain/ports/")
        if imports_namespace(tree, "potpie_context_engine.core.lifecycle")
    }
    deps = {
        re.split(r"[\s<>=!~;\[]", item, maxsplit=1)[0].lower()
        for item in metadata["engine"]["project"]["dependencies"]
    }
    assert "potpie" not in deps
    assert "cli-auth" not in metadata["engine"]["project"]["optional-dependencies"]
    hook = (ROOT / "potpie/context-engine/sentry_defaults_hook.py").read_text()
    assert all(
        field not in hook
        for field in (
            "posthog_api_key",
            "posthog_host",
            "linear_client_id",
            "github_client_id",
        )
    )


def test_sources_keep_product_capabilities_in_their_own_packages(trees):
    removed = ["potpie/context-core", "potpie/runtime/legacy_host_adapter.py"]
    removed += [f"potpie/{name}" for name in FORBIDDEN_SOURCE_UMBRELLAS]
    removed += [
        ENGINE + name
        for name in (
            "host",
            "bootstrap/host_wiring.py",
            "bootstrap/cli_auth_wiring.py",
            "bootstrap/runtime_settings.py",
            "bootstrap/env_bootstrap.py",
            "domain/ports/install.py",
            "adapters/outbound/cli_auth",
            "domain/ports/cli_auth",
            "domain/ports/services",
            "adapters/outbound/install",
            "adapters/outbound/pots",
            "adapters/outbound/skills",
            "adapters/inbound/mcp",
        )
    ]
    removed += [
        ENGINE + f"application/services/{name}.py"
        for name in (
            "agent_context",
            "auth_service",
            "config_service",
            "pot_management",
            "setup_orchestrator",
            "skill_manager",
        )
    ]
    assert not {
        path
        for path in trees
        if any(path == prefix or path.startswith(prefix + "/") for prefix in removed)
    }
    for relative in (
        "potpie/runtime/settings.py",
        "potpie/runtime/env_bootstrap.py",
        "scripts/build_config_values.py",
        "scripts/distribution_defaults_hook.py",
    ):
        assert (ROOT / relative).is_file(), relative


def test_runtime_callers_use_current_symbols(trees):
    offenders = set()
    for path, tree in trees.items():
        for node in ast.walk(tree):
            names = (
                (node.id,)
                if isinstance(node, ast.Name)
                else (node.attr,)
                if isinstance(node, ast.Attribute)
                else (node.name,)
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                else tuple(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom)
                else ()
            )
            if FORBIDDEN_IDENTIFIERS.intersection(names):
                offenders.add(path)
    assert not offenders


def test_product_has_no_module_aliases(trees):
    namespace = "potpie.product"
    offenders = set()
    for path, tree in trees.items():
        if not path.startswith(PRODUCT):
            continue
        if has_literal(tree, namespace):
            offenders.add(path)
        for node in ast.walk(tree):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else (node.target,)
                if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                else ()
            )
            for target in targets:
                if not isinstance(target, ast.Subscript) or not isinstance(
                    target.slice, ast.Constant
                ):
                    continue
                key = target.slice.value
                if (
                    ast.unparse(target.value) == "sys.modules"
                    and isinstance(key, str)
                    and (key == namespace or key.startswith(namespace + "."))
                ):
                    offenders.add(path)
    assert not offenders


@pytest.mark.parametrize(
    "relative,expected",
    EXPECTED_CAPABILITY_SYMBOLS.items(),
    ids=EXPECTED_CAPABILITY_SYMBOLS,
)
def test_capability_module_owns_its_symbols(trees, relative, expected):
    assert expected <= definitions(trees[relative])


@pytest.mark.parametrize("relative", DOCUMENTATION_ONLY_INITIALIZERS)
def test_capability_initializers_are_documentation_only(trees, relative):
    body = trees[relative].body
    assert len(body) == 1
    assert isinstance(body[0], ast.Expr)
    assert isinstance(body[0].value, ast.Constant)
    assert isinstance(body[0].value.value, str)


def test_runtime_composition_owns_concrete_assembly(trees):
    callers = defaultdict(set)
    for path, tree in trees.items():
        if path.startswith(PRODUCT):
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in CONCRETE_ASSEMBLY_TYPES
                ):
                    callers[node.func.id].add(path)
    assert callers == {
        name: {"potpie/runtime/composition.py"} for name in CONCRETE_ASSEMBLY_TYPES
    }


def test_service_groups_separate_product_and_engine():
    from potpie.runtime.local_engine import LocalEngineServices
    from potpie.runtime.root_services import RootRuntimeServices

    assert set(RootRuntimeServices.__dataclass_fields__) == {
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
    assert set(LocalEngineServices.__dataclass_fields__) == {
        "agent_context",
        "backend",
        "graph",
        "graph_workbench",
        "ingestion",
        "ingestion_events",
        "nudge",
        "pots",
        "resources",
    }


def test_runtime_has_one_launcher_and_explicit_routes(trees):
    owners = {name: set() for name in CANONICAL_RUNTIME_DEFINITIONS}
    launchers, forbidden_routes, discovery_literals = set(), set(), set()
    for path, tree in trees.items():
        if not path.startswith("potpie/"):
            continue
        for name in definitions(tree) & owners.keys():
            owners[name].add(path)
        if has_literal(tree, "daemon.json"):
            discovery_literals.add(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Tuple):
                values = tuple(
                    item.value if isinstance(item, ast.Constant) else None
                    for item in node.elts
                )
                if values[-2:] == ("-m", "potpie.daemon"):
                    launchers.add(path)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and decorator.args
                        and isinstance(decorator.args[0], ast.Constant)
                        and decorator.args[0].value in {"/rpc", "/attr"}
                    ):
                        forbidden_routes.add(path)
    assert owners == {
        name: {path} for name, path in CANONICAL_RUNTIME_DEFINITIONS.items()
    }
    assert launchers == {"potpie/daemon/lifecycle.py"}
    assert not forbidden_routes
    assert not discovery_literals
    assert (ROOT / "potpie/daemon/__main__.py").is_file()


def test_distribution_metadata_owns_product_packages_and_entrypoints(metadata):
    root = metadata["root"]
    build = root["tool"]["hatch"]["build"]["targets"]
    assert build["wheel"]["packages"] == ["potpie"]
    includes = set(build["sdist"]["include"])
    assert EXPECTED_SDIST_CAPABILITIES <= includes
    assert not any("product" in item.split("/") for item in includes)
    assert root["project"]["scripts"] == {
        "potpie": "potpie.cli.main:main",
        "potpie-daemon": "potpie.daemon.__main__:main",
    }
    for owner in metadata.values():
        names = {
            re.split(r"[\s<>=!~;\[]", item, maxsplit=1)[0].lower()
            for item in owner["project"]["dependencies"]
        }
        assert "mcp" not in names


def test_metadata_uses_current_context_packages():
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        *(ROOT / "potpie").rglob("pyproject.toml"),
    ]
    offenders = {
        path.relative_to(ROOT).as_posix()
        for path in paths
        if any(
            "potpie-context-core" in value.lower()
            or "potpie/context-core" in value.lower()
            for value in nested_strings(tomllib.loads(path.read_text()))
        )
    }
    assert not offenders
    workflow = (ROOT / ".github/workflows/test.yml").read_text()
    assert "potpie-context-core" not in workflow
    assert "potpie/context-core" not in workflow
