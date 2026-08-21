"""Permanent negative locks for the final Context Runtime architecture."""

# ruff: noqa: S101 - pytest architecture tests use assertions intentionally.

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
import tomllib


ROOT = Path(__file__).resolve().parents[2]
THIS_TEST = Path(__file__).resolve()

FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "DaemonRpcClient",
        "HostShell",
        "LegacyEngineClientAdapter",
        "LegacyRequestInvoker",
        "RemoteHostShell",
        "RemoteSurface",
        "build_host_shell",
        "build_legacy_engine_client",
        "get_host",
        "set_host",
    }
)

REMOVED_PATHS = (
    ROOT / "potpie" / "context-core",
    ROOT
    / "potpie"
    / "context-engine"
    / "src"
    / "potpie_context_engine"
    / "bootstrap"
    / "host_wiring.py",
    ROOT / "potpie" / "context-engine" / "src" / "potpie_context_engine" / "host",
    ROOT / "potpie" / "runtime" / "legacy_host_adapter.py",
)

CANONICAL_RUNTIME_DEFINITIONS = {
    "CanonicalDaemonRuntime": "potpie/runtime/server.py",
    "DaemonController": "potpie/runtime/controller.py",
    "DaemonEngineClient": "potpie/runtime/clients.py",
    "write_daemon_discovery": "potpie/daemon/discovery.py",
}


def _python_files() -> list[Path]:
    return [
        path
        for search_root in (ROOT / "potpie", ROOT / "tests", ROOT / "scripts")
        for path in search_root.rglob("*.py")
        if path != THIS_TEST
    ]


def _imports_module(node: ast.AST, module_name: str) -> bool:
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        return module == module_name or module.startswith(f"{module_name}.")
    if isinstance(node, ast.Import):
        return any(
            alias.name == module_name or alias.name.startswith(f"{module_name}.")
            for alias in node.names
        )
    return False


def _uses_forbidden_identifier(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in FORBIDDEN_IDENTIFIERS
    if isinstance(node, ast.Attribute):
        return node.attr in FORBIDDEN_IDENTIFIERS
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name in FORBIDDEN_IDENTIFIERS
    if isinstance(node, ast.ImportFrom):
        return any(alias.name in FORBIDDEN_IDENTIFIERS for alias in node.names)
    return False


def _defines_reflective_route(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or not decorator.args:
            continue
        first = decorator.args[0]
        if isinstance(first, ast.Constant) and first.value in {"/rpc", "/attr"}:
            return True
    return False


def _toml_strings(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _toml_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _toml_strings(nested)
    elif isinstance(value, str):
        yield value


def _contains_architecture_file(path: Path) -> bool:
    return any(
        item.is_file()
        and all(
            part != "__pycache__" and not part.startswith(".")
            for part in item.relative_to(path).parts
        )
        for item in path.rglob("*")
    )


def test_removed_compatibility_paths_do_not_return() -> None:
    remaining = {
        path.relative_to(ROOT).as_posix()
        for path in REMOVED_PATHS
        if path.is_file() or (path.is_dir() and _contains_architecture_file(path))
    }

    assert remaining == set()


def test_python_callers_cannot_use_removed_runtime_symbols_or_core_imports() -> None:
    forbidden_identifiers: set[str] = set()
    context_core_importers: set[str] = set()

    for path in _python_files():
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        if any(_uses_forbidden_identifier(node) for node in ast.walk(tree)):
            forbidden_identifiers.add(relative)
        if any(_imports_module(node, "potpie_context_core") for node in ast.walk(tree)):
            context_core_importers.add(relative)

    assert forbidden_identifiers == set()
    assert context_core_importers == set()


def test_engine_production_code_does_not_import_root_potpie() -> None:
    engine_source = ROOT / "potpie" / "context-engine" / "src" / "potpie_context_engine"
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in engine_source.rglob("*.py")
        if any(
            _imports_module(node, "potpie")
            for node in ast.walk(
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            )
        )
    }

    assert importers == set()


def test_reflective_routes_and_legacy_discovery_are_absent_from_production() -> None:
    reflective_routes: set[str] = set()
    legacy_discovery_mentions: set[str] = set()

    for path in (ROOT / "potpie").rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        if any(_defines_reflective_route(node) for node in ast.walk(tree)):
            reflective_routes.add(relative)
        if any(
            isinstance(node, ast.Constant) and node.value == "daemon.json"
            for node in ast.walk(tree)
        ):
            legacy_discovery_mentions.add(relative)

    assert reflective_routes == set()
    assert legacy_discovery_mentions == set()


def test_canonical_runtime_has_one_entrypoint_launcher_and_definition_set() -> None:
    definitions = {name: set() for name in CANONICAL_RUNTIME_DEFINITIONS}
    launchers: set[str] = set()

    for path in (ROOT / "potpie").rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in definitions:
                    definitions[node.name].add(relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Tuple):
                continue
            values = tuple(
                item.value if isinstance(item, ast.Constant) else None
                for item in node.elts
            )
            if values[-2:] == ("-m", "potpie.daemon"):
                launchers.add(relative)

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = metadata["project"]["scripts"]
    daemon_scripts = {
        name: target
        for name, target in scripts.items()
        if isinstance(target, str) and "potpie.daemon" in target
    }

    assert definitions == {
        name: {relative} for name, relative in CANONICAL_RUNTIME_DEFINITIONS.items()
    }
    assert daemon_scripts == {"potpie-daemon": "potpie.daemon.__main__:main"}
    assert launchers == {"potpie/daemon/lifecycle.py"}
    assert (ROOT / "potpie" / "daemon" / "__main__.py").is_file()


def test_metadata_and_ci_do_not_restore_context_core() -> None:
    offenders: set[str] = set()
    metadata_paths = [*ROOT.rglob("pyproject.toml"), ROOT / "uv.lock"]

    for path in metadata_paths:
        metadata = tomllib.loads(path.read_text(encoding="utf-8"))
        if any(
            "potpie-context-core" in value.lower()
            or "potpie/context-core" in value.lower()
            for value in _toml_strings(metadata)
        ):
            offenders.add(path.relative_to(ROOT).as_posix())

    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert offenders == set()
    assert "potpie-context-core" not in workflow
    assert "potpie/context-core" not in workflow
