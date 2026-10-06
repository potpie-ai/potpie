"""The base ``potpie`` install is the local product without the ML stack.

A bare ``pip install potpie`` gets the CLI, the daemon and the local FalkorDB
backend. Embeddings (sentence-transformers, and through it torch), ingestion
clients, telemetry and the CLI logo are extras. Every CI job and the dev venv
install all extras, so a regression in the base set is invisible everywhere
else until it reaches a user's first run, or trips an installer that refuses
torch. These tests read the declared metadata and the lockfile, which is what
an installer resolves from.
"""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = ROOT / "pyproject.toml"
ENGINE_PYPROJECT = ROOT / "potpie" / "context-engine" / "pyproject.toml"
LOCKFILE = ROOT / "uv.lock"

ENGINE = "potpie-context-engine"

#: Extras that installers already pass, plus the heavy parts of the product.
EXPECTED_EXTRAS = {
    "daemon",
    "local",
    "auth",
    "embeddings",
    "ingest",
    "telemetry",
    "ui",
    "all",
}

#: Installs that must stay free of the ML stack: the base, and the extra sets
#: the VS Code extension passes (Windows, then every other OS).
LEAN_INSTALLS = [
    (),
    ("daemon", "auth", "telemetry"),
    ("local", "auth", "telemetry"),
]

#: Never reachable from a lean install. `nvidia-*` CUDA wheels come with torch.
ML_STACK = {"torch", "sentence-transformers", "transformers", "triton"}

#: The embedded FalkorDBLite store and the pins that exist only for it.
EMBEDDED_STORE = {"falkordblite", "redis", "hiredis"}


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _requirements(raw: Iterable[str]) -> list[Requirement]:
    return [Requirement(item) for item in raw]


# --- the lockfile: what a base install actually resolves ---------------------


def _lock_packages() -> dict[str, list[dict]]:
    lock = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    return packages


def _resolved_closure(extras: Iterable[str]) -> set[str]:
    """Every package a ``potpie[extras]`` install can pull, on any platform.

    Markers are ignored on purpose: the union over platforms is the strictest
    reading of "this install never reaches torch".
    """
    packages = _lock_packages()
    seen: set[tuple[str, frozenset[str]]] = set()
    names: set[str] = set()
    pending = [("potpie", frozenset(extras))]
    while pending:
        name, wanted = pending.pop()
        if (name, wanted) in seen:
            continue
        seen.add((name, wanted))
        names.add(name)
        for package in packages.get(name, []):
            edges = list(package.get("dependencies", []))
            optional = package.get("optional-dependencies", {})
            for extra in wanted:
                edges.extend(optional.get(extra, []))
            for edge in edges:
                pending.append((edge["name"], frozenset(edge.get("extra", []))))
    return names


@pytest.mark.parametrize("extras", LEAN_INSTALLS, ids=lambda e: ",".join(e) or "base")
def test_lean_installs_resolve_no_ml_stack(extras: tuple[str, ...]) -> None:
    closure = _resolved_closure(extras)

    assert closure & ML_STACK == set()
    assert not any(name.startswith("nvidia-") for name in closure)
    # The local product is still there.
    assert {"potpie-context-engine", "falkordb", "fastapi", "uvicorn"} <= closure


def test_the_embeddings_extra_is_where_the_ml_stack_lives() -> None:
    """Guards the walker above: it must be able to find torch at all."""
    closure = _resolved_closure(("embeddings",))

    assert {"sentence-transformers", "torch"} <= closure


def test_all_restores_every_engine_adapter() -> None:
    closure = _resolved_closure(("all",))

    for package in ("sentence-transformers", "neo4j", "pygithub", "sentry-sdk"):
        assert package in closure, package


# --- the declared metadata ----------------------------------------------------


def test_named_extras_exist_and_all_covers_them() -> None:
    project = _project(ROOT_PYPROJECT)
    extras = project["optional-dependencies"]

    assert set(extras) == EXPECTED_EXTRAS
    umbrella = [
        req
        for req in _requirements(extras["all"])
        if canonicalize_name(req.name) == "potpie"
    ]
    assert len(umbrella) == 1
    assert set(umbrella[0].extras) == EXPECTED_EXTRAS - {"all"}


def test_engine_pin_is_the_single_unconditional_exact_pin() -> None:
    """The release validator reads exactly this requirement."""
    project = _project(ROOT_PYPROJECT)
    engine_version = _project(ENGINE_PYPROJECT)["version"]
    pins = [
        req
        for req in _requirements(project["dependencies"])
        if canonicalize_name(req.name) == ENGINE
    ]

    assert len(pins) == 1
    (pin,) = pins
    assert pin.marker is None
    assert str(pin.specifier) == f"=={engine_version}"
    assert {"http", "local"} <= set(pin.extras)
    assert not {"all", "embeddings"} & set(pin.extras)


def test_base_declares_no_heavy_or_optional_package() -> None:
    names = {
        canonicalize_name(req.name)
        for req in _requirements(_project(ROOT_PYPROJECT)["dependencies"])
    }

    for absent in ("sentence-transformers", "torch", "keyring", "pillow", "sentry-sdk"):
        assert absent not in names, f"{absent} is a base dependency again"


def test_typer_is_not_capped() -> None:
    """The CLI imports no private typer module, so no upper bound is needed."""
    (typer,) = [
        req
        for req in _requirements(_project(ROOT_PYPROJECT)["dependencies"])
        if canonicalize_name(req.name) == "typer"
    ]

    assert not any(spec.operator in {"<", "<=", "=="} for spec in typer.specifier)


# --- platform markers ---------------------------------------------------------


def _environment(sys_platform: str) -> dict[str, str]:
    return {
        "python_version": "3.12",
        "python_full_version": "3.12.10",
        "sys_platform": sys_platform,
        "platform_system": {"win32": "Windows", "darwin": "Darwin"}.get(
            sys_platform, "Linux"
        ),
        "os_name": "nt" if sys_platform == "win32" else "posix",
        "extra": "",
    }


def _declared_install(extras: Iterable[str], sys_platform: str) -> set[str]:
    """Package names one level through the engine's extras, markers applied."""
    root = _project(ROOT_PYPROJECT)
    engine = _project(ENGINE_PYPROJECT)
    environment = _environment(sys_platform)
    wanted = set(extras)
    selected = list(_requirements(root["dependencies"]))
    for extra in wanted:
        selected.extend(_requirements(root["optional-dependencies"][extra]))

    names: set[str] = set()
    engine_extras: set[str] = set()
    for req in selected:
        if req.marker is not None and not req.marker.evaluate(environment):
            continue
        name = canonicalize_name(req.name)
        if name == "potpie":
            wanted |= set(req.extras)
            continue
        names.add(name)
        if name == ENGINE:
            engine_extras |= set(req.extras)
    if "all" in engine_extras:
        engine_extras |= {
            extra
            for extra in engine["optional-dependencies"]
            if extra not in {"all", "benchmarks", "dev"}
        }
    for extra in engine_extras:
        for req in _requirements(engine["optional-dependencies"][extra]):
            if req.marker is None or req.marker.evaluate(environment):
                names.add(canonicalize_name(req.name))
    return names


def test_root_never_redeclares_the_embedded_store() -> None:
    """The FalkorDBLite markers live once, on the engine's `local` extra."""
    project = _project(ROOT_PYPROJECT)
    declared = list(project["dependencies"])
    for requirements in project["optional-dependencies"].values():
        declared.extend(requirements)

    names = {canonicalize_name(req.name) for req in _requirements(declared)}

    assert names & EMBEDDED_STORE == set()


def test_windows_install_skips_the_embedded_store() -> None:
    installed = _declared_install(("daemon", "auth", "telemetry"), "win32")

    assert installed & EMBEDDED_STORE == set()
    assert "falkordb" in installed


@pytest.mark.parametrize("sys_platform", ["linux", "darwin"])
def test_other_platforms_get_the_embedded_store_from_the_base(
    sys_platform: str,
) -> None:
    installed = _declared_install((), sys_platform)

    assert EMBEDDED_STORE | {"falkordb"} <= installed
    assert "sentence-transformers" not in installed


# --- the wheel ----------------------------------------------------------------


def test_root_wheel_ships_built_ui_but_no_tests_or_ui_sources() -> None:
    wheel = tomllib.loads(ROOT_PYPROJECT.read_text(encoding="utf-8"))["tool"]["hatch"][
        "build"
    ]["targets"]["wheel"]

    assert "potpie/daemon/http/ui/frontend/dist/**/*" in wheel["include"]
    assert "**/tests/**" in wheel["exclude"]
    assert "/potpie/daemon/http/ui/frontend/*" in wheel["exclude"]
    assert "!/potpie/daemon/http/ui/frontend/dist" in wheel["exclude"]
    # Without this, `!` lines in the root .gitignore re-include files under
    # excluded directories (the context-engine test fixtures did ship).
    assert wheel["skip-excluded-dirs"] is True


def test_engine_wheel_excludes_tests() -> None:
    wheel = tomllib.loads(ENGINE_PYPROJECT.read_text(encoding="utf-8"))["tool"][
        "hatch"
    ]["build"]["targets"]["wheel"]

    assert "**/tests/**" in wheel["exclude"]
