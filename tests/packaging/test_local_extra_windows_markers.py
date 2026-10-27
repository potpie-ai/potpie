"""The ``local`` extra installs on Windows, where FalkorDBLite has no build."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_ENGINE_PYPROJECT = (
    Path(__file__).resolve().parents[2] / "potpie" / "context-engine" / "pyproject.toml"
)

#: The embedded FalkorDBLite store and the pins that exist only for it.
_EMBEDDED_STORE = {"falkordblite", "potpie-falkordblite", "redis", "hiredis"}


def _environment(
    sys_platform: str, platform_system: str, machine: str = "x86_64"
) -> dict[str, str]:
    return {
        "python_version": "3.12",
        "python_full_version": "3.12.10",
        "sys_platform": sys_platform,
        "platform_system": platform_system,
        "platform_machine": machine,
        "os_name": "nt" if sys_platform == "win32" else "posix",
        "extra": "local",
    }


def _local_extra_installs(environment: dict[str, str]) -> set[str]:
    pyproject = tomllib.loads(_ENGINE_PYPROJECT.read_text(encoding="utf-8"))
    installed: set[str] = set()
    for raw in pyproject["project"]["optional-dependencies"]["local"]:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            installed.add(requirement.name)
    return installed


def test_windows_installs_ladybug_and_keeps_the_server_client() -> None:
    installed = _local_extra_installs(_environment("win32", "Windows"))

    assert installed.isdisjoint(_EMBEDDED_STORE)
    assert {"ladybug", "falkordb"} <= installed


@pytest.mark.parametrize(
    ("sys_platform", "platform_system"),
    [("linux", "Linux"), ("darwin", "Darwin")],
)
def test_other_platforms_install_the_embedded_store(
    sys_platform: str, platform_system: str
) -> None:
    installed = _local_extra_installs(_environment(sys_platform, platform_system))

    assert {"falkordblite", "redis", "hiredis", "falkordb"} <= installed
    assert "potpie-falkordblite" not in installed
    assert "ladybug" not in installed


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "ARM64"])
def test_linux_arm_installs_only_the_corrected_native_distribution(
    machine: str,
) -> None:
    installed = _local_extra_installs(_environment("linux", "Linux", machine))

    assert {"potpie-falkordblite", "falkordb", "redis", "hiredis"} <= installed
    assert {"falkordblite", "ladybug"}.isdisjoint(installed)


def test_macos_arm_keeps_upstream_native_store() -> None:
    installed = _local_extra_installs(_environment("darwin", "Darwin", "arm64"))

    assert "falkordblite" in installed
    assert "potpie-falkordblite" not in installed


@pytest.mark.parametrize("version", ["3.12", "3.13", "3.14"])
@pytest.mark.parametrize("machine", ["aarch64", "arm64", "ARM64", "AMD64"])
def test_windows_vector_dependency_is_portable_on_every_architecture(
    version: str, machine: str
) -> None:
    project = tomllib.loads(_ENGINE_PYPROJECT.read_text(encoding="utf-8"))["project"]
    requirement = next(
        Requirement(raw)
        for raw in project["dependencies"]
        if Requirement(raw).name == "httpx"
    )
    environment = _environment("win32", "Windows", machine)
    environment.update(python_version=version, python_full_version=f"{version}.0")

    assert requirement.marker is None or requirement.marker.evaluate(environment)
    assert all(Requirement(raw).name != "sqlite-vec" for raw in project["dependencies"])


@pytest.mark.parametrize("machine", ["aarch64", "arm64", "ARM64", "AMD64"])
def test_windows_telemetry_uses_installable_transport(machine: str) -> None:
    project = tomllib.loads(_ENGINE_PYPROJECT.read_text(encoding="utf-8"))["project"]
    environment = _environment("win32", "Windows", machine)
    selected = {
        req.name
        for raw in project["optional-dependencies"]["observability"]
        if (req := Requirement(raw)).marker is None or req.marker.evaluate(environment)
    }
    if machine == "AMD64":
        assert "opentelemetry-exporter-otlp" in selected
        assert "opentelemetry-exporter-otlp-proto-http" not in selected
    else:
        assert "opentelemetry-exporter-otlp" not in selected
        assert "opentelemetry-exporter-otlp-proto-http" in selected


pytestmark = [pytest.mark.source_contract]
