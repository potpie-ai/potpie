"""The ``local`` extra installs on Windows, where FalkorDBLite has no build."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_ENGINE_PYPROJECT = (
    Path(__file__).resolve().parents[2] / "potpie" / "context-engine" / "pyproject.toml"
)

#: The embedded FalkorDBLite store and the pins that exist only for it.
_EMBEDDED_STORE = {"falkordblite", "redis", "hiredis"}


def _environment(sys_platform: str, platform_system: str) -> dict[str, str]:
    return {
        "python_version": "3.12",
        "python_full_version": "3.12.10",
        "sys_platform": sys_platform,
        "platform_system": platform_system,
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


def test_windows_skips_the_embedded_store_and_keeps_the_server_client() -> None:
    installed = _local_extra_installs(_environment("win32", "Windows"))

    assert installed.isdisjoint(_EMBEDDED_STORE)
    assert "falkordb" in installed


@pytest.mark.parametrize(
    ("sys_platform", "platform_system"),
    [("linux", "Linux"), ("darwin", "Darwin")],
)
def test_other_platforms_install_the_embedded_store(
    sys_platform: str, platform_system: str
) -> None:
    installed = _local_extra_installs(_environment(sys_platform, platform_system))

    assert _EMBEDDED_STORE | {"falkordb"} <= installed
