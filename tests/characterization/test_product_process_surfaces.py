"""Behavior locks for relocating Potpie's product processes.

These tests resolve the implementations through the root distribution's
console-script metadata. They intentionally avoid asserting module paths so a
pure relocation can change ownership without changing the public surfaces.
"""

# ruff: noqa: S101 - pytest characterization tests use assertions intentionally.

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any
import tomllib

from typer.main import get_command


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SCRIPTS = {"potpie", "potpie-daemon"}
EXPECTED_CLI_COMMANDS = {
    "auth",
    "backend",
    "cloud",
    "config",
    "confluence",
    "daemon",
    "doctor",
    "git",
    "github",
    "gitbucket",
    "gitlab",
    "graph",
    "jira",
    "ledger",
    "linear",
    "login",
    "logout",
    "pot",
    "record",
    "resolve",
    "search",
    "setup",
    "skills",
    "source",
    "status",
    "telemetry",
    "timeline",
    "ui",
    "use",
    "whoami",
}


def _scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["scripts"]


def _load_script(name: str) -> tuple[ModuleType, Any]:
    module_name, separator, attribute_name = _scripts()[name].partition(":")
    assert separator == ":", f"{name} must use a module:callable target"
    module = import_module(module_name)
    target = getattr(module, attribute_name)
    assert callable(target), f"{name} target must be callable"
    return module, target


def test_root_distribution_exposes_cli_and_daemon_processes() -> None:
    scripts = _scripts()
    assert set(scripts) == EXPECTED_SCRIPTS
    for name in sorted(EXPECTED_SCRIPTS):
        _load_script(name)


def test_cli_top_level_command_surface_is_unchanged() -> None:
    module, _main = _load_script("potpie")
    command = get_command(module.app)
    assert set(command.commands) == EXPECTED_CLI_COMMANDS


def test_shipped_daemon_entrypoint_is_canonical_and_nonreflective() -> None:
    module, _main = _load_script("potpie-daemon")
    assert module.__name__ == "potpie.daemon.__main__"
    assert not hasattr(module, "create_app")
