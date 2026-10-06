"""The local installer counts the CLI as installed only when agents can use it."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from potpie.setup import cli_probe, local_installer
from potpie.setup.cli_probe import (
    REQUIRED_COMMAND_GROUPS,
    REQUIRED_COMMANDS,
    probe_cli_surface,
)
from potpie.setup.local_installer import LocalInstaller

pytestmark = pytest.mark.unit


class _HelpRun:
    """Stands in for ``subprocess.run``; writes each group's help to stdout."""

    def __init__(self, help_for) -> None:
        self.help_for = help_for
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        returncode, text = self.help_for(tuple(argv[1:-1]))
        kwargs["stdout"].write(text.encode("utf-8"))
        return subprocess.CompletedProcess(argv, returncode)


def _full_help(group: tuple[str, ...]) -> tuple[int, str]:
    return 0, "\n".join(f"│ {name}  help" for name in REQUIRED_COMMAND_GROUPS[group])


def test_installer_rejects_missing_executable(monkeypatch) -> None:
    monkeypatch.setattr(local_installer.shutil, "which", lambda _name: None)
    assert LocalInstaller().is_installed() is False


def test_installer_requires_workflow_commands(monkeypatch) -> None:
    monkeypatch.setattr(local_installer.shutil, "which", lambda _name: "potpie")
    monkeypatch.setattr(
        cli_probe.subprocess,
        "run",
        _HelpRun(lambda _group: (0, "│ doctor help\n│ status help\n")),
    )
    assert LocalInstaller().is_installed() is False


def test_installer_accepts_usable_cli(monkeypatch) -> None:
    monkeypatch.setattr(local_installer.shutil, "which", lambda _name: "potpie")
    monkeypatch.setattr(cli_probe.subprocess, "run", _HelpRun(_full_help))
    assert LocalInstaller().is_installed() is True


def test_probe_reports_every_command_it_found(monkeypatch) -> None:
    run = _HelpRun(_full_help)
    monkeypatch.setattr(cli_probe.subprocess, "run", run)

    probe = probe_cli_surface("potpie")

    assert probe["ok"] is True
    assert probe["commands"] == list(REQUIRED_COMMANDS)
    assert probe["missing_commands"] == []
    assert sorted(argv[1:-1] for argv, _kwargs in run.calls) == sorted(
        list(group) for group in REQUIRED_COMMAND_GROUPS
    )


def test_probe_reports_a_partial_cli(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_probe.subprocess,
        "run",
        _HelpRun(lambda _group: (0, "│ status  help\n│ doctor  help\n")),
    )

    probe = probe_cli_surface("potpie")

    assert probe["ok"] is False
    assert "record" in probe["missing_commands"]
    assert "graph read" in probe["missing_commands"]


def test_a_failing_help_command_is_not_usable(monkeypatch) -> None:
    def help_for(group):
        if group == ("resource",):
            return 2, "No such command 'resource'."
        return _full_help(group)

    monkeypatch.setattr(cli_probe.subprocess, "run", _HelpRun(help_for))

    probe = probe_cli_surface("potpie")

    assert probe["ok"] is False
    assert {"resource get", "resource import"} <= set(probe["missing_commands"])
    failed = [row for row in probe["probes"] if not row["ok"]]
    assert [row["command"] for row in failed] == ["resource --help"]


def test_probe_spawns_help_without_pipes_or_a_terminal(monkeypatch) -> None:
    """A pipe plus a timeout can wedge on Windows; a file cannot."""
    run = _HelpRun(_full_help)
    monkeypatch.setattr(cli_probe.subprocess, "run", run)

    probe_cli_surface("potpie")

    for _argv, kwargs in run.calls:
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stdout"] not in (None, subprocess.PIPE)
        assert "capture_output" not in kwargs
        assert kwargs["timeout"] > 0
        assert ("creationflags" in kwargs) == (os.name == "nt")


def test_an_absent_executable_is_reported_without_spawning(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("nothing to spawn")

    monkeypatch.setattr(cli_probe.subprocess, "run", refuse)

    probe = probe_cli_surface(None)

    assert probe["ok"] is False
    assert probe["missing_commands"] == list(REQUIRED_COMMANDS)
    assert probe["detail"] == "not on PATH"


def test_the_installed_cli_serves_the_workflow_commands() -> None:
    """The real CLI this checkout installs lists every required command."""
    executable = Path(sys.executable).with_name("potpie")
    if not executable.exists():
        pytest.skip("the potpie console script is not installed next to python")

    probe = probe_cli_surface(str(executable))

    assert probe["missing_commands"] == []
    assert probe["ok"] is True
