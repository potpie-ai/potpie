"""One hang-proof git probe, shared by every place the CLI asks git a question."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from potpie.cli import repo_location
from potpie.cli.commands import _common, graph
from potpie.setup import orchestrator
from potpie_context_engine.domain import git_probe
from potpie_context_engine.domain.git_probe import run_git_probe


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_probe_answers_inside_a_repository_and_not_outside(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")

    assert run_git_probe(["rev-parse", "--is-inside-work-tree"], cwd=repo) == "true"
    # git exits non-zero for a directory that does not exist; nothing raises.
    assert run_git_probe(["rev-parse", "HEAD"], cwd=tmp_path / "missing") is None


def test_current_git_remote_is_normalized_to_the_pot_store_key(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    assert repo_location.current_git_remote(repo) is None

    _git(repo, "remote", "add", "origin", "git@github.com:Acme/Shop.git")

    assert repo_location.current_git_remote(repo) == "github.com/acme/shop"


class _RecordingRun:
    """Stands in for ``subprocess.run`` and records how git was spawned."""

    def __init__(self, stdout: bytes = b"https://github.com/Acme/Shop.git\n") -> None:
        self.stdout = stdout
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append((argv, kwargs))
        kwargs["stdout"].write(self.stdout)
        return subprocess.CompletedProcess(argv, 0)


def test_every_call_site_spawns_git_without_pipes_or_a_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``capture_output`` with a timeout can wedge on Windows; a file cannot."""
    run = _RecordingRun()
    monkeypatch.setattr(git_probe.subprocess, "run", run)
    monkeypatch.chdir(tmp_path)

    assert repo_location.current_git_remote(tmp_path) == "github.com/acme/shop"
    assert _common._current_git_remote(tmp_path) == "github.com/acme/shop"
    setup_remote = orchestrator._current_git_remote(tmp_path) or ""
    assert setup_remote.lower() == "github.com/acme/shop"
    assert graph._current_repo_remote_for_scope() == "github.com/acme/shop"

    assert len(run.calls) == 4
    for argv, kwargs in run.calls:
        assert argv[0] == "git"
        if argv[1] == "-C":
            assert Path(argv[2]).samefile(tmp_path)
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdout"] not in (None, subprocess.PIPE)
        assert "capture_output" not in kwargs
        assert kwargs["timeout"] > 0
        assert ("creationflags" in kwargs) == (os.name == "nt")


def test_windows_probe_never_opens_a_console(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _RecordingRun()
    monkeypatch.setattr(git_probe, "_windows", lambda: True)
    monkeypatch.setattr(git_probe.subprocess, "run", run)

    run_git_probe(["remote", "get-url", "origin"], cwd=".")

    ((_argv, kwargs),) = run.calls
    assert kwargs["creationflags"] == 0x08000000  # CREATE_NO_WINDOW


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(cmd="git", timeout=2),
        FileNotFoundError("git"),
    ],
)
def test_a_probe_that_cannot_answer_returns_none(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(git_probe.subprocess, "run", _raise)

    assert run_git_probe(["remote", "get-url", "origin"]) is None
