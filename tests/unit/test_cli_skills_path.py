"""``--path`` has to be resolved in the caller's process, not the daemon's.

Every daemon-backed ``skills`` subcommand crosses an RPC to a daemon running with
whatever working directory it was launched from — often ``/``, or a directory
belonging to a terminal closed weeks ago. A relative ``--path`` therefore resolved *there*:
``skills install --path .`` wrote eleven skill directories into the daemon's cwd
and reported the install as done for the repo the user was standing in. A quoted
``~/project`` was worse, because nothing expanded it and the daemon created a
directory literally named ``~``.

These assert on what the CLI hands across the boundary, because that is where
the only process that knows the caller's cwd stops being involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, skills


class _Skills:
    """Records the ``path`` each command actually sends to the host."""

    def __init__(self) -> None:
        self.paths: list[str | None] = []

    def _record(self, path: str | None):
        self.paths.append(path)

    def list(self, *, agent, scope="global", path=None):
        del agent, scope
        self._record(path)
        return []

    def install(self, *, agent, skill_id=None, path=None, scope="global"):
        del skill_id, scope
        self._record(path)
        return _Result(agent)

    def update(self, *, agent, skill_id=None, all_=False, path=None, scope="global"):
        del skill_id, all_, scope
        self._record(path)
        return _Result(agent)

    def remove(self, *, agent, skill_id=None, all_=False, path=None, scope="global"):
        del skill_id, all_, scope
        self._record(path)
        return _Result(agent)

    def status(self, *, agent, path=None, scope="global"):
        del scope
        self._record(path)
        return _Status(agent)


class _Result:
    def __init__(self, agent: str) -> None:
        self.agent = agent
        self.changed: tuple[str, ...] = ()
        self.metadata: dict[str, object] = {}


class _Status:
    def __init__(self, agent: str) -> None:
        self.agent = agent
        self.installed: tuple[object, ...] = ()
        self.missing: tuple[object, ...] = ()
        self.outdated: tuple[object, ...] = ()


@pytest.fixture()
def recorded(monkeypatch) -> _Skills:
    service = _Skills()
    monkeypatch.setattr(skills, "_skills", lambda: service)
    return service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_host(None)
    _common.set_json(False)


def _run(*args: str, cwd: Path):
    _common.set_json(True)
    return CliRunner().invoke(skills.skills_app, list(args), env={"PWD": str(cwd)})


@pytest.mark.parametrize(
    "args",
    [
        ("list",),
        ("install",),
        ("update",),
        ("remove", "--all"),
        ("status",),
    ],
)
def test_every_command_sends_an_absolute_path(
    recorded, monkeypatch, tmp_path, args
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    result = _run(*args, "--path", ".", cwd=repo)

    assert result.exit_code == 0, result.output
    assert recorded.paths == [str(repo.resolve())]


def test_a_tilde_path_is_expanded_before_it_crosses_the_boundary(
    recorded, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path)

    result = _run("install", "--path", "~/project", cwd=tmp_path)

    assert result.exit_code == 0, result.output
    sent = Path(recorded.paths[0] or "")
    assert sent.is_absolute()
    assert "~" not in sent.parts
    assert sent == (tmp_path / "project").resolve()


def test_no_path_stays_unset(recorded, monkeypatch, tmp_path) -> None:
    """Global scope has no path, and must not acquire the caller's cwd."""
    monkeypatch.chdir(tmp_path)

    result = _run("status", cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert recorded.paths == [None]


def test_install_can_bypass_the_daemon(recorded, monkeypatch, tmp_path) -> None:
    local = _Skills()
    monkeypatch.setattr(skills, "_local_skills", lambda: local)
    monkeypatch.chdir(tmp_path)

    result = _run("install", "--no-daemon", cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert local.paths == [None]
    assert recorded.paths == []


@pytest.mark.parametrize(
    "args",
    [
        ("list",),
        ("install",),
        ("update",),
        ("remove", "--all"),
        ("status",),
    ],
)
def test_a_path_that_is_not_there_is_refused_not_created(
    recorded, monkeypatch, tmp_path, args
) -> None:
    """The installer creates whatever it is pointed at, so a typo grew a tree.

    ``skills install --path ~/porject`` built an entire skills directory in a
    repository nobody had, reported the install as done, and left the real one
    untouched — the failure mode is silent by construction, because the check
    that would have caught it is the one the command performs.
    """
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "porject"

    result = _run(*args, "--path", str(missing), cwd=tmp_path)

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert str(missing) in payload["message"]
    assert "mkdir" in (payload["recommended_next_action"] or "")
    # Refused before the host was asked, so nothing was written anywhere.
    assert recorded.paths == []
    assert not missing.exists()


def test_a_path_pointing_at_a_file_is_refused(recorded, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    not_a_dir = tmp_path / "README.md"
    not_a_dir.write_text("x", encoding="utf-8")

    result = _run("install", "--path", str(not_a_dir), cwd=tmp_path)

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "is a file" in payload["message"]
    assert recorded.paths == []
