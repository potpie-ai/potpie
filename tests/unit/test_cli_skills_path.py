"""``--path`` has to be resolved in the caller's process, not the daemon's.

Every ``skills`` subcommand crosses an RPC to a daemon running with whatever
working directory it was launched from — often ``/``, or a directory belonging
to a terminal closed weeks ago. A relative ``--path`` therefore resolved *there*:
``skills install --path .`` wrote eleven skill directories into the daemon's cwd
and reported the install as done for the repo the user was standing in. A quoted
``~/project`` was worse, because nothing expanded it and the daemon created a
directory literally named ``~``.

These assert on what the CLI hands across the boundary, because that is where
the only process that knows the caller's cwd stops being involved.
"""

from __future__ import annotations

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
    monkeypatch.chdir(tmp_path)

    result = _run("install", "--path", "~/project", cwd=tmp_path)

    assert result.exit_code == 0, result.output
    sent = Path(recorded.paths[0] or "")
    assert sent.is_absolute()
    assert "~" not in sent.parts


def test_no_path_stays_unset(recorded, monkeypatch, tmp_path) -> None:
    """Global scope has no path, and must not acquire the caller's cwd."""
    monkeypatch.chdir(tmp_path)

    result = _run("status", cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert recorded.paths == [None]
