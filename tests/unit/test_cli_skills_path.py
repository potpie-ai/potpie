"""``--path`` is resolved against the caller's cwd before the skill service sees it.

The service receives an absolute, existing directory: a relative ``--path`` is
resolved here, a quoted ``~/project`` is expanded (nothing downstream creates a
directory literally named ``~``), and a path that does not exist is refused
rather than grown into a skills tree nobody meant.

These assert on what the CLI hands the skill service, because that is the
boundary after which nothing knows the caller's cwd.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, skills


class _Skills:
    """Records the ``path`` each command actually sends to the skill service."""

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
        self.disabled: tuple[object, ...] = ()


@pytest.fixture()
def recorded(monkeypatch) -> _Skills:
    service = _Skills()
    monkeypatch.setattr(skills, "get_skill_service", lambda: service)
    return service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
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


def test_a_tilde_path_is_expanded_before_the_service_sees_it(
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


def test_install_still_accepts_no_daemon(recorded, monkeypatch, tmp_path) -> None:
    """The hidden compatibility flag is accepted and changes nothing."""
    monkeypatch.chdir(tmp_path)

    result = _run("install", "--no-daemon", cwd=tmp_path)

    assert result.exit_code == 0, result.output
    assert recorded.paths == [None]


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
    # Refused before the service was asked, so nothing was written anywhere.
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
