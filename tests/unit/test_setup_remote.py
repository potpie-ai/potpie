"""``potpie setup --remote`` — the first run of a machine that provisions nothing.

``_refuse_remote_setup_target`` is right that ``setup`` provisions *this*
machine and must not be aimed at a remote one. But a remote-only install still
has a first run, and until this existed there was no command that performed it:
on Windows, on Linux older than glibc 2.39, and on anyone who simply does not
want a local graph, ``potpie setup`` was a wizard whose every step was
inapplicable.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli.commands import _common, host as host_cmd
from potpie.cli.main import app


class _Pot:
    def __init__(self, pot_id: str) -> None:
        self.pot_id = pot_id


class _Installed:
    agent = "claude"
    changed = ("potpie-cli", "potpie-graph")


@pytest.fixture()
def remote(monkeypatch, tmp_path):
    """A reachable managed endpoint and a skill manager that touches no disk."""
    _common.set_host(None)
    hosts.reset_for_tests()
    monkeypatch.setattr(hosts, "home_dir", lambda: tmp_path)
    monkeypatch.setattr(hosts, "managed_env_override", lambda: None)

    written: dict[str, Any] = {}
    monkeypatch.setattr(
        hosts,
        "set_managed_endpoint",
        lambda url, token: written.update(url=url, token=token),
    )
    monkeypatch.setattr(
        hosts, "set_persisted_origin", lambda o: written.update(origin=o)
    )
    monkeypatch.setattr(hosts, "stored_managed_token", lambda: "")
    monkeypatch.setattr(
        host_cmd, "probe_managed_endpoint", lambda url, token: [_Pot("pot_1")]
    )

    from potpie_context_engine.bootstrap import host_wiring

    installs: list[dict[str, Any]] = []

    class _Skills:
        def install(self, **kwargs: Any) -> Any:
            installs.append(kwargs)
            return _Installed()

    monkeypatch.setattr(host_wiring, "build_skill_manager", lambda: _Skills())

    # Nothing here may build a host: that is the property under test.
    monkeypatch.setattr(
        hosts,
        "build_host",
        lambda origin: pytest.fail(f"remote setup built the {origin} host"),
    )
    yield written, installs
    _common.set_host(None)
    hosts.reset_for_tests()


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def test_remote_setup_stores_activates_and_installs_skills(remote) -> None:
    written, installs = remote

    result = _run("--json", "setup", "--remote", "http://svc.example")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "remote"
    assert payload["endpoint"] == "http://svc.example"
    assert payload["active_origin"] == hosts.MANAGED
    assert payload["pots_visible"] == 1
    assert payload["skills_changed"] == 2
    assert written == {
        "url": "http://svc.example",
        "token": "",
        "origin": hosts.MANAGED,
    }
    # Skills are files on *this* machine, so they install at global scope with
    # no host of any kind involved.
    assert installs == [{"agent": "claude", "skill_id": None, "scope": "global"}]


def test_remote_setup_passes_the_token_through(remote) -> None:
    written, _ = remote

    result = _run("--json", "setup", "--remote", "http://svc.example", "--token", "k3y")

    assert result.exit_code == 0, result.output
    assert written["token"] == "k3y"  # noqa: S105 - a fixture value, not a credential


def test_an_unreachable_endpoint_is_not_stored(remote, monkeypatch) -> None:
    """A setup that stored a dead endpoint and exited 0 would leave the machine
    in the state it was run to get out of."""
    written, _ = remote

    def _refuse(url: str, token: str) -> Any:
        from potpie.cli.commands._common import fail

        fail(code="unavailable", message=f"The managed host at {url} did not answer")

    monkeypatch.setattr(host_cmd, "probe_managed_endpoint", _refuse)

    result = _run("--json", "setup", "--remote", "http://svc.example")

    assert result.exit_code != 0
    assert written == {}


def test_an_unparseable_url_is_refused_before_anything_is_written(remote) -> None:
    written, _ = remote

    result = _run("--json", "setup", "--remote", "svc.example")

    assert result.exit_code != 0, result.output
    assert json.loads(result.output)["code"] == "validation_error"
    assert written == {}


@pytest.mark.parametrize(
    "flag",
    [
        ["--backend", "neo4j"],
        ["--daemon"],
        ["--embeddings", "none"],
        ["--embedding-model", "all-MiniLM-L6-v2"],
    ],
)
def test_local_only_flags_are_refused_rather_than_ignored(remote, flag) -> None:
    """Each of these provisions local storage. A run that silently dropped them
    would report success for a setup that did none of what was asked."""
    written, _ = remote

    result = _run("--json", "setup", "--remote", "http://svc.example", *flag)

    assert result.exit_code != 0, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert flag[0] in payload["message"]
    assert written == {}


def test_dry_run_writes_nothing(remote) -> None:
    written, installs = remote

    result = _run("--json", "setup", "--remote", "http://svc.example", "--dry-run")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["endpoint"] == "http://svc.example"
    assert written == {} and installs == []


def test_token_without_remote_is_refused(remote) -> None:
    """Otherwise it reads as a credential that was accepted and then ignored."""
    result = _run("--json", "setup", "--token", "k3y")

    assert result.exit_code != 0, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "--remote" in payload["recommended_next_action"]
