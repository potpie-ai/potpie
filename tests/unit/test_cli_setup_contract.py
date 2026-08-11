"""What ``potpie setup`` promises: one repo source, and no false failures.

Two defects met here. Its own docstring says "idempotent first-run", but every
run appended another repo source, because the dedup compared the raw ``--repo .``
flag against a stored *resolved* location and could never match. And ``--json``
setup — the only mode agents and CI use — sent the entire first run over the
daemon RPC under one 30s client deadline, so a cold machine reported
``unavailable`` at exit 2 for work the daemon went on to finish. The two
compound: the natural response to a false failure is to re-run, and re-running
was what duplicated the sources.

The transport half is exercised over a real loopback socket rather than a
monkeypatched ``httpx.post``, because the claim under test is about what the
adapter does with a real timeout while the server keeps working.
"""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.daemon.client import DaemonRpcClient
from potpie.daemon.lifecycle import Daemon
from potpie_context_core.errors import ContextEngineDisabled
from potpie_context_core.lifecycle import DONE, SKIPPED, SetupPlan, StepResult
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.application.services import setup_orchestrator
from potpie_context_engine.bootstrap.host_wiring import build_host_shell

runner = CliRunner()


@pytest.fixture()
def host(tmp_path, monkeypatch):
    """A real host on a real (temp) home — the store is what is under test."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return build_host_shell(backend=InMemoryGraphBackend())


def _plan(**overrides) -> SetupPlan:
    base = {
        "host_mode": "in_process",
        "backend": "in_memory",
        "repo": ".",
        "pot": "p",
        "agent": "default",
        "embeddings": "none",
    }
    return SetupPlan(**{**base, **overrides})


def _repo_sources(host, pot_id: str) -> list:
    return [s for s in host.pots.list_sources(pot_id=pot_id) if s.kind == "repo"]


def _step(report, name: str) -> StepResult:
    return next(s for s in report.steps if s.step == name)


# --- P1-6: N runs, one source ----------------------------------------------


def test_three_setup_runs_leave_one_repo_source(host, tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    # No remote: `.` resolves to the absolute working tree, which is the exact
    # spelling the old guard compared against the literal string ".".
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)

    reports = [host.setup.run(_plan()) for _ in range(3)]

    assert all(r.ok for r in reports)
    active = host.pots.active_pot()
    assert active is not None
    sources = _repo_sources(host, active.pot_id)
    assert len(sources) == 1, [s.location for s in sources]
    assert sources[0].location == str(repo.resolve())


def test_a_repeat_run_reports_the_source_step_as_skipped_with_the_existing_id(
    host, tmp_path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)

    first = host.setup.run(_plan())
    second = host.setup.run(_plan())

    registered = _step(first, "source")
    repeated = _step(second, "source")
    assert registered.state == DONE
    assert repeated.state == SKIPPED
    assert repeated.metadata["already_registered"] is True
    assert repeated.metadata["source_id"] == registered.metadata["source_id"]
    # A converged re-run is not a degraded run.
    assert second.to_dict()["ok"] is True


def test_the_same_repo_spelled_two_ways_is_one_source(host) -> None:
    first = host.setup.run(_plan(repo="git@github.com:Acme/Shop.git"))
    second = host.setup.run(_plan(repo="https://github.com/acme/shop/"))

    active = host.pots.active_pot()
    assert active is not None
    assert len(_repo_sources(host, active.pot_id)) == 1
    assert _step(first, "source").state == DONE
    assert _step(second, "source").state == SKIPPED


def test_setup_lower_cases_a_remote_the_way_source_add_does(
    host, tmp_path, monkeypatch
) -> None:
    """One repo, one spelling — through the real ``git remote get-url`` path.

    The orchestrator carried a private copy of the shared normalizer that had
    lost its ``.lower()``, so ``setup`` persisted ``github.com/Potpie-AI/Potpie``
    where ``source add repo .`` persisted ``github.com/potpie-ai/potpie`` for the
    same repository. Nothing caught it because every lookup re-normalized both
    sides — but the dedup above compares stored refs.
    """
    repo = tmp_path / "mixedcase"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "git@github.com:Potpie-AI/Potpie.git"],
        cwd=repo,
        check=True,
    )
    monkeypatch.chdir(repo)

    host.setup.run(_plan(repo="."))

    active = host.pots.active_pot()
    assert active is not None
    sources = _repo_sources(host, active.pot_id)
    assert [s.location for s in sources] == ["github.com/potpie-ai/potpie"]
    assert host.pots.repo_default(repo="github.com/potpie-ai/potpie") == active.pot_id


def test_a_repeat_run_does_not_steal_a_deliberate_repo_default(host) -> None:
    host.setup.run(_plan(repo="github.com/acme/shop"))
    other = host.pots.create_pot(name="chosen-by-hand")
    host.pots.set_repo_default(repo="github.com/acme/shop", pot_id=other.pot_id)

    host.setup.run(_plan(repo="github.com/acme/shop"))

    assert host.pots.repo_default(repo="github.com/acme/shop") == other.pot_id
    active = host.pots.active_pot()
    assert active is not None and active.pot_id != other.pot_id
    assert len(_repo_sources(host, active.pot_id)) == 1


def test_a_dangling_repo_default_is_still_repaired_by_a_repeat_run(host) -> None:
    # The carve-out: a run that died between add_source and the binding, or a
    # binding left pointing at a pot that no longer exists, must still converge.
    host.setup.run(_plan(repo="github.com/acme/shop"))
    # Straight to the store: the service refuses to bind a pot that is gone,
    # which is precisely the state left behind when one is removed afterwards.
    host.pots.store.set_repo_default(repo="github.com/acme/shop", pot_id="pot_deleted")

    host.setup.run(_plan(repo="github.com/acme/shop"))

    active = host.pots.active_pot()
    assert active is not None
    assert host.pots.repo_default(repo="github.com/acme/shop") == active.pot_id


# --- P1-5: a client deadline is not a failure -------------------------------


@contextmanager
def _loopback_rpc(*, sleep_s: float):
    """A real HTTP endpoint that answers ``sleep_s`` late, and says it finished."""
    finished: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            time.sleep(sleep_s)
            finished.append("done")
            body = json.dumps({"ok": True, "result": "completed"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:  # keep pytest output clean
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", finished
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _Endpoint:
    """An address and a key, which is all ``DaemonRpcClient`` asks its daemon for."""

    def __init__(self, base_url: str) -> None:
        self._discovery = {"base_url": base_url, "token": "t"}

    def discovery(self) -> dict[str, str]:
        return dict(self._discovery)


def test_a_client_deadline_is_not_reported_as_the_work_failing() -> None:
    with _loopback_rpc(sleep_s=1.0) as (base_url, finished):
        client = DaemonRpcClient(daemon=_Endpoint(base_url), timeout_s=0.2)

        with pytest.raises(ContextEngineDisabled) as raised:
            client.call("pots", "list_pots")

        message = str(raised.value)
        assert "did not answer within 0.2s" in message
        assert "may still be running" in message
        assert "before re-running" in raised.value.recommended_next_action
        # The point of the whole exception: the server was not failing.
        time.sleep(1.5)
        assert finished == ["done"]


def test_the_setup_surface_carries_no_client_deadline() -> None:
    client = DaemonRpcClient(timeout_s=30.0)
    assert client._deadline_for("setup") is None
    assert client._deadline_for("pots") == 30.0

    # Declared *and* wired: the same server that times a `pots` call out answers
    # a `setup` call, because the deadline never reached httpx for that surface.
    with _loopback_rpc(sleep_s=0.5) as (base_url, _finished):
        deadlined = DaemonRpcClient(daemon=_Endpoint(base_url), timeout_s=0.05)
        assert deadlined.call("setup", "run") == "completed"
        with pytest.raises(ContextEngineDisabled):
            deadlined.call("pots", "list_pots")


def test_an_unreachable_host_still_reads_as_unreachable() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]

    client = DaemonRpcClient(
        daemon=_Endpoint(f"http://127.0.0.1:{closed_port}"), timeout_s=2.0
    )

    with pytest.raises(ContextEngineDisabled) as raised:
        client.call("pots", "list_pots")

    message = str(raised.value)
    assert "is unavailable" in message
    assert "did not answer within" not in message


# --- P1-5: --json setup does not cross the RPC at all -----------------------


def test_json_setup_runs_in_process_and_never_crosses_the_daemon_rpc(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # Daemon host mode is the shipped default and the mode that used to route
    # the whole run through one 30s-deadlined POST.
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)
    monkeypatch.setattr(
        Daemon,
        "ensure",
        lambda self, plan=None: StepResult("daemon", DONE, "stubbed"),
    )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("setup crossed the daemon RPC")

    monkeypatch.setattr(httpx, "post", _forbidden)

    result = runner.invoke(
        cli_main.app,
        [
            "--json",
            "setup",
            "--backend",
            "in_memory",
            "--agent",
            "default",
            "--embeddings",
            "none",
            "--pot",
            "p",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["plan"]["host_mode"] == "daemon"
    steps = {s["step"]: s for s in payload["steps"]}
    assert steps["source"]["state"] == DONE
    assert steps["source"]["metadata"]["location"] == str(repo.resolve())


def test_json_setup_registers_the_callers_repo_not_another_directorys(
    tmp_path, monkeypatch
) -> None:
    """The cwd the location resolves against is the caller's, run after run.

    Resolved inside the daemon it was the cwd of whichever invocation first
    started that daemon, so `potpie --json setup` in repo B registered repo A
    and re-pointed A's routing at B's pot.
    """
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "daemon")
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)
    monkeypatch.setattr(
        Daemon,
        "ensure",
        lambda self, plan=None: StepResult("daemon", DONE, "stubbed"),
    )
    locations = []
    for name in ("repoA", "repoB"):
        repo = tmp_path / name
        repo.mkdir()
        monkeypatch.chdir(repo)
        result = runner.invoke(
            cli_main.app,
            [
                "--json",
                "setup",
                "--backend",
                "in_memory",
                "--agent",
                "default",
                "--embeddings",
                "none",
                "--pot",
                name,
            ],
        )
        assert result.exit_code == 0, result.stdout
        steps = {s["step"]: s for s in json.loads(result.stdout)["steps"]}
        locations.append(steps["source"]["metadata"]["location"])

    assert locations == [
        str((tmp_path / "repoA").resolve()),
        str((tmp_path / "repoB").resolve()),
    ]


def test_a_repeat_json_setup_keeps_one_source_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "ce"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setattr(setup_orchestrator, "_current_git_remote", lambda cwd: None)
    monkeypatch.setattr(
        Daemon,
        "ensure",
        lambda self, plan=None: StepResult("daemon", DONE, "stubbed"),
    )
    argv = [
        "--json",
        "setup",
        "--backend",
        "in_memory",
        "--agent",
        "default",
        "--embeddings",
        "none",
        "--pot",
        "p",
    ]

    for _ in range(3):
        assert runner.invoke(cli_main.app, argv).exit_code == 0

    pots = json.loads((tmp_path / "ce" / "pots.json").read_text(encoding="utf-8"))
    rows = [
        row
        for sources in pots["sources"].values()
        for row in sources
        if row["kind"] == "repo"
    ]
    assert len(rows) == 1, rows
    assert Path(rows[0]["location"]) == repo.resolve()
