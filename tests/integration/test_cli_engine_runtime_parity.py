from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from potpie.cli import host_cli
from potpie.cli.commands import _common
from potpie.daemon.lifecycle import Daemon
from potpie.setup import SetupPlan
from tests.boundary.normalization import normalize_engine_result


def _reset_cli_runtime() -> None:
    _common.reset_cli_runtime()


def _invoke_json(runner: CliRunner, args: list[str]) -> dict:
    result = runner.invoke(host_cli.app, ["--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_engine_workflow_cli_has_local_daemon_parity(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "product"
    monkeypatch.setenv("POTPIE_HOME", str(home))
    monkeypatch.setenv("POTPIE_GRAPH_BACKEND", "embedded")
    monkeypatch.setenv("POTPIE_RUNTIME_MODE", "in-process")
    runner = CliRunner()

    _invoke_json(runner, ["pot", "create", "parity", "--use"])
    _invoke_json(
        runner,
        ["source", "add", "repo", "owner/parity", "--pot", "parity", "--no-default"],
    )
    commands = (
        ["pot", "list"],
        ["pot", "info"],
        ["source", "list", "--pot", "parity"],
        ["resolve", "parity", "--pot", "parity"],
        ["graph", "catalog", "--pot", "parity"],
        ["graph", "status", "--pot", "parity"],
        ["timeline", "recent", "--pot", "parity"],
        ["ledger", "status"],
        ["ledger", "sources", "--pot", "parity"],
    )
    local = [_invoke_json(runner, command) for command in commands]

    _reset_cli_runtime()
    monkeypatch.setenv("POTPIE_RUNTIME_MODE", "daemon")
    daemon = Daemon(home=home, in_process=False, startup_timeout_s=15)
    try:
        daemon.ensure(SetupPlan(backend="embedded"))
        remote = [_invoke_json(runner, command) for command in commands]
    finally:
        daemon.stop()
        _reset_cli_runtime()

    assert normalize_engine_result(remote) == normalize_engine_result(local)
