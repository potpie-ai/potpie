"""Smoke tests for the setup POC wizard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main
from adapters.inbound.cli.setup_poc import read_setup_state, setup_state_path


@pytest.fixture
def isolated_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cfg = tmp_path / "config" / "potpie"
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("POTPIE_DATA_HOME", str(data))
    return cfg


def test_setup_yes_json_exits_zero(isolated_setup: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=isolated_setup.parent.parent):
        result = runner.invoke(
            cli_main.app,
            ["--json", "setup", "--repo", ".", "--yes"],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["already_setup"] is False
    assert len(payload["steps"]) >= 7
    assert all("name" in s and "status" in s for s in payload["steps"])
    assert payload["next_commands"] == ["potpie status"]


def test_setup_idempotent_second_run(isolated_setup: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=isolated_setup.parent.parent):
        first = runner.invoke(cli_main.app, ["--json", "setup", "--repo", ".", "--yes"])
        assert first.exit_code == 0, first.stdout
        second = runner.invoke(cli_main.app, ["--json", "setup", "--repo", ".", "--yes"])
    assert second.exit_code == 0
    payload = json.loads(second.stdout)
    assert payload["ok"] is True
    assert payload["already_setup"] is True
    state = read_setup_state()
    assert state.get("status") == "complete"
    assert setup_state_path().is_file()
