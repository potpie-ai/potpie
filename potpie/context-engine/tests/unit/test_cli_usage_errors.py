"""Parse-time CLI failures must honor the --json error contract."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from adapters.inbound.cli import host_cli
from adapters.inbound.cli.commands import _common


@pytest.fixture(autouse=True)
def _reset_cli_output_state() -> None:
    _common.set_json(False)
    _common.set_verbose(False)
    yield
    _common.set_json(False)
    _common.set_verbose(False)


def test_missing_argument_emits_structured_json_via_run_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        host_cli.run_cli(["--json", "pot", "create"])

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "usage_error"
    assert "NAME" in payload["message"]
    assert payload["recommended_next_action"]


def test_unknown_subcommand_emits_structured_json_via_run_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        host_cli.run_cli(["--json", "pot", "nope"])

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "usage_error"
    assert "nope" in payload["message"]
    assert payload["recommended_next_action"]


def test_missing_argument_keeps_typer_text_in_human_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        host_cli.run_cli(["pot", "create"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Missing argument" in captured.err or "Missing argument" in captured.out


def test_cli_runner_still_uses_typer_for_direct_app_invoke() -> None:
    result = CliRunner().invoke(host_cli.app, ["--json", "pot", "create"])
    assert result.exit_code == 2
    assert "Missing argument" in result.output


def test_bootstrap_output_flags_reset_per_argv() -> None:
    _common.bootstrap_output_flags_from_argv(["--json", "pot", "list"])
    assert _common.is_json()
    assert not _common.is_verbose()

    _common.bootstrap_output_flags_from_argv(["pot", "list"])
    assert not _common.is_json()
    assert not _common.is_verbose()

    _common.bootstrap_output_flags_from_argv(["-v", "pot", "list"])
    assert not _common.is_json()
    assert _common.is_verbose()


def test_bootstrap_output_flags_ignore_positional_after_double_dash() -> None:
    _common.bootstrap_output_flags_from_argv(
        ["source", "add", "repo", ".", "--", "--json", "--verbose"]
    )
    assert not _common.is_json()
    assert not _common.is_verbose()
