"""Parse-time CLI failures must honor the --json error contract."""

from __future__ import annotations

import json
from unittest.mock import patch

import click
import pytest
import typer
from typer._click.exceptions import Abort as TyperAbort
from typer.testing import CliRunner

from potpie.cli import host_cli
from potpie.cli.commands import _common


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


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--json", "pot", "jira-project", "ingest", "ENG"], "jira-project"),
        (["--json", "pot", "linear-team", "ingest", "ENG"], "linear-team"),
        (["--json", "pot", "jira-project", "diff-sync"], "jira-project"),
        (["--json", "pot", "linear-team", "diff-sync"], "linear-team"),
    ],
)
def test_removed_pot_queue_commands_emit_structured_unknown_command(
    argv: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        host_cli.run_cli(argv)

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code"] == "usage_error"
    assert expected in payload["message"]


def test_pot_help_no_longer_lists_removed_queue_groups() -> None:
    result = CliRunner().invoke(host_cli.app, ["pot", "--help"])
    assert result.exit_code == 0
    assert "linear-team" not in result.output
    assert "jira-project" not in result.output


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

    _common.bootstrap_output_flags_from_argv(["pot", "list"])
    assert not _common.is_json()
    assert not _common.is_verbose()


def test_bootstrap_output_flags_ignore_positional_after_double_dash() -> None:
    _common.bootstrap_output_flags_from_argv(
        ["source", "add", "repo", ".", "--", "--json", "--verbose"]
    )
    assert not _common.is_json()
    assert not _common.is_verbose()


@pytest.mark.parametrize("abort_exc", [TyperAbort(), click.Abort()])
def test_run_cli_abort_exits_cleanly(
    abort_exc: BaseException,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch.object(host_cli, "app", side_effect=abort_exc):
        with pytest.raises(typer.Exit) as exc_info:
            host_cli.run_cli(["pot", "list"])

    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_converts_abort_to_system_exit() -> None:
    with patch.object(host_cli, "app", side_effect=TyperAbort()):
        with pytest.raises(SystemExit) as exc_info:
            host_cli.main()

    assert exc_info.value.code == 1
