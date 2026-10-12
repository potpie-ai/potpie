"""``potpie daemon status`` exits on liveness; ``daemon logs`` is bounded."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli import main as host_cli
from potpie.cli.commands import _common
from potpie.cli.commands import daemon as daemon_commands
from potpie.daemon.lifecycle import Daemon

runner = CliRunner()


class _StatusOnly:
    def __init__(self, status: dict[str, Any]) -> None:
        self._status = status

    def status(self) -> dict[str, Any]:
        return dict(self._status)


def _use(monkeypatch: pytest.MonkeyPatch, daemon: object) -> None:
    monkeypatch.setattr(daemon_commands, "Daemon", lambda **_kwargs: daemon)


def test_status_exits_unavailable_when_no_daemon_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch, Daemon(home=tmp_path, in_process=False))

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.stdout
    payload = json.loads(result.stdout)
    assert payload["code"] == "daemon_unavailable"
    # The status survives under the error keys, so a caller gating on the exit
    # code can still report what it found.
    assert payload["up"] is False
    assert payload["home"] == str(tmp_path)
    assert payload["recommended_next_action"] == "start it with 'potpie daemon start'"


def test_status_exits_unavailable_for_a_process_that_does_not_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(
        monkeypatch,
        _StatusOnly(
            {
                "up": True,
                "ready": False,
                "mode": "detached",
                "home": str(tmp_path),
                "pid": 4242,
                "detail": "detached daemon running",
            }
        ),
    )

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.stdout
    payload = json.loads(result.stdout)
    assert payload["pid"] == 4242
    assert "4242" in payload["message"]
    assert payload["recommended_next_action"] == (
        "restart it with 'potpie daemon restart'"
    )


def test_status_exits_zero_only_when_the_daemon_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(
        monkeypatch,
        _StatusOnly(
            {
                "up": True,
                "ready": True,
                "mode": "detached",
                "home": str(tmp_path),
                "pid": 4242,
            }
        ),
    )

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ready"] is True
    assert "code" not in payload


def _log(home: Path, lines: list[str]) -> Path:
    path = home / "logs" / "potpied.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_logs_tail_is_bounded_and_names_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_file = _log(tmp_path, [f"line {n}" for n in range(10)])
    _use(monkeypatch, Daemon(home=tmp_path, in_process=False))

    result = runner.invoke(host_cli.app, ["--json", "daemon", "logs", "--tail", "2"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["lines"] == ["line 8", "line 9"]
    assert payload["log_file"] == str(log_file)
    assert payload["follow"] is False


def test_logs_since_filters_by_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _log(
        tmp_path,
        [
            "2026-08-12 09:00:00,000 INFO potpie [] old",
            "2026-08-12 11:00:00,000 INFO potpie [] new",
        ],
    )
    _use(monkeypatch, Daemon(home=tmp_path, in_process=False))

    result = runner.invoke(
        host_cli.app,
        ["--json", "daemon", "logs", "--since", "2026-08-12T10:00:00"],
    )

    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["lines"] == [
        "2026-08-12 11:00:00,000 INFO potpie [] new"
    ]


@pytest.mark.parametrize(
    "args",
    [["--since", "yesterday"], ["--tail", "-1"]],
)
def test_logs_refuse_unreadable_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    _use(monkeypatch, Daemon(home=tmp_path, in_process=False))

    result = runner.invoke(host_cli.app, ["--json", "daemon", "logs", *args])

    assert result.exit_code == _common.EXIT_VALIDATION, result.stdout
    assert json.loads(result.stdout)["code"] == "validation_error"


class _Following:
    def __init__(self) -> None:
        self.asked: dict[str, Any] = {}

    def follow_logs(self, *, tail: int | None, since: object) -> list[str]:
        self.asked = {"tail": tail, "since": since}
        return ["one", "two"]


def test_follow_streams_one_json_object_per_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    following = _Following()
    _use(monkeypatch, following)

    result = runner.invoke(
        host_cli.app, ["--json", "daemon", "logs", "--follow", "--tail", "5"]
    )

    assert result.exit_code == 0, result.stdout
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        {"line": "one"},
        {"line": "two"},
    ]
    assert following.asked == {"tail": 5, "since": None}
