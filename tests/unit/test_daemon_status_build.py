"""``potpie daemon status`` says which build is serving, and whether it is this one's."""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from potpie import build_info
from potpie.cli import main as host_cli
from potpie.cli.commands import daemon as daemon_commands
from potpie.daemon.lifecycle import Daemon
from potpie.runtime import (
    PROTOCOL_VERSION,
    DaemonBuild,
    DaemonStatusPayload,
    DaemonStatusRequest,
    DaemonStatusResult,
    SuccessResponse,
)
from potpie.runtime.codec import decode_response, encode_response
from potpie_context_engine import Failure, Success

FIXTURE_REV = "a" * 40
OTHER_REV = "b" * 40

runner = CliRunner()


def _status_result(build: DaemonBuild | None) -> DaemonStatusResult:
    return DaemonStatusResult(
        instance_id="instance-1",
        pid=os.getpid(),
        lifecycle_state="ready",
        backend_profile="falkordb_lite",
        ui_url="http://127.0.0.1:8765",
        version="2.0.1" if build is not None else None,
        build=build,
    )


class _Observer:
    def __init__(self, result: DaemonStatusResult) -> None:
        self._result = result

    async def status(self) -> Success[DaemonStatusResult]:
        return Success(self._result)

    async def close(self) -> None:
        return None


class _Controller:
    async def status(self) -> SimpleNamespace:
        return SimpleNamespace(ready=True)


def _attached_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, served: DaemonStatusResult
) -> Daemon:
    """A detached daemon whose live status answer is ``served``.

    The recorded PID is this test process, so liveness is genuinely true.
    """
    daemon = Daemon(home=tmp_path, in_process=False)
    discovery = SimpleNamespace(
        endpoint=SimpleNamespace(kind="uds", display=str(tmp_path / "runtime.sock"))
    )
    monkeypatch.setattr(daemon, "_recorded_pid", os.getpid)
    monkeypatch.setattr(
        daemon, "_connection_for_pid", lambda _pid: (discovery, _Observer(served))
    )
    monkeypatch.setattr(
        daemon, "_controller_for_existing", lambda _pid, *, observer: _Controller()
    )
    return daemon


def test_status_reports_the_served_build_and_that_it_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": FIXTURE_REV})
    served = DaemonBuild(rev=OTHER_REV, dirty=False, built_at="2026-06-28T00:00:00Z")
    daemon = _attached_daemon(monkeypatch, tmp_path, _status_result(served))

    status = daemon.status()

    assert status["up"] is True
    assert status["backend"] == "falkordb_lite"
    assert status["version"] == "2.0.1"
    assert status["build"] == {
        "rev": OTHER_REV,
        "dirty": False,
        "built_at": "2026-06-28T00:00:00Z",
    }
    assert status["stale"] is True


def test_status_of_this_build_is_not_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": FIXTURE_REV})
    served = DaemonBuild(rev=FIXTURE_REV, dirty=False, built_at=None)
    daemon = _attached_daemon(monkeypatch, tmp_path, _status_result(served))

    assert daemon.status()["stale"] is False


def test_an_unstamped_build_is_not_evidence_of_staleness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": FIXTURE_REV})
    daemon = _attached_daemon(monkeypatch, tmp_path, _status_result(DaemonBuild()))

    status = daemon.status()

    assert status["build"] == {"rev": None, "dirty": None, "built_at": None}
    assert status["stale"] is None


def test_a_daemon_that_does_not_report_its_build_has_no_build_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    daemon = _attached_daemon(monkeypatch, tmp_path, _status_result(None))

    status = daemon.status()

    assert status["up"] is True
    assert "version" not in status
    assert "build" not in status
    assert "stale" not in status


def test_in_process_status_is_untouched(tmp_path: Path) -> None:
    status = Daemon(home=tmp_path, in_process=True).status()

    assert status["mode"] == "in_process"
    assert "build" not in status
    assert "stale" not in status


# --- the command ---------------------------------------------------------------


class _FakeDaemon:
    def __init__(self, status: dict[str, object]) -> None:
        self._status = status

    def status(self) -> dict[str, object]:
        return dict(self._status)


_STALE_STATUS = {
    "up": True,
    "ready": True,
    "mode": "detached",
    "pid": 4242,
    "version": "2.0.1",
    "build": {"rev": OTHER_REV, "dirty": False, "built_at": None},
    "stale": True,
}


def test_json_daemon_status_carries_build_rev_and_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        daemon_commands, "Daemon", lambda **_kwargs: _FakeDaemon(_STALE_STATUS)
    )

    result = runner.invoke(host_cli.app, ["--json", "daemon", "status"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["build"]["rev"] == OTHER_REV
    assert payload["stale"] is True


def test_human_daemon_status_names_the_build_and_the_way_out_when_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_info, "build_stamp", lambda: {"rev": FIXTURE_REV})
    monkeypatch.setattr(
        daemon_commands, "Daemon", lambda **_kwargs: _FakeDaemon(_STALE_STATUS)
    )

    stale = runner.invoke(host_cli.app, ["daemon", "status"])
    monkeypatch.setattr(
        daemon_commands,
        "Daemon",
        lambda **_kwargs: _FakeDaemon({**_STALE_STATUS, "stale": False}),
    )
    current = runner.invoke(host_cli.app, ["daemon", "status"])

    assert stale.exit_code == 0, stale.stdout
    assert "bbbbbbbbbb" in stale.stdout
    assert "aaaaaaaaaa" in stale.stdout
    assert "potpie daemon restart" in stale.stdout
    assert current.exit_code == 0, current.stdout
    assert "bbbbbbbbbb (current)" in current.stdout


# --- the wire ------------------------------------------------------------------


def _status_request() -> DaemonStatusRequest:
    return DaemonStatusRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="status-1",
        payload=DaemonStatusPayload(),
    )


def _status_document(result: dict[str, object]) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "status-1",
        "outcome": {"ok": True, "value": result},
    }


def test_status_result_round_trips_the_build() -> None:
    request = _status_request()
    response = SuccessResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        outcome=Success(
            _status_result(
                DaemonBuild(
                    rev=FIXTURE_REV, dirty=True, built_at="2026-06-28T00:00:00Z"
                )
            )
        ),
    )

    decoded = decode_response(encode_response(response), request=request)

    assert isinstance(decoded, Success)
    assert decoded.value == response


def test_status_from_a_daemon_without_build_reporting_still_decodes() -> None:
    document = _status_document(
        {
            "instance_id": "instance-1",
            "pid": 42,
            "lifecycle_state": "ready",
            "backend_profile": "embedded",
            "ui_url": "http://127.0.0.1:8765",
        }
    )

    decoded = decode_response(document, request=_status_request())

    assert isinstance(decoded, Success)
    status = decoded.value.outcome.value
    assert status.version is None
    assert status.build is None


@pytest.mark.parametrize(
    "extra",
    [
        {"version": "2.0.1"},
        {"version": "2.0.1", "build": {"rev": FIXTURE_REV, "dirty": False}},
        {
            "version": "2.0.1",
            "build": {"rev": FIXTURE_REV, "dirty": "no", "built_at": None},
        },
        {
            "version": "2.0.1",
            "build": {"rev": FIXTURE_REV, "dirty": False, "built_at": None},
            "channel": "beta",
        },
    ],
)
def test_status_with_a_malformed_build_is_refused(extra: dict[str, object]) -> None:
    document = _status_document(
        {
            "instance_id": "instance-1",
            "pid": 42,
            "lifecycle_state": "ready",
            "backend_profile": "embedded",
            "ui_url": "http://127.0.0.1:8765",
            **extra,
        }
    )

    decoded = decode_response(document, request=_status_request())

    assert isinstance(decoded, Failure)
    assert decoded.error.code == "response_result_malformed"
