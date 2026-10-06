"""The runtime records name the process that serves, not the one launched."""

# ruff: noqa: S101 - pytest assertions are intentional.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from potpie.daemon.discovery import discovery_path, read_daemon_pid
from potpie.daemon.lifecycle import Daemon
from potpie.runtime import ControllerStatus, RuntimeEndpoint
from potpie.runtime.protocol import DaemonStatusResult, ProtocolError
from potpie_context_engine import Failure, Success

_LAUNCHED_PID = 4100
_SERVING_PID = 4242
_INSTANCE = "instance-under-test"


class _OwnedController:
    """A controller whose child is a launcher that started the real daemon."""

    pid = _LAUNCHED_PID

    async def start(self):
        return Success(
            ControllerStatus(
                running=True,
                ready=True,
                pid=_LAUNCHED_PID,
                instance_id=_INSTANCE,
                exit_code=None,
            )
        )

    async def stop(self):
        return Success(None)


class _Observer:
    def __init__(self, outcome) -> None:
        self._outcome = outcome

    async def status(self):
        return self._outcome


def _started_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observer: _Observer | None
) -> tuple[Daemon, dict]:
    owned = _OwnedController()
    monkeypatch.setattr(Daemon, "_new_controller", lambda *_a, **_k: owned)
    monkeypatch.setattr("potpie.daemon.lifecycle._pid_alive", lambda pid: True)
    daemon = Daemon(home=tmp_path, in_process=False)
    daemon._pending_endpoint = RuntimeEndpoint(
        kind="uds", address=str(tmp_path / "daemon.sock")
    )
    daemon._pending_instance_id = _INSTANCE
    daemon._pending_observer = observer  # type: ignore[assignment]
    info = daemon.start()
    return daemon, info


def _status(*, instance_id: str = _INSTANCE, pid: int = _SERVING_PID):
    return Success(
        DaemonStatusResult(
            instance_id=instance_id,
            pid=pid,
            lifecycle_state="ready",
            backend_profile="in_memory",
            ui_url="http://127.0.0.1:1",
        )
    )


def test_records_carry_the_pid_the_daemon_reports_for_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a Windows venv redirector the launched pid is the redirector."""
    daemon, info = _started_daemon(tmp_path, monkeypatch, _Observer(_status()))

    assert info["pid"] == _SERVING_PID
    assert read_daemon_pid(tmp_path) == _SERVING_PID
    discovery = json.loads(discovery_path(tmp_path).read_text(encoding="utf-8"))
    assert discovery["pid"] == _SERVING_PID
    # The owned controller still drives the daemon it launched.
    assert daemon._controller_for_existing(_SERVING_PID, observer=None) is (
        daemon._controller
    )


@pytest.mark.parametrize(
    "outcome",
    [
        Failure(ProtocolError(code="unavailable", message="no answer")),
        _status(instance_id="another-boot"),
        _status(pid=0),
    ],
)
def test_the_launched_pid_is_the_fallback_when_the_report_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome
) -> None:
    _daemon, info = _started_daemon(tmp_path, monkeypatch, _Observer(outcome))

    assert info["pid"] == _LAUNCHED_PID
    assert read_daemon_pid(tmp_path) == _LAUNCHED_PID
