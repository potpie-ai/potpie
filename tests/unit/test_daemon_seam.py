"""potpie.daemon.lifecycle.Daemon seam: in-process stand-in vs detached lifecycle (launcher faked)."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from potpie_context_core.lifecycle import DONE, SKIPPED
from potpie.daemon.lifecycle import (
    STATE_IN_PROCESS,
    STATE_NOT_RUNNING,
    STATE_RUNNING,
    STATE_UNRESPONSIVE,
    Daemon,
    parse_since,
)


def test_in_process_status_health_and_ensure_skips(tmp_path: pathlib.Path):
    d = Daemon(home=tmp_path, in_process=True)
    assert d.status()["mode"] == "in_process" and d.status()["up"] is True
    assert d.status()["state"] == STATE_IN_PROCESS
    assert d.health() == {"live": True, "mode": "in_process"}
    res = d.ensure()
    assert res.state == SKIPPED and res.metadata["mode"] == "in_process"


def test_detached_ensure_starts_when_not_running(tmp_path: pathlib.Path, monkeypatch):
    started = {}

    def fake_start_detached(home, **kw):
        started["home"] = home
        return {
            "pid": 4242,
            "socket": str(home / "daemon.sock"),
            "bind": f"unix:{home}/daemon.sock",
        }

    monkeypatch.setattr(
        "potpie.daemon.process.launcher.start_detached", fake_start_detached
    )
    d = Daemon(home=tmp_path, in_process=False)
    res = d.ensure()
    assert res.state == DONE
    assert res.metadata["pid"] == 4242
    assert started["home"] == tmp_path


def test_detached_ensure_reuses_running_daemon(tmp_path: pathlib.Path, monkeypatch):
    # Pretend a live daemon is already recorded.
    (tmp_path / "daemon.pid").write_text("999999\n")
    (tmp_path / "discovery.json").write_text('{"bind": "unix:/x/daemon.sock"}')
    monkeypatch.setattr("potpie.daemon.lifecycle._pid_alive", lambda pid: True)

    def _boom(*a, **k):  # must NOT be called when already running
        raise AssertionError("start_detached should not be called when daemon is up")

    monkeypatch.setattr("potpie.daemon.process.launcher.start_detached", _boom)
    d = Daemon(home=tmp_path, in_process=False)
    res = d.ensure()
    assert res.state == SKIPPED and "already running" in (res.detail or "")


def test_install_is_idempotent_noop(tmp_path: pathlib.Path):
    d = Daemon(home=tmp_path, in_process=False)
    out = d.install()
    assert (
        out["installed"] is False
    )  # never raises; does not gate the installer setup step


# --- S2-15: a live pid is not a serving daemon ------------------------------
#
# Everything below runs against real processes, real sockets and real files.
# The defect lived in exactly the gap between "the kernel still knows this pid"
# and "this process answers", so a fake process or a stubbed health probe would
# test the assumption that produced the bug.


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        body = json.dumps({"ok": True, "mode": "daemon", "backend": "falkordb_lite"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *_args) -> None:  # keep pytest output clean
        return


@pytest.fixture
def serving_daemon(tmp_path: pathlib.Path) -> Iterator[Daemon]:
    """A real HTTP server answering ``/health``, recorded like a real daemon."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _record_daemon(tmp_path, pid=os.getpid(), port=server.server_address[1])
        yield Daemon(home=tmp_path, in_process=False, health_timeout_s=2.0)
    finally:
        server.shutdown()
        server.server_close()


def _record_daemon(home: pathlib.Path, *, pid: int, port: int) -> None:
    (home / "daemon.pid").write_text(f"{pid}\n")
    (home / "discovery.json").write_text(
        json.dumps({"base_url": f"http://127.0.0.1:{port}", "pid": pid})
    )


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_serving_daemon_is_running(serving_daemon: Daemon) -> None:
    status = serving_daemon.status()

    assert status["state"] == STATE_RUNNING
    assert status["up"] is True and status["serving"] is True
    assert status["backend"] == "falkordb_lite"


def test_a_live_pid_that_does_not_answer_is_not_running(
    tmp_path: pathlib.Path,
) -> None:
    """The SIGSTOP shape: the process exists, so ``os.kill(pid, 0)`` succeeds,
    but nothing is served. This reported "detached daemon running"."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _record_daemon(tmp_path, pid=child.pid, port=_closed_port())
        status = Daemon(home=tmp_path, in_process=False, health_timeout_s=2.0).status()
    finally:
        child.kill()
        child.wait(timeout=10)

    assert status["state"] == STATE_UNRESPONSIVE
    # ``up`` stays true on purpose: ensure()/start() must still see the pid and
    # refuse to run a second daemon for this home.
    assert status["up"] is True
    assert status["serving"] is False
    assert "not answering" in status["detail"]


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="SIGSTOP is POSIX-only")
def test_a_sigstopped_daemon_stops_reporting_as_running(
    tmp_path: pathlib.Path,
) -> None:
    """The literal report: a daemon that was serving, then was stopped."""
    port = _closed_port()
    child = subprocess.Popen(  # noqa: S603 - fixed script, only the port varies
        [
            sys.executable,
            "-c",
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "class H(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Length', '11')\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b'{\"ok\":true}')\n"
            "    def log_message(self, *a): pass\n"
            f"HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n",
        ]
    )
    try:
        _record_daemon(tmp_path, pid=child.pid, port=port)
        daemon = Daemon(home=tmp_path, in_process=False, health_timeout_s=2.0)
        _wait_until(lambda: daemon.status()["state"] == STATE_RUNNING)

        os.kill(child.pid, signal.SIGSTOP)
        stopped = daemon.status()
    finally:
        try:
            os.kill(child.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        child.kill()
        child.wait(timeout=10)

    assert stopped["state"] == STATE_UNRESPONSIVE
    assert stopped["serving"] is False
    assert stopped["detail"] != "detached daemon running"


def test_ensure_says_so_when_the_daemon_it_reuses_is_not_answering(
    tmp_path: pathlib.Path,
) -> None:
    """``up`` is the right gate for ``ensure`` — a second daemon for one home
    would be refused anyway — but it is not the right report. This handed setup
    a green "daemon already running" step for a daemon serving nothing, and the
    operator found out one command later."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _record_daemon(tmp_path, pid=child.pid, port=_closed_port())
        result = Daemon(home=tmp_path, in_process=False, health_timeout_s=2.0).ensure()
    finally:
        child.kill()
        child.wait(timeout=10)

    assert result.state == SKIPPED
    assert "not answering" in (result.detail or "")
    assert result.metadata["serving"] is False


def test_ensure_reuses_a_serving_daemon_without_qualification(
    serving_daemon: Daemon,
) -> None:
    """The control: the qualification above is not on every reuse."""
    result = serving_daemon.ensure()

    assert result.state == SKIPPED
    assert result.detail == f"daemon already running (pid={os.getpid()})"
    assert result.metadata["serving"] is True


def test_no_process_is_not_running(tmp_path: pathlib.Path) -> None:
    status = Daemon(home=tmp_path, in_process=False).status()

    assert status["state"] == STATE_NOT_RUNNING
    assert status["up"] is False and status["serving"] is False


def _wait_until(predicate, *, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition never became true")


# --- S2-11 / S2-40: logs are bounded, filterable, and followable ------------


def _write_log(home: pathlib.Path, lines: list[str]) -> pathlib.Path:
    log_file = home / "logs" / "potpied.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_file


def test_logs_are_capped_by_default(tmp_path: pathlib.Path) -> None:
    """The old implementation read the whole unrotated file every time."""
    _write_log(tmp_path, [f"line {n}" for n in range(5000)])

    lines = Daemon(home=tmp_path, in_process=False).logs()

    assert len(lines) == 200
    assert lines[0] == "line 4800"
    assert lines[-1] == "line 4999"


def test_logs_honour_an_explicit_tail(tmp_path: pathlib.Path) -> None:
    _write_log(tmp_path, [f"line {n}" for n in range(5000)])

    daemon = Daemon(home=tmp_path, in_process=False)

    assert daemon.logs(tail=3) == ["line 4997", "line 4998", "line 4999"]
    # The whole file stays reachable, but only when asked for by name.
    assert len(daemon.logs(tail=None)) == 5000


def test_logs_tail_spans_more_than_one_read_chunk(tmp_path: pathlib.Path) -> None:
    """The backwards reader walks in 64KiB steps; a tail wider than one step
    must not come back short."""
    padding = "x" * 512
    _write_log(tmp_path, [f"{n} {padding}" for n in range(1000)])

    lines = Daemon(home=tmp_path, in_process=False).logs(tail=400)

    assert len(lines) == 400
    assert lines[0].startswith("600 ")


def test_logs_since_keeps_the_traceback_under_its_header(
    tmp_path: pathlib.Path,
) -> None:
    _write_log(
        tmp_path,
        [
            "2026-08-12 09:00:00,000 INFO potpie [] old news",
            "2026-08-12 11:00:00,000 ERROR potpie [] boom",
            "Traceback (most recent call last):",
            '  File "x.py", line 1, in <module>',
            "RuntimeError: boom",
        ],
    )

    lines = Daemon(home=tmp_path, in_process=False).logs(
        since=datetime(2026, 8, 12, 10, 0, 0)
    )

    assert lines[0].endswith("boom")
    assert "old news" not in "\n".join(lines)
    # The stack belongs to the line above it, not to whatever came before.
    assert lines[-1] == "RuntimeError: boom"


def test_logs_since_reads_the_json_formatter_too(tmp_path: pathlib.Path) -> None:
    _write_log(
        tmp_path,
        [
            json.dumps({"ts": "2026-08-12T09:00:00+0000", "msg": "old"}),
            json.dumps({"ts": "2026-08-12T23:00:00+0000", "msg": "new"}),
        ],
    )

    # Through ``parse_since`` so the cutoff lands in the same frame the offsets
    # above are converted into — otherwise this test only passes in UTC.
    lines = Daemon(home=tmp_path, in_process=False).logs(
        since=parse_since("2026-08-12T12:00:00+00:00")
    )

    assert len(lines) == 1
    assert "new" in lines[0]


def test_follow_yields_lines_appended_after_the_backlog(
    tmp_path: pathlib.Path,
) -> None:
    """``--follow`` used to be accepted and dropped: the flag parsed, the whole
    file printed, and the command exited."""
    log_file = _write_log(tmp_path, ["first"])
    stream = Daemon(home=tmp_path, in_process=False).follow_logs(
        tail=10, poll_interval=0.01
    )

    assert next(stream) == "first"

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("second\n")
    assert next(stream) == "second"

    # Partial writes are held back until the line is whole.
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("thi")
        handle.flush()
        handle.write("rd\n")
    assert next(stream) == "third"
    stream.close()


def test_follow_waits_for_a_log_that_does_not_exist_yet(
    tmp_path: pathlib.Path,
) -> None:
    stream = Daemon(home=tmp_path, in_process=False).follow_logs(poll_interval=0.02)
    produced: list[str] = []

    def _consume() -> None:
        produced.append(next(stream))

    # The reader exits on its first line, leaving the generator suspended — so
    # nothing here outlives the test whether or not the assertion holds.
    reader = threading.Thread(target=_consume, daemon=True)
    reader.start()
    time.sleep(0.1)
    _write_log(tmp_path, ["late arrival"])
    reader.join(timeout=10)

    assert produced == ["late arrival"]


@pytest.mark.parametrize(
    "text, delta",
    [
        ("30s", timedelta(seconds=30)),
        ("15m", timedelta(minutes=15)),
        ("2h", timedelta(hours=2)),
        ("1d", timedelta(days=1)),
    ],
)
def test_parse_since_accepts_relative_ages(text: str, delta: timedelta) -> None:
    parsed = parse_since(text)

    assert abs((datetime.now() - delta) - parsed) < timedelta(seconds=5)


def test_parse_since_accepts_iso_and_refuses_nonsense() -> None:
    assert parse_since("2026-08-12T09:00:00") == datetime(2026, 8, 12, 9, 0, 0)

    with pytest.raises(ValueError, match="cannot read 'yesterday' as a time"):
        parse_since("yesterday")
