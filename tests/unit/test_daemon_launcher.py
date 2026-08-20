from __future__ import annotations

import contextlib
import json
import pathlib
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from potpie.cli.telemetry.preferences import (
    TelemetryPreferences,
    save_preferences,
)
from potpie.daemon.process import launcher
from potpie_context_engine.bootstrap import runtime_settings


class _HealthHandler(BaseHTTPRequestHandler):
    """A real server answering ``/health`` the way the daemon does."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        body = json.dumps(
            {"ok": True, "mode": "daemon", "pid": self.server.daemon_pid}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # keep pytest output clean
        return


@contextlib.contextmanager
def _daemon_serving(pid: int) -> Iterator[str]:
    """A live address that answers ``/health`` as the daemon with ``pid``."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    server.daemon_pid = pid
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def _daemon_serving_on_socket(pid: int) -> Iterator[pathlib.Path]:
    """The same, over a unix socket. Its own short temp dir, not ``tmp_path``:
    an ``AF_UNIX`` path is capped near 104 bytes and pytest's is longer."""
    server_type = getattr(socketserver, "ThreadingUnixStreamServer", None)
    if server_type is None:
        pytest.skip("Unix-domain socket server is unavailable on this platform")

    class _UnixHealthServer(server_type):
        def get_request(self):
            request, _client = super().get_request()
            # BaseHTTPRequestHandler indexes client_address; a UDS peer has none.
            return request, ("localhost", 0)

    with tempfile.TemporaryDirectory() as short_dir:
        path = pathlib.Path(short_dir) / "daemon.sock"
        server = _UnixHealthServer(str(path), _HealthHandler)
        server.daemon_pid = pid
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            yield path
        finally:
            server.shutdown()
            server.server_close()


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(autouse=True)
def _clear_runtime_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("POTPIE_ENVIRONMENT", "test")
    monkeypatch.delenv("POTPIE_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setattr(runtime_settings, "load_distribution_defaults", lambda: {})


def test_start_detached_enables_telemetry_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    env = _captured_child_env(monkeypatch, tmp_path)

    assert env["POTPIE_TELEMETRY_DISABLED"] == "0"


def test_start_detached_applies_persisted_cli_telemetry_preference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    save_preferences(TelemetryPreferences(enabled=False))

    env = _captured_child_env(monkeypatch, tmp_path)

    assert env["POTPIE_TELEMETRY_DISABLED"] == "1"


def test_start_detached_env_block_wins_over_persisted_enable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setenv("POTPIE_TELEMETRY_DISABLED", "1")
    save_preferences(TelemetryPreferences(enabled=True))

    env = _captured_child_env(monkeypatch, tmp_path)

    assert env["POTPIE_TELEMETRY_DISABLED"] == "1"


# --- S2-27: an address that cannot be read is not a started daemon ----------


class _WritesDiscovery:
    """A ``Popen`` stand-in that writes whatever the daemon would have written.

    The process launch is the outermost adapter and the only thing faked here;
    the discovery file, its contents and the read that consumes them are all
    real, because that read is where the defect lived.
    """

    pid = 4243

    def __init__(self, contents: str | None) -> None:
        self._contents = contents

    def __call__(self, _args, *, env, **_kwargs):
        if self._contents is not None:
            home = pathlib.Path(env["CONTEXT_ENGINE_HOME"])
            home.mkdir(parents=True, exist_ok=True)
            (home / "discovery.json").write_text(self._contents, encoding="utf-8")
        return self

    def poll(self) -> None:
        return None


@pytest.mark.parametrize(
    "contents, expected",
    [
        ("{not json at all", "not readable JSON"),
        ('["a", "list"]', "does not hold a JSON object"),
        ('{"transport": "http"}', "names neither a socket nor a URL"),
        ('{"bind": "", "base_url": ""}', "names neither a socket nor a URL"),
    ],
    ids=["torn", "wrong-type", "no-address", "empty-address"],
)
def test_an_unreadable_discovery_file_fails_the_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    contents: str,
    expected: str,
) -> None:
    """This was swallowed to ``{}`` and reported as a *started* daemon with an
    empty URL, so the failure surfaced later, somewhere unrelated, against a
    daemon ``start`` had just said was up."""
    monkeypatch.setattr(launcher.subprocess, "Popen", _WritesDiscovery(contents))
    killed: list[int] = []
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: killed.append(pid))

    with pytest.raises(launcher.DaemonStartError) as raised:
        launcher.start_detached(tmp_path / "home", ready_timeout_s=0.4)

    assert "address could not be read" in str(raised.value)
    assert expected in str(raised.value)
    assert raised.value.log_path is not None
    # A half-up daemon is not left behind just because we refused its address.
    assert killed == [_WritesDiscovery.pid]


def test_a_readable_discovery_file_still_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The control: the refusal above must not have made every start fail."""
    with _daemon_serving(_WritesDiscovery.pid) as url:
        monkeypatch.setattr(
            launcher.subprocess,
            "Popen",
            _WritesDiscovery(
                json.dumps({"bind": "unix:/tmp/potpie.sock", "base_url": url})
            ),
        )

        result = launcher.start_detached(tmp_path / "home", ready_timeout_s=5)

    assert result["socket"] == "/tmp/potpie.sock"  # noqa: S108 - a fixture path, not a file
    assert result["url"] == url


def test_a_torn_read_that_completes_in_time_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The daemon truncates and rewrites this file, so a read can land
    mid-write. That must resolve into a success, not a refusal."""
    home = tmp_path / "home"

    class _WritesLate:
        pid = 4244

        def __init__(self, url: str) -> None:
            self._url = url
            self._reads = 0

        def __call__(self, _args, *, env, **_kwargs):
            self._home = pathlib.Path(env["CONTEXT_ENGINE_HOME"])
            self._home.mkdir(parents=True, exist_ok=True)
            (self._home / "discovery.json").write_text('{"base_ur', encoding="utf-8")
            return self

        def poll(self):
            self._reads += 1
            if self._reads == 2:
                (self._home / "discovery.json").write_text(
                    json.dumps({"base_url": self._url}), encoding="utf-8"
                )
            return None

    with _daemon_serving(_WritesLate.pid) as url:
        monkeypatch.setattr(launcher.subprocess, "Popen", _WritesLate(url))

        result = launcher.start_detached(home, ready_timeout_s=5)

    assert result["url"] == url


# --- S2-20 / S2-21 / S2-25: "started" has to mean the daemon is serving -----
#
# Reproduced against the real CLI before any of this existed: with the daemon's
# port already held by another process, ``potpie daemon start`` printed
# ``daemon started (pid=50538)`` and exited 0, while logs/potpied.log recorded
# ``[Errno 48] ... address already in use`` two lines later.
#
# The reason a file could say that: the daemon publishes its address from the
# ASGI lifespan, and uvicorn runs the lifespan BEFORE it binds the listening
# socket. So the file the launcher waited for is written by a daemon that has
# not yet discovered it cannot have the port. Everything below therefore runs
# against real sockets — a stubbed probe would test the assumption that made
# the bug.


def test_a_published_address_that_never_answers_is_not_a_started_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        _WritesDiscovery(
            json.dumps({"base_url": f"http://127.0.0.1:{_closed_port()}"})
        ),
    )
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: None)

    with pytest.raises(launcher.DaemonStartError) as raised:
        launcher.start_detached(tmp_path / "home", ready_timeout_s=0.4)

    assert "nothing answered /health" in str(raised.value)


def test_an_address_served_by_a_stranger_is_not_the_daemon_we_started(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Why readiness cannot be "can I connect to it?".

    The failure this is all about is a port that was *already taken*, so in the
    one case that matters something is listening on the published address and
    answering happily. Only the pid it names tells the two apart.
    """
    with _daemon_serving(999999) as url:
        monkeypatch.setattr(
            launcher.subprocess,
            "Popen",
            _WritesDiscovery(json.dumps({"base_url": url})),
        )
        monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: None)

        with pytest.raises(launcher.DaemonStartError) as raised:
            launcher.start_detached(tmp_path / "home", ready_timeout_s=0.4)

    assert (
        f"served by pid 999999, not the daemon just started "
        f"(pid {_WritesDiscovery.pid})"
    ) in str(raised.value)


def test_a_dead_child_is_not_started_however_good_its_discovery_file_looks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The exit is read before the file is believed.

    A daemon that loses the bind race publishes its address and *then* dies, so
    for a moment there is a complete, serving-looking discovery file belonging
    to a process that no longer exists.
    """
    home = tmp_path / "home"

    with _daemon_serving(4247) as url:

        class _PublishesThenDies:
            pid = 4247

            def __call__(self, _args, *, env, **_kwargs):
                child_home = pathlib.Path(env["CONTEXT_ENGINE_HOME"])
                child_home.mkdir(parents=True, exist_ok=True)
                (child_home / "discovery.json").write_text(
                    json.dumps({"base_url": url}), encoding="utf-8"
                )
                return self

            def poll(self) -> int:
                return 1

        monkeypatch.setattr(launcher.subprocess, "Popen", _PublishesThenDies())

        with pytest.raises(launcher.DaemonStartError) as raised:
            launcher.start_detached(home, ready_timeout_s=5)

    assert "daemon failed to start (exit 1)" in str(raised.value)


def test_a_stale_discovery_file_is_cleared_before_the_child_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A SIGKILLed daemon never removes its discovery file.

    Left in place, the first read of the readiness loop finds the *dead*
    daemon's address and reports it as this child's, before the child has
    finished importing.
    """
    home = tmp_path / "home"
    home.mkdir(parents=True)
    (home / "discovery.json").write_text(
        json.dumps({"base_url": "http://127.0.0.1:9"}), encoding="utf-8"
    )
    seen: dict[str, bool] = {}

    with _daemon_serving(4246) as url:

        class _LooksForLeftovers:
            pid = 4246

            def __call__(self, _args, *, env, **_kwargs):
                child_home = pathlib.Path(env["CONTEXT_ENGINE_HOME"])
                seen["stale"] = (child_home / "discovery.json").exists()
                (child_home / "discovery.json").write_text(
                    json.dumps({"base_url": url}), encoding="utf-8"
                )
                return self

            def poll(self) -> None:
                return None

        monkeypatch.setattr(launcher.subprocess, "Popen", _LooksForLeftovers())

        result = launcher.start_detached(home, ready_timeout_s=5)

    assert seen["stale"] is False
    assert result["url"] == url


def test_a_daemon_on_a_unix_socket_is_probed_there(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The other address shape, probed over the socket rather than a port."""
    with _daemon_serving_on_socket(_WritesDiscovery.pid) as sock_path:
        monkeypatch.setattr(
            launcher.subprocess,
            "Popen",
            _WritesDiscovery(json.dumps({"bind": f"unix:{sock_path}"})),
        )

        result = launcher.start_detached(tmp_path / "home", ready_timeout_s=5)

    assert result["socket"] == str(sock_path)


def test_a_dead_unix_socket_is_not_a_started_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The control on the branch above: a socket path nothing is bound to."""
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        _WritesDiscovery(json.dumps({"bind": f"unix:{tmp_path / 'gone.sock'}"})),
    )
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: None)

    with pytest.raises(launcher.DaemonStartError) as raised:
        launcher.start_detached(tmp_path / "home", ready_timeout_s=0.4)

    assert "nothing answered /health" in str(raised.value)


# --- the cause, not a generic exit code -------------------------------------


class _DiesOnBind:
    """A ``Popen`` stand-in that writes what uvicorn writes when it cannot bind.

    Including the chatter that follows: the daemon logs its shutdown, and the
    telemetry flush logs after that, so the *last* lines of the log are the
    least informative part of it.
    """

    pid = 4248
    returncode = 1
    TAIL = (
        "INFO:     Started server process [4248]\n"
        "INFO:     Waiting for application startup.\n"
        "INFO:     Application startup complete.\n"
        "ERROR:    [Errno 48] error while attempting to bind on address "
        "('127.0.0.1', 8123): address already in use\n"
        "INFO:     Waiting for application shutdown.\n"
        "INFO:     Application shutdown complete.\n"
        "Sentry is attempting to send 2 pending events\n"
        "Press Ctrl-C to quit\n"
    )

    def __init__(self, tail: str | None = None) -> None:
        self._tail = self.TAIL if tail is None else tail

    def __call__(self, _args, *, env, **_kwargs):
        log = pathlib.Path(env["CONTEXT_ENGINE_HOME"]) / "logs" / "potpied.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(self._tail)
        return self

    def poll(self) -> int:
        return 1


def test_a_child_that_dies_is_reaped_at_once_and_reports_its_own_words(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """``daemon failed to start (exit 1)`` is true and useless: a taken port, a
    missing dependency and a bad config all read the same. And waiting out the
    readiness deadline for a process that is already gone turns a five-second
    answer into a sixty-second one."""
    monkeypatch.setattr(launcher.subprocess, "Popen", _DiesOnBind())

    started = time.monotonic()
    with pytest.raises(launcher.DaemonStartError) as raised:
        launcher.start_detached(tmp_path / "home", ready_timeout_s=30)
    elapsed = time.monotonic() - started

    message = str(raised.value)
    assert "daemon failed to start (exit 1)" in message
    assert "address already in use" in message
    # The tail of the log is not the cause; the line naming the fault is.
    assert "Ctrl-C" not in message
    assert elapsed < 5, "a dead child waited out the readiness deadline"
    assert raised.value.log_path == tmp_path / "home" / "logs" / "potpied.log"


def test_the_previous_run_s_crash_is_not_reported_as_this_one_s(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Nothing rotates this log, so it opens holding every earlier run. Quoting
    the last error in the file would blame this start for last week's."""
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "potpied.log").write_text(
        "ERROR:    [Errno 48] address already in use (a start from last week)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        _DiesOnBind("Traceback (most recent call last):\nRuntimeError: no backend\n"),
    )

    with pytest.raises(launcher.DaemonStartError) as raised:
        launcher.start_detached(home, ready_timeout_s=30)

    assert "RuntimeError: no backend" in str(raised.value)
    assert "last week" not in str(raised.value)


# --- S2-29: never signal a process that is not ours -------------------------


def test_stop_refuses_to_signal_an_unrelated_live_process(
    tmp_path: pathlib.Path,
) -> None:
    """A recycled pid used to get SIGTERM and then SIGKILL. The pid here is a
    real, live process that is provably not a Potpie daemon."""
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text(f"{stranger.pid}\n")
    try:
        message = launcher.stop_daemon(tmp_path)
        still_alive = stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait(timeout=10)

    assert still_alive, "stop_daemon signalled a process it did not own"
    assert "unrelated process" in message
    # The pid file named a process that is not ours, so it was stale by
    # definition and removing it is what unblocks the next start.
    assert not pid_file.exists()


def test_identity_is_read_off_the_real_command_line() -> None:
    """Three answers, read from real processes.

    The daemon-shaped one carries the entrypoint in its argv exactly as
    ``start_detached`` spells it, so the check must recognise it — a check that
    only ever said "not ours" would refuse to stop any daemon at all.
    """
    daemon_like = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "-m",
            "potpie.daemon.main",
        ]
    )
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert launcher.is_daemon_process(daemon_like.pid) is True
        assert launcher.is_daemon_process(stranger.pid) is False
    finally:
        for proc in (daemon_like, stranger):
            proc.kill()
            proc.wait(timeout=10)


def _stopping_kill(pid_file: pathlib.Path, signalled: list[int]):
    """Stand in for a daemon that honours SIGTERM: it drops its pid file."""

    def _kill(pid: int, _sig: int) -> None:
        signalled.append(pid)
        pid_file.unlink(missing_ok=True)

    return _kill


def test_a_pid_that_cannot_be_identified_is_left_to_the_old_behaviour(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """``None`` means "the machine will not say", and the absence of proof is
    never treated as proof of a stranger — otherwise a platform without a
    readable command line could never stop its own daemon."""
    monkeypatch.setattr(launcher, "_process_command_line", lambda pid: None)
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("424242\n")
    signalled: list[int] = []
    monkeypatch.setattr(launcher.os, "kill", _stopping_kill(pid_file, signalled))

    assert launcher.stop_daemon(tmp_path) == "daemon stopped"
    assert signalled == [424242]


def test_windows_access_denied_pid_is_treated_as_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(launcher.os, "name", "nt")
    monkeypatch.setattr(launcher, "_process_command_line", lambda pid: None)
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("424242\n")

    def _access_denied(_pid: int, _sig: int) -> None:
        error = OSError("access denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(launcher.os, "kill", _access_denied)

    assert launcher.stop_daemon(tmp_path) == "stale pid file removed"
    assert not pid_file.exists()


def test_a_recognised_daemon_is_still_signalled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(
        launcher,
        "_process_command_line",
        lambda pid: "/usr/bin/python3 -m potpie.daemon.main",
    )
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("424243\n")
    signalled: list[int] = []
    monkeypatch.setattr(launcher.os, "kill", _stopping_kill(pid_file, signalled))

    assert launcher.stop_daemon(tmp_path) == "daemon stopped"
    assert signalled == [424243]


def test_force_kill_uses_taskkill_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher.os, "name", "nt")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def _run(args: list[str], **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(launcher.subprocess, "run", _run)

    launcher._force_kill(424244)

    assert calls == [
        (
            ["taskkill", "/PID", "424244", "/T", "/F"],
            {
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
                "check": False,
                "creationflags": getattr(launcher.subprocess, "CREATE_NO_WINDOW", 0),
            },
        )
    ]


def _captured_child_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> dict[str, str]:
    captured: dict[str, dict[str, str]] = {}

    with _daemon_serving(4242) as url:

        class _FakePopen:
            pid = 4242

            def __init__(
                self,
                _args: list[str],
                *,
                stdin: object,
                stdout: object,
                stderr: int,
                start_new_session: bool = False,
                close_fds: bool,
                env: dict[str, str],
                creationflags: int = 0,
            ) -> None:
                del stdin, stdout, stderr, start_new_session, close_fds, creationflags
                captured["env"] = env
                home = pathlib.Path(env["CONTEXT_ENGINE_HOME"])
                (home / "discovery.json").write_text(
                    json.dumps({"bind": "unix:/tmp/potpie.sock", "base_url": url}),
                    encoding="utf-8",
                )

            def poll(self) -> None:
                return None

        monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

        result = launcher.start_detached(tmp_path / "home", ready_timeout_s=5)

    assert result["pid"] == 4242
    return captured["env"]
