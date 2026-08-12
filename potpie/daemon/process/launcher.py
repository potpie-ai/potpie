"""Detached daemon launcher: start the daemon as a background process and block until ready, or stop it.

This is the reliable start/stop mechanism the ``potpie.daemon.lifecycle.Daemon`` seam
drives. ``start_detached`` is QUIET — it returns daemon discovery metadata once
the daemon answers on the address it published, or raises ``DaemonStartError``;
it never prints, so it is safe to call from inside a rich/live setup UI.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Final

from potpie.daemon.process.pidfile import read_pid_file
from potpie_context_engine.bootstrap.runtime_settings import (
    RuntimeSettings,
    project_child_environment,
)

#: What a Potpie daemon's command line always contains, because :func:`start_detached`
#: is the only thing that starts one and it always spells the module out.
_DAEMON_ENTRYPOINT = "potpie.daemon.main"

#: How long one readiness probe waits for ``/health``. The daemon is on
#: loopback, so this is generous; it stays short because every probe that runs
#: out the clock is time the crash check below does not get to run.
_PROBE_TIMEOUT_S: Final[float] = 2.0

#: How far back into the daemon log a failed start reads for the cause.
_LOG_TAIL_BYTES: Final[int] = 64 * 1024

#: How much of that line is quoted back before it stops being a message.
_MAX_CAUSE_CHARS: Final[int] = 400

#: Words that mark the line a failed start is actually about. The daemon logs
#: its shutdown chatter (and a telemetry flush) *after* whatever killed it, so
#: the plain last line of the log is usually the least informative one in it.
_FAULT_MARKERS: Final[tuple[str, ...]] = (
    "error",
    "traceback",
    "exception",
    "critical",
    "fatal",
    "refused",
    "denied",
)


class DaemonStartError(Exception):
    """Raised by start_detached() when the daemon does not come up.

    Carries the daemon log path (when available) so callers can surface the cause.
    """

    def __init__(self, message: str, *, log_path: "pathlib.Path | None" = None) -> None:
        super().__init__(message)
        self.log_path = log_path


def start_detached(
    home: pathlib.Path,
    *,
    ready_timeout_s: float = 60.0,
    backend: str | None = None,
) -> dict:
    """Start the daemon detached for ``home`` and block until it is fully serving.

    Returns discovery metadata once the daemon *answers* on the address it
    published, or raises :class:`DaemonStartError` on any failure (already
    running, child crash, an unreadable address, or a readiness timeout).

    Readiness deliberately costs a round trip. "The discovery file appeared" is
    not readiness: the daemon publishes its address from the ASGI lifespan,
    which uvicorn runs *before* it binds the listening socket, so a daemon whose
    port is already taken publishes an address, fails to bind, and exits — and
    this used to hand that address back as ``daemon started (pid=…)`` at exit 0.
    """
    home = pathlib.Path(home)
    pid_file = home / "daemon.pid"
    disc_file = home / "discovery.json"
    log_path = home / "logs" / "potpied.log"
    if pid_file.exists():
        existing = read_pid_file(pid_file)
        if existing:
            try:
                os.kill(existing, 0)
                raise DaemonStartError(f"daemon already running (pid={existing})")
            except ProcessLookupError:
                pid_file.unlink()  # stale
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Only the child started below gets to signal readiness. A daemon killed
    # with SIGKILL never removed its discovery file, and the dead daemon's
    # address read as this one's returns "started" before the new child has
    # finished importing.
    _unlink(disc_file)
    # Where this child's own output begins. The log is opened for append and
    # nothing rotates it, so everything before this offset belongs to earlier
    # runs; quoting that back as the cause of *this* failure would be a lie.
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    log_fp = log_path.open("a")
    child_env = project_child_environment(
        _load_daemon_child_runtime_settings(),
        os.environ,
        overrides={
            "CONTEXT_ENGINE_HOME": str(home),
            **({"CONTEXT_ENGINE_BACKEND": backend} if backend else {}),
        },
    )
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "potpie.daemon.main"],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=child_env,
        )
    finally:
        log_fp.close()
    deadline = time.time() + ready_timeout_s
    # Why these are remembered rather than raised on sight: the daemon truncates
    # and rewrites discovery.json, so a read can land mid-write, and a probe can
    # land before the socket is accepting. Retrying until the deadline turns
    # either into a slightly later success; only a fault that is *still* there
    # when time runs out is a real failure — and then it is the fault that gets
    # reported, not a generic timeout.
    discovery_fault: str | None = None
    serving_fault: str | None = None
    while True:
        # Reaped first, and read second. A child that has already exited cannot
        # become ready, so nothing it left behind on the way down — least of all
        # the address it published before failing to bind — counts as a start.
        exited = proc.poll()
        if disc_file.exists():
            address, discovery_fault = _read_discovery(disc_file)
            if address is not None and exited is None:
                serving_fault = _serving_fault(address, proc.pid)
                if serving_fault is None:
                    return {"pid": proc.pid, **address}
        if exited is not None:
            raise DaemonStartError(
                _crash_message(exited, log_path, log_offset), log_path=log_path
            )
        if time.time() >= deadline:
            break
        time.sleep(0.1)
    # Alive but never served — stop it so we don't leave a half-up daemon.
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if serving_fault is not None:
        raise DaemonStartError(
            f"daemon did not become ready within {int(ready_timeout_s)}s: "
            f"{serving_fault}",
            log_path=log_path,
        )
    if discovery_fault is not None:
        # The failure this replaces: an unreadable discovery file was swallowed
        # to ``{}`` and reported as a *started* daemon with an empty URL, so the
        # very next command failed somewhere else entirely, against a daemon
        # `start` had just said was up.
        raise DaemonStartError(
            f"daemon started but its address could not be read: {discovery_fault}",
            log_path=log_path,
        )
    raise DaemonStartError(
        f"daemon did not become ready within {int(ready_timeout_s)}s", log_path=log_path
    )


def _read_discovery(disc_file: pathlib.Path) -> tuple[dict | None, str | None]:
    """``(address, None)`` once the file is complete, else ``(None, why not)``."""
    try:
        disc = json.loads(disc_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{disc_file} is not readable JSON ({exc})"
    if not isinstance(disc, dict):
        return None, f"{disc_file} does not hold a JSON object"
    bind = str(disc.get("bind") or "")
    base_url = str(disc.get("base_url") or "")
    if not bind and not base_url:
        return None, f"{disc_file} names neither a socket nor a URL"
    socket_path = bind[len("unix:") :] if bind.startswith("unix:") else bind
    return {"socket": socket_path, "bind": bind, "url": base_url}, None


def _serving_fault(address: dict[str, str], pid: int) -> str | None:
    """``None`` once the daemon answers on its own address, else why it has not.

    Two things have to be true and only one of them is a file. The address must
    answer — the daemon publishes it before uvicorn binds, so publishing it
    proves nothing — and the thing answering must be *this* daemon. The second
    half is not paranoia: the failure this exists for is a port that was already
    taken, which means the one case where an address is published and never
    served is exactly the case where something else is listening on it.
    ``/health`` names the pid it is served by, so the check is a comparison.
    """
    label = address.get("url") or address.get("bind") or "the published address"
    client = _probe_client(address)
    if client is None:
        # A discovery shape with no address this can reach. Nothing writes one
        # today; take it at its word rather than refuse a start over it.
        return None
    try:
        with client:
            response = client.get("/health", timeout=_PROBE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 - every way of not answering is one answer
        return f"nothing answered /health at {label} ({exc.__class__.__name__})"
    if not 200 <= response.status_code < 300:
        return f"/health at {label} answered HTTP {response.status_code}"
    try:
        served = response.json()
    except ValueError:
        return f"{label} is not answering as a Potpie daemon"
    served_pid = served.get("pid") if isinstance(served, dict) else None
    if isinstance(served_pid, int) and served_pid != pid:
        return (
            f"{label} is served by pid {served_pid}, not the daemon just "
            f"started (pid {pid})"
        )
    return None


def _probe_client(address: dict[str, str]):
    """An ``httpx`` client aimed at the daemon's own address, or ``None``.

    The two shapes a Potpie daemon publishes: a loopback URL (what
    ``potpie.daemon.main`` writes) and a unix socket. ``httpx`` is imported here
    rather than at module scope because this module is on the path of every CLI
    invocation and the probe runs at most once per start.
    """
    import httpx

    url = address.get("url") or ""
    bind = address.get("bind") or ""
    if url:
        return httpx.Client(base_url=url)
    if bind.startswith("unix:"):
        return httpx.Client(
            transport=httpx.HTTPTransport(uds=bind[len("unix:") :]),
            base_url="http://localhost",
        )
    return None


def _crash_message(returncode: int, log_path: pathlib.Path, offset: int) -> str:
    """Why the daemon exited, in its own words wherever it left any.

    ``daemon failed to start (exit 1)`` is true and useless: a taken port, a
    missing dependency and an unreadable config all read the same, and the
    operator is sent to go and find the difference. The child wrote the reason
    to the daemon log on its way out, so the start that reaps it quotes it back.
    """
    cause = _fault_line(log_path, offset)
    exited = f"daemon failed to start (exit {returncode})"
    return f"{exited}: {cause}" if cause else exited


def _fault_line(log_path: pathlib.Path, offset: int) -> str | None:
    """The line naming the fault, out of what *this* child wrote to the log."""
    try:
        with log_path.open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            start = max(offset, end - _LOG_TAIL_BYTES)
            handle.seek(start)
            written = handle.read(max(end - start, 0))
    except OSError:
        return None
    text = written.decode("utf-8", errors="replace")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    if start > offset and lines:
        lines.pop(0)  # a line the byte cap cut in half is not evidence
    lines = [line for line in lines if line]
    if not lines:
        return None
    for line in reversed(lines):
        lowered = line.lower()
        if any(marker in lowered for marker in _FAULT_MARKERS):
            return _clip(line)
    return _clip(lines[-1])


def _clip(line: str) -> str:
    if len(line) <= _MAX_CAUSE_CHARS:
        return line
    return line[: _MAX_CAUSE_CHARS - 3] + "..."


def stop_daemon(home: pathlib.Path) -> str:
    """Stop the daemon if running. Returns a human-readable message; never raises."""
    home = pathlib.Path(home)
    pid_file = home / "daemon.pid"
    pid = read_pid_file(pid_file)
    if not pid:
        return "daemon not running"
    if is_daemon_process(pid) is False:
        # The pid file outlived its daemon and the number came back around to
        # something else. SIGTERM followed by SIGKILL is not a recoverable
        # mistake to make against a stranger's process, and a pid file naming a
        # process that is provably not ours is by definition stale.
        _unlink(pid_file)
        return (
            f"stale pid file removed; pid {pid} belongs to an unrelated process "
            "and was left alone"
        )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _unlink(pid_file)
        return "stale pid file removed"
    deadline = time.time() + 10
    while time.time() < deadline:
        if not pid_file.exists():
            return "daemon stopped"
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _unlink(pid_file)
    return "daemon killed (forced after timeout)"


def is_daemon_process(pid: int) -> bool | None:
    """Is ``pid`` a Potpie daemon? ``None`` when the machine will not say.

    Three answers, not two, and the third is the important one. Callers signal
    on ``False`` being *proof* of a stranger — never on the absence of proof —
    so a platform where the command line cannot be read behaves exactly as it
    did before this check existed, and a live daemon whose command line we
    misread is never mistaken for a leftover.
    """
    command = _process_command_line(pid)
    if command is None:
        return None
    return _DAEMON_ENTRYPOINT in command


def _process_command_line(pid: int) -> str | None:
    """The process's argv as one string, or ``None`` if it cannot be read."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil ships with redislite
        return _process_command_line_via_ps(pid)
    try:
        return " ".join(psutil.Process(pid).cmdline())
    except Exception:  # noqa: BLE001 - exited, denied, or zombie: all "cannot say"
        return None


def _process_command_line_via_ps(pid: int) -> str | None:  # pragma: no cover - fallback
    try:
        # A fixed argv of literals plus an integer pid, resolved on PATH because
        # ``ps`` lives in different places across the POSIX systems this has to
        # run on. Nothing in the command line comes from a caller.
        completed = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "args="],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _unlink(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _load_daemon_child_runtime_settings() -> RuntimeSettings:
    from potpie.cli.telemetry.settings import load_cli_runtime_settings

    return load_cli_runtime_settings()
