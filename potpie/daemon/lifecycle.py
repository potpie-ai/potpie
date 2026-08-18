"""``Daemon`` - local host lifecycle.

The daemon shell is the local background process for lifecycle, IPC, health,
and logs. It is not the business layer. When ``in_process`` is true, the host
runs in the CLI process and reports synthetic liveness. When detached, the
daemon process runs ``potpie.daemon.main`` and serves HostShell RPC over loopback
HTTP.

Liveness and readiness are separate: the daemon can be live while a backend or
semantic index is not ready. *Existing* and *serving* are separate too: a
process that has been SIGSTOPped, or that is wedged inside a call it will never
return from, still owns its pid and still answers ``os.kill(pid, 0)``. Reporting
that as "running" is the answer an operator is least able to act on, so
:meth:`Daemon.status` probes the health endpoint and reports the two facts
apart — see :data:`STATE_RUNNING` and friends.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Iterator

from potpie_context_engine.adapters.outbound.pots.local_pot_store import default_home
from potpie_context_core.lifecycle import DONE, SKIPPED, SetupPlan, StepResult

#: The daemon is answering. The only state a liveness check may treat as good.
STATE_RUNNING: Final[str] = "running"

#: No process — nothing to talk to, and nothing wrong with the machine either.
STATE_NOT_RUNNING: Final[str] = "not_running"

#: A process exists and does not answer: SIGSTOPped, wedged, or still booting.
#: Distinct from :data:`STATE_NOT_RUNNING` because the repair is different —
#: ``daemon start`` will refuse (the pid is live), so this one needs a restart.
STATE_UNRESPONSIVE: Final[str] = "unresponsive"

#: The in-process stand-in: no detached daemon exists and none is wanted.
STATE_IN_PROCESS: Final[str] = "in_process"

#: How many lines :meth:`Daemon.logs` returns when the caller does not say.
#:
#: The daemon log is append-only and nothing rotates it, so it grows for as long
#: as the machine has been up. The previous implementation did a whole-file
#: ``read_text()`` and returned every line — on a long-lived daemon that is tens
#: of megabytes materialised as a Python list, JSON-encoded, and printed.
DEFAULT_LOG_TAIL_LINES: Final[int] = 200

#: Chunk the backwards tail read walks. Only enough of the end of the file to
#: hold ``tail`` newlines is ever read.
_TAIL_CHUNK_BYTES: Final[int] = 64 * 1024

#: How often ``--follow`` looks for new bytes. Short enough to feel live, long
#: enough that tailing an idle daemon costs nothing.
FOLLOW_POLL_SECONDS: Final[float] = 0.4

#: ``2026-08-12 11:21:33,123`` (stdlib ``asctime``) or an ISO-8601 ``T``.
_PLAIN_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:[.,](\d+))?"
)

#: ``--since 10m`` — the shape anyone reaching for a log actually wants.
_RELATIVE_SINCE = re.compile(r"^(\d+)\s*([smhd])$", re.IGNORECASE)

_RELATIVE_SINCE_UNITS: Final[dict[str, str]] = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError, SystemError):
        return False
    return True


@dataclass
class Daemon:
    """Local daemon lifecycle: in-process stand-in or detached process."""

    home: Path = field(default_factory=default_home)
    in_process: bool = True
    startup_timeout_s: float = 60.0
    #: How long the ``/health`` probe waits before calling the daemon wedged.
    #: A SIGSTOPped process still completes the TCP handshake — the kernel
    #: accepts into its backlog — so "unresponsive" can only ever be a timeout.
    health_timeout_s: float = 3.0

    def discovery(self) -> dict[str, str] | None:
        """Return daemon discovery metadata for either supported local daemon."""
        from potpie.daemon.process.ipc_client import load_discovery

        discovery = load_discovery(self.home)
        if discovery is not None:
            return discovery
        legacy_path = self.home / "daemon.json"
        if not legacy_path.exists():
            return None
        try:
            raw = json.loads(legacy_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict):
            return None
        return {str(key): str(value) for key, value in raw.items()}

    def status(self) -> dict[str, Any]:
        """Liveness, as three distinguishable states rather than one boolean.

        ``up`` still means "a process owns the pid file", because that is what
        :meth:`ensure` and every start/stop race must key on — two daemons for
        one home is the failure that matters there, and a wedged one still
        counts. ``serving`` is the fact a liveness check needs: the process
        answered ``/health``. A SIGSTOPped daemon is ``up`` and not ``serving``,
        which is exactly the case that used to report "detached daemon running".
        """
        if self.in_process:
            return {
                "up": True,
                "serving": True,
                "state": STATE_IN_PROCESS,
                "mode": "in_process",
                "home": str(self.home),
                "detail": "in-process host; no detached daemon",
            }
        from potpie.daemon.process.pidfile import read_pid_file

        pid = read_pid_file(self.home / "daemon.pid")
        discovery = self.discovery() or {}
        bind = discovery.get("bind", "")
        socket = bind.removeprefix("unix:") if bind.startswith("unix:") else bind
        base_url = discovery.get("base_url", "")
        # One probe, two answers: whether it is serving, and what backend it
        # serves. Only paid for when a process actually exists.
        up = bool(pid and _pid_alive(pid))
        health = self.health() if (up or base_url) else {}
        health_pid = health.get("pid")
        if (
            not up
            and pid
            and health.get("live")
            and health_pid is not None
            and str(health_pid) == str(pid)
        ):
            # On Windows, signal-zero can reject a detached Python process even
            # while its health endpoint is serving normally.
            up = True
        serving = bool(health.get("live"))
        if not up:
            state = STATE_NOT_RUNNING
        elif serving:
            state = STATE_RUNNING
        else:
            state = STATE_UNRESPONSIVE
        status = {
            "up": up,
            "serving": serving,
            "state": state,
            "mode": "detached",
            "home": str(self.home),
            "pid": pid,
            "detail": _STATE_DETAIL[state].format(pid=pid),
        }
        if socket:
            status["socket"] = socket
        if base_url:
            status["url"] = base_url
        if "backend" in health:
            status["backend"] = health["backend"]
        return status

    def health(self) -> dict[str, Any]:
        if self.in_process:
            return {"live": True, "mode": "in_process"}
        discovery = self.discovery() or {}
        base_url = discovery.get("base_url")
        if base_url:
            try:
                import httpx

                response = httpx.get(
                    f"{base_url.rstrip('/')}/health", timeout=self.health_timeout_s
                )
                data = response.json()
                return {
                    "live": 200 <= response.status_code < 300,
                    "mode": "detached",
                    **data,
                }
            except Exception:  # noqa: BLE001 - daemon health must be best-effort.
                return {"live": False, "mode": "detached"}
        from potpie.daemon.process.ipc_client import client_for

        try:
            with client_for(self.home) as client:
                response = client.get("/admin/health")
                return {
                    "live": 200 <= response.status_code < 300,
                    "mode": "detached",
                    **response.json(),
                }
        except Exception:  # noqa: BLE001 - daemon health must be best-effort.
            return {"live": False, "mode": "detached"}

    def log_path(self) -> Path | None:
        """The log this daemon writes, or ``None`` if it has not written one."""
        for name in ("logs/potpied.log", "daemon.log"):
            log_file = self.home / name
            if log_file.exists():
                return log_file
        return None

    def logs(
        self,
        *,
        tail: int | None = DEFAULT_LOG_TAIL_LINES,
        since: datetime | None = None,
    ) -> list[str]:
        """The last ``tail`` lines of the daemon log, newest last.

        ``tail=None`` (or ``0``) reads the whole file — an explicit request,
        which is the only way the unbounded read this used to do by default
        should ever happen.
        """
        log_file = self.log_path()
        if log_file is None:
            return []
        lines = _read_tail_lines(log_file, tail)
        return _filter_since(lines, since) if since is not None else lines

    def follow_logs(
        self,
        *,
        tail: int | None = DEFAULT_LOG_TAIL_LINES,
        since: datetime | None = None,
        poll_interval: float = FOLLOW_POLL_SECONDS,
    ) -> Iterator[str]:
        """Yield the tail, then every line appended after it, forever.

        Polls rather than watches: the one consumer is a CLI command a human
        interrupts, and a filesystem-notification dependency for that is not a
        trade worth making. Handles a truncated or replaced file by restarting
        from its beginning, so the stream survives whatever rotation someone
        eventually adds.
        """
        log_file = self.log_path()
        position = 0
        if log_file is not None:
            # Backlog and resume offset come out of one open, and the offset is
            # fixed *before* the first yield. Taking it afterwards means the
            # consumer's own pause is a window in which appended lines land
            # past the backlog and behind the new offset — read by nobody.
            try:
                backlog, position = _tail_with_offset(log_file, tail)
            except OSError:  # pragma: no cover - vanished between the two calls
                log_file = None
                backlog = []
            for line in backlog if since is None else _filter_since(backlog, since):
                yield line
        pending = ""
        while True:
            if log_file is None:
                log_file = self.log_path()
                if log_file is None:
                    time.sleep(poll_interval)
                    continue
                position = 0
            try:
                size = log_file.stat().st_size
            except OSError:
                # Rediscover the file next pass. Sleeping first matters: a path
                # that exists but cannot be stat'd would otherwise spin.
                log_file = None
                time.sleep(poll_interval)
                continue
            if size < position:  # truncated or replaced under us
                position = 0
                pending = ""
            if size == position:
                time.sleep(poll_interval)
                continue
            with log_file.open("rb") as handle:
                handle.seek(position)
                chunk = handle.read()
                position = handle.tell()
            pending += chunk.decode("utf-8", errors="replace")
            while "\n" in pending:
                line, _, pending = pending.partition("\n")
                if since is None or _at_or_after(line, since):
                    yield line

    def ensure(self, plan: SetupPlan | None = None) -> StepResult:
        if self.in_process:
            return StepResult(
                "daemon.ensure",
                SKIPPED,
                "in-process host; no detached daemon to start",
                metadata={"mode": "in_process"},
            )
        status = self.status()
        if status["up"]:
            # ``up`` is the right thing to gate on — the pid file is owned, so a
            # second daemon for this home would be refused anyway — and the
            # wrong thing to *report*. A wedged daemon is up and serving
            # nothing, and a step that says only "already running" hands setup a
            # green line for a daemon that will not answer the very next
            # command the operator runs.
            serving = bool(status.get("serving"))
            detail = f"daemon already running (pid={status.get('pid')})"
            if not serving:
                detail += " but not answering; restart it with 'potpie daemon restart'"
            return StepResult(
                "daemon.ensure",
                SKIPPED,
                detail,
                metadata={
                    "mode": "detached",
                    "pid": status.get("pid"),
                    "serving": serving,
                    "socket": status.get("socket"),
                    "url": status.get("url"),
                },
            )
        info = self.start(backend=plan.backend if plan is not None else None)
        return StepResult(
            "daemon.ensure",
            DONE,
            f"daemon started (pid={info.get('pid')})",
            metadata={"mode": "detached", **info},
        )

    def install(self) -> dict[str, Any]:
        return {
            "installed": False,
            "detail": "no service unit for the local OSS daemon",
        }

    def start(self, *, backend: str | None = None) -> dict[str, Any]:
        from potpie.daemon.process.launcher import start_detached

        return start_detached(
            self.home,
            ready_timeout_s=self.startup_timeout_s,
            backend=backend,
        )

    def stop(self) -> dict[str, Any]:
        from potpie.daemon.process.launcher import stop_daemon

        return {"detail": stop_daemon(self.home)}

    def restart(self) -> dict[str, Any]:
        current_status = self.status()
        current_backend = current_status.get("backend")
        if current_status.get("up") and not isinstance(current_backend, str):
            raise RuntimeError(
                "cannot determine running daemon backend; "
                "refusing restart to avoid backend drift"
            )
        self.stop()
        info = self.start(
            backend=current_backend if isinstance(current_backend, str) else None
        )
        status = self.status()
        if "backend" in status:
            info = {**info, "backend": status["backend"]}
        return {**info, "started": info}


#: Prose for each :meth:`Daemon.status` state, keyed so the three readings
#: cannot drift apart in a chain of ternaries.
_STATE_DETAIL: Final[dict[str, str]] = {
    STATE_RUNNING: "detached daemon running",
    STATE_UNRESPONSIVE: (
        "detached daemon process {pid} exists but is not answering "
        "(stopped, wedged, or still starting)"
    ),
    STATE_NOT_RUNNING: "detached daemon not running",
}


def parse_since(value: str) -> datetime:
    """``--since`` as either an ISO-8601 instant or a relative age (``15m``).

    Raises :class:`ValueError` — the CLI's error boundary already renders that
    as a validation failure with the offending value in the message.
    """
    text = value.strip()
    relative = _RELATIVE_SINCE.match(text)
    if relative is not None:
        amount = int(relative.group(1))
        unit = _RELATIVE_SINCE_UNITS[relative.group(2).lower()]
        return datetime.now() - timedelta(**{unit: amount})
    try:
        return _as_naive_local(datetime.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"cannot read {value!r} as a time; use an ISO-8601 timestamp "
            "(2026-08-12T09:00:00) or a relative age (30s, 15m, 2h, 1d)"
        ) from None


def _read_tail_lines(path: Path, tail: int | None) -> list[str]:
    """The last ``tail`` lines of ``path``, reading only as far back as needed."""
    return _tail_with_offset(path, tail)[0]


def _tail_with_offset(path: Path, tail: int | None) -> tuple[list[str], int]:
    """The tail, plus the byte offset it ends at, from a single open.

    The offset is what ``--follow`` resumes from, and it has to describe the
    same instant as the lines above it — two separate opens would either lose
    or repeat whatever was appended between them.
    """
    with path.open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        if tail is None or tail <= 0:
            handle.seek(0)
            data = handle.read(end)
        else:
            position = end
            data = b""
            # ``<=`` not ``<``: the final line usually ends in a newline, so N
            # newlines can bound only N-1 whole lines.
            while position > 0 and data.count(b"\n") <= tail:
                step = min(_TAIL_CHUNK_BYTES, position)
                position -= step
                handle.seek(position)
                data = handle.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    return (lines if tail is None or tail <= 0 else lines[-tail:]), end


def _filter_since(lines: list[str], since: datetime) -> list[str]:
    """Drop lines older than ``since``, keeping their untimestamped tails.

    A traceback is one log event spread over a dozen lines and only the first
    carries a timestamp. Filtering line-by-line would keep the header and throw
    away the stack, so an undated line inherits the verdict of the last dated
    one above it.
    """
    kept: list[str] = []
    including = False
    for line in lines:
        stamp = _line_timestamp(line)
        if stamp is not None:
            including = stamp >= since
        if including:
            kept.append(line)
    return kept


def _at_or_after(line: str, since: datetime) -> bool:
    stamp = _line_timestamp(line)
    # An undated line in a live stream follows a line already admitted (the
    # tail filter above dropped everything older), so it belongs to the stream.
    return True if stamp is None else stamp >= since


def _line_timestamp(line: str) -> datetime | None:
    """The instant a log line carries, in either configured format."""
    text = line.lstrip()
    if text.startswith("{"):
        try:
            stamp = json.loads(text).get("ts")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(stamp, str):
            return None
        try:
            return _as_naive_local(datetime.fromisoformat(stamp))
        except ValueError:
            return None
    match = _PLAIN_TIMESTAMP.match(text)
    if match is None:
        return None
    fraction = (match.group(3) or "0").ljust(6, "0")[:6]
    try:
        return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}.{fraction}")
    except ValueError:  # pragma: no cover - the regex already constrains this
        return None


def _as_naive_local(value: datetime) -> datetime:
    """Compare timestamps in one frame: the machine's own wall clock.

    The plain formatter writes naive local time and the JSON one writes an
    offset, so a mixed log would otherwise raise on the first comparison.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone().replace(tzinfo=None)


__all__ = [
    "DEFAULT_LOG_TAIL_LINES",
    "STATE_IN_PROCESS",
    "STATE_NOT_RUNNING",
    "STATE_RUNNING",
    "STATE_UNRESPONSIVE",
    "Daemon",
    "parse_since",
]
