"""Local unified-exec session manager (pipe + PTY).

Backs :class:`LocalSubprocessRuntimeProvider`'s Codex-style session surface:
a command is spawned detached, a reader thread drains its output into a
buffer, and each ``start`` / ``write`` / ``poll`` call collects the newly
produced output for a bounded *yield window* and reports whether the command
is still running. Mirrors OpenAI Codex's unified-exec contract.

This is local-dev / analysis only and not a security boundary (same posture
as the provider). ``pty`` is Unix-only; the manager is imported lazily so the
package still loads on platforms without it.
"""

from __future__ import annotations

import asyncio
import os
import pty
import shlex
import subprocess
import threading
import time

from sandbox.domain.errors import ExecSessionNotFound
from sandbox.domain.models import (
    SESSION_OUTPUT_MAX_BYTES,
    ExecSessionResult,
    new_id,
)

_READ_CHUNK = 65_536
_POLL_INTERVAL_S = 0.05


class _LocalSession:
    """A single spawned command: a process plus a background output reader.

    Output (stdout+stderr interleaved, like a terminal) accumulates in
    ``_buf``; ``read_delta`` returns the slice since the last read so each
    collect call only sees new bytes.
    """

    def __init__(
        self, session_id: str, proc: subprocess.Popen[bytes], *, master_fd: int | None
    ) -> None:
        self.id = session_id
        self._proc = proc
        self._master_fd = master_fd
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._cursor = 0
        self._exit_code: int | None = None
        self._done = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_fd(self) -> int:
        if self._master_fd is not None:
            return self._master_fd
        assert self._proc.stdout is not None
        return self._proc.stdout.fileno()

    def _read_loop(self) -> None:
        fd = self._read_fd()
        try:
            while True:
                try:
                    # os.read returns as soon as >=1 byte is available (true
                    # streaming) — file objects' .read(n) would block until n.
                    chunk = os.read(fd, _READ_CHUNK)
                except OSError:
                    break  # PTY master raises EIO when the slave closes (exit)
                if not chunk:
                    break
                with self._lock:
                    self._buf.extend(chunk)
        finally:
            try:
                self._exit_code = self._proc.wait()
            except Exception:  # noqa: BLE001
                self._exit_code = -1
            self._done.set()

    def read_delta(self) -> bytes:
        with self._lock:
            data = bytes(self._buf[self._cursor :])
            self._cursor = len(self._buf)
        return data

    def write(self, data: str) -> None:
        raw = data.encode("utf-8")
        if self._master_fd is not None:
            os.write(self._master_fd, raw)
        elif self._proc.stdin is not None:
            self._proc.stdin.write(raw)
            self._proc.stdin.flush()

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def exit_code(self) -> int | None:
        return self._exit_code

    def kill(self) -> None:
        try:
            self._proc.kill()
        except Exception:  # noqa: BLE001
            pass
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except Exception:  # noqa: BLE001
                pass


class LocalExecSessions:
    """Registry + collect loop for local unified-exec sessions."""

    def __init__(self, *, max_sessions: int = 64) -> None:
        self._sessions: dict[str, _LocalSession] = {}
        self._max = max_sessions

    async def start(
        self,
        *,
        cwd: str,
        env: dict[str, str],
        cmd: tuple[str, ...],
        shell: bool,
        tty: bool,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        self._reap()
        session_id = new_id("es")
        proc, master = await asyncio.to_thread(self._spawn, cwd, env, cmd, shell, tty)
        session = _LocalSession(session_id, proc, master_fd=master)
        self._sessions[session_id] = session
        return await self._collect(session, yield_time_ms, max_output_bytes)

    async def write(
        self,
        session_id: str,
        data: str,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        session = self._require(session_id)
        if data:
            await asyncio.to_thread(session.write, data)
        return await self._collect(session, yield_time_ms, max_output_bytes)

    async def poll(
        self, session_id: str, yield_time_ms: int, max_output_bytes: int | None
    ) -> ExecSessionResult:
        return await self._collect(
            self._require(session_id), yield_time_ms, max_output_bytes
        )

    async def kill(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await asyncio.to_thread(session.kill)

    # -- internals ------------------------------------------------------
    def _require(self, session_id: str) -> _LocalSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ExecSessionNotFound(session_id)
        return session

    @staticmethod
    def _spawn(
        cwd: str, env: dict[str, str], cmd: tuple[str, ...], shell: bool, tty: bool
    ) -> tuple[subprocess.Popen[bytes], int | None]:
        if shell:
            inner = cmd[0] if len(cmd) == 1 else " ".join(shlex.quote(c) for c in cmd)
            argv = ["/bin/sh", "-lc", inner]
        else:
            argv = list(cmd)
        if tty:
            master, slave = pty.openpty()
            proc = subprocess.Popen(  # noqa: S603
                argv,
                cwd=cwd,
                env=env,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                close_fds=True,
            )
            os.close(slave)
            return proc, master
        proc = subprocess.Popen(  # noqa: S603
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc, None

    async def _collect(
        self,
        session: _LocalSession,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        deadline = time.monotonic() + yield_time_ms / 1000.0
        out = bytearray()
        while True:
            out.extend(session.read_delta())
            if session.finished:
                out.extend(session.read_delta())  # final drain
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(_POLL_INTERVAL_S)

        running = not session.finished
        if not running:
            self._sessions.pop(session.id, None)

        cap = max_output_bytes or SESSION_OUTPUT_MAX_BYTES
        data = bytes(out)
        truncated = False
        if cap > 0 and len(data) > cap:
            data = data[-cap:]  # tail-truncate (keep the most recent output)
            truncated = True
        return ExecSessionResult(
            session_id=session.id,
            output=data.decode("utf-8", errors="replace"),
            running=running,
            exit_code=None if running else session.exit_code,
            truncated=truncated,
        )

    def _reap(self) -> None:
        for sid in [s for s, sess in self._sessions.items() if sess.finished]:
            self._sessions.pop(sid, None)
        # LRU-ish: oldest insertion order first, kill down to the cap.
        overflow = len(self._sessions) - self._max
        for sid in list(self._sessions.keys())[: max(0, overflow)]:
            sess = self._sessions.pop(sid, None)
            if sess is not None:
                sess.kill()
