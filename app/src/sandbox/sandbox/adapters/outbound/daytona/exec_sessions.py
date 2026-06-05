"""Daytona unified-exec session manager (pipe + PTY).

Two backends behind one contract (mirrors the local manager and Codex):

* **pipe** — a Daytona *session command* run async
  (``process.create_session`` → ``execute_session_command(run_async=True)``);
  progress is read by polling ``get_session_command_logs().output`` with a
  cursor, completion via ``get_session_command().exit_code``, and stdin via
  ``send_session_command_input``.
* **pty** — a real terminal (``process.create_pty_session`` → ``PtyHandle``);
  a background thread drains ``PtyHandle.wait(on_data=…)`` into a buffer and
  ``send_input`` feeds stdin. Use for REPLs / interactive programs.

All SDK calls are synchronous, so they run on worker threads. The ``sandbox``
handle is passed in per call (the provider resolves it, with auto-start /
recovery, before delegating here).
"""

from __future__ import annotations

import asyncio
import shlex
import threading
import time
from typing import Any

from sandbox.domain.errors import ExecSessionNotFound
from sandbox.domain.models import (
    SESSION_OUTPUT_MAX_BYTES,
    ExecSessionResult,
    new_id,
)

_PIPE_POLL_INTERVAL_S = 0.25  # remote API — don't hammer it like the local fd
_PTY_POLL_INTERVAL_S = 0.05


class _DaytonaSession:
    def __init__(self, kind: str, *, daytona_session_id: str) -> None:
        self.kind = kind
        self.daytona_session_id = daytona_session_id
        # pipe
        self.cmd_id: str | None = None
        self.cursor = 0
        # pty
        self.pty_handle: Any = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._pty_cursor = 0
        self.exit_code: int | None = None
        self._done = threading.Event()

    # -- pty buffer plumbing -------------------------------------------
    def append(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)

    def read_pty_delta(self) -> bytes:
        with self._lock:
            out = bytes(self._buf[self._pty_cursor :])
            self._pty_cursor = len(self._buf)
        return out

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    def mark_done(self, exit_code: int | None) -> None:
        self.exit_code = exit_code
        self._done.set()


class DaytonaExecSessions:
    def __init__(self, *, max_sessions: int = 64) -> None:
        self._sessions: dict[str, _DaytonaSession] = {}
        self._max = max_sessions

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------
    async def start(
        self,
        sandbox: Any,
        *,
        cwd: str | None,
        env: dict[str, str],
        cmd: tuple[str, ...],
        shell: bool,
        tty: bool,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        self._reap()
        our_id = new_id("es")
        if tty:
            session = await asyncio.to_thread(
                self._start_pty, sandbox, our_id, cwd, env, cmd, shell
            )
        else:
            session = await asyncio.to_thread(
                self._start_pipe, sandbox, our_id, cwd, env, cmd, shell
            )
        self._sessions[our_id] = session
        return await self._collect(
            sandbox, our_id, session, yield_time_ms, max_output_bytes
        )

    def _start_pipe(
        self,
        sandbox: Any,
        our_id: str,
        cwd: str | None,
        env: dict[str, str],
        cmd: tuple[str, ...],
        shell: bool,
    ) -> _DaytonaSession:
        from daytona import SessionExecuteRequest

        daytona_session_id = our_id
        sandbox.process.create_session(daytona_session_id)
        full = _compose_command(cwd, env, cmd, shell)
        resp = sandbox.process.execute_session_command(
            daytona_session_id,
            SessionExecuteRequest(command=full, run_async=True),
        )
        session = _DaytonaSession("pipe", daytona_session_id=daytona_session_id)
        session.cmd_id = str(getattr(resp, "cmd_id", "") or "")
        return session

    def _start_pty(
        self,
        sandbox: Any,
        our_id: str,
        cwd: str | None,
        env: dict[str, str],
        cmd: tuple[str, ...],
        shell: bool,
    ) -> _DaytonaSession:
        handle = sandbox.process.create_pty_session(
            id=our_id, cwd=cwd or None, envs=env or None
        )
        handle.wait_for_connection(timeout=10.0)
        session = _DaytonaSession("pty", daytona_session_id=our_id)
        session.pty_handle = handle

        def _runner() -> None:
            try:
                result = handle.wait(on_data=session.append)
                session.mark_done(getattr(result, "exit_code", None))
            except Exception:  # noqa: BLE001
                session.mark_done(-1)

        threading.Thread(target=_runner, daemon=True).start()
        # The PTY starts a shell; send the requested command as the first line
        # so it behaves like `exec_command` (further input via write_stdin).
        inner = (
            cmd[0]
            if (shell and len(cmd) == 1)
            else " ".join(shlex.quote(c) for c in cmd)
        )
        if inner.strip():
            handle.send_input(inner + "\n")
        return session

    # ------------------------------------------------------------------
    # write / poll / kill
    # ------------------------------------------------------------------
    async def write(
        self,
        sandbox: Any,
        session_id: str,
        data: str,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        session = self._require(session_id)
        if data:
            await asyncio.to_thread(self._send_input, sandbox, session, data)
        return await self._collect(
            sandbox, session_id, session, yield_time_ms, max_output_bytes
        )

    async def poll(
        self,
        sandbox: Any,
        session_id: str,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        session = self._require(session_id)
        return await self._collect(
            sandbox, session_id, session, yield_time_ms, max_output_bytes
        )

    async def kill(self, sandbox: Any, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        await asyncio.to_thread(self._kill_sync, sandbox, session)

    def _send_input(self, sandbox: Any, session: _DaytonaSession, data: str) -> None:
        if session.kind == "pty" and session.pty_handle is not None:
            session.pty_handle.send_input(data)
            return
        if session.cmd_id:
            sandbox.process.send_session_command_input(
                session.daytona_session_id, session.cmd_id, data
            )

    def _kill_sync(self, sandbox: Any, session: _DaytonaSession) -> None:
        try:
            if session.kind == "pty" and session.pty_handle is not None:
                session.pty_handle.kill()
            else:
                # Best-effort interrupt, then drop the whole session.
                if session.cmd_id:
                    try:
                        sandbox.process.send_session_command_input(
                            session.daytona_session_id, session.cmd_id, "\x03"
                        )
                    except Exception:  # noqa: BLE001
                        pass
                sandbox.process.delete_session(session.daytona_session_id)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # collect
    # ------------------------------------------------------------------
    async def _collect(
        self,
        sandbox: Any,
        session_id: str,
        session: _DaytonaSession,
        yield_time_ms: int,
        max_output_bytes: int | None,
    ) -> ExecSessionResult:
        deadline = time.monotonic() + yield_time_ms / 1000.0
        out = bytearray()
        interval = (
            _PTY_POLL_INTERVAL_S if session.kind == "pty" else _PIPE_POLL_INTERVAL_S
        )
        while True:
            delta, finished, exit_code = await asyncio.to_thread(
                self._read_once, sandbox, session
            )
            if delta:
                out.extend(delta)
            if finished:
                drained, _, _ = await asyncio.to_thread(
                    self._read_once, sandbox, session
                )
                if drained:
                    out.extend(drained)
                session.exit_code = (
                    exit_code if exit_code is not None else session.exit_code
                )
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(interval)

        running = not self._is_finished(session)
        if not running:
            self._sessions.pop(session_id, None)
            await asyncio.to_thread(self._cleanup, sandbox, session)

        cap = max_output_bytes or SESSION_OUTPUT_MAX_BYTES
        data = bytes(out)
        truncated = False
        if cap > 0 and len(data) > cap:
            data = data[-cap:]
            truncated = True
        return ExecSessionResult(
            session_id=session_id,
            output=data.decode("utf-8", errors="replace"),
            running=running,
            exit_code=None if running else session.exit_code,
            truncated=truncated,
        )

    def _read_once(
        self, sandbox: Any, session: _DaytonaSession
    ) -> tuple[bytes, bool, int | None]:
        """One synchronous read: (delta_bytes, finished, exit_code)."""
        if session.kind == "pty":
            return session.read_pty_delta(), session.finished, session.exit_code
        # pipe: full combined log, sliced by cursor.
        logs = sandbox.process.get_session_command_logs(
            session.daytona_session_id, session.cmd_id
        )
        full = getattr(logs, "output", None) or ""
        delta = full[session.cursor :]
        session.cursor = len(full)
        cmd = sandbox.process.get_session_command(
            session.daytona_session_id, session.cmd_id
        )
        exit_code = getattr(cmd, "exit_code", None)
        return delta.encode("utf-8"), exit_code is not None, exit_code

    @staticmethod
    def _is_finished(session: _DaytonaSession) -> bool:
        if session.kind == "pty":
            return session.finished
        return session.exit_code is not None

    def _cleanup(self, sandbox: Any, session: _DaytonaSession) -> None:
        try:
            if session.kind == "pipe":
                sandbox.process.delete_session(session.daytona_session_id)
        except Exception:  # noqa: BLE001
            pass

    def _require(self, session_id: str) -> _DaytonaSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ExecSessionNotFound(session_id)
        return session

    def _reap(self) -> None:
        for sid in [s for s, sess in self._sessions.items() if self._is_finished(sess)]:
            self._sessions.pop(sid, None)
        overflow = len(self._sessions) - self._max
        for sid in list(self._sessions.keys())[: max(0, overflow)]:
            self._sessions.pop(sid, None)


def _compose_command(
    cwd: str | None, env: dict[str, str], cmd: tuple[str, ...], shell: bool
) -> str:
    """Build one shell line: cd into cwd, export env, then run the command."""
    inner = (
        cmd[0] if (shell and len(cmd) == 1) else " ".join(shlex.quote(c) for c in cmd)
    )
    exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in env.items())
    if cwd:
        return f"cd {shlex.quote(cwd)} && {{ {exports}{inner}; }}"
    return f"{exports}{inner}"
