"""Runtime provider port."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from sandbox.domain.models import (
    ExecChunk,
    ExecRequest,
    ExecResult,
    ExecSessionResult,
    Runtime,
    RuntimeCapabilities,
    RuntimeSpec,
    SessionExecRequest,
    SessionInputRequest,
)


class RuntimeProvider(Protocol):
    kind: str
    capabilities: RuntimeCapabilities

    async def create(self, workspace_id: str, spec: RuntimeSpec) -> Runtime: ...

    async def get(self, runtime_id: str) -> Runtime | None: ...

    async def start(self, runtime: Runtime) -> Runtime: ...

    async def stop(self, runtime: Runtime) -> Runtime: ...

    async def destroy(self, runtime: Runtime) -> None: ...

    async def exec(self, runtime: Runtime, request: ExecRequest) -> ExecResult: ...

    async def exec_stream(
        self, runtime: Runtime, request: ExecRequest
    ) -> AsyncIterator[ExecChunk]: ...

    # --- Unified exec (Codex-style streamable sessions) -----------------
    # Backends that can't host long-lived sessions raise
    # ``SessionsUnsupported``; ``capabilities.interactive_session`` advertises
    # support so callers can branch before invoking.

    async def exec_session_start(
        self, runtime: Runtime, request: SessionExecRequest
    ) -> ExecSessionResult:
        """Start a command in a new session; collect output for the yield window."""
        ...

    async def exec_session_write(
        self, runtime: Runtime, request: SessionInputRequest
    ) -> ExecSessionResult:
        """Write stdin to a running session, then collect new output."""
        ...

    async def exec_session_poll(
        self,
        runtime: Runtime,
        session_id: str,
        *,
        yield_time_ms: int,
        max_output_bytes: int | None = None,
    ) -> ExecSessionResult:
        """Collect new output from a session without writing (progress read)."""
        ...

    async def exec_session_kill(self, runtime: Runtime, session_id: str) -> None:
        """Terminate a session and release its resources."""
        ...
