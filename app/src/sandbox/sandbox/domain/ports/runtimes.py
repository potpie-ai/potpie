"""Runtime provider port."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from sandbox.domain.models import (
    ExecChunk,
    ExecRequest,
    ExecResult,
    Runtime,
    RuntimeCapabilities,
    RuntimeSpec,
)


class RuntimeProvider(Protocol):
    kind: str
    capabilities: RuntimeCapabilities

    async def create(self, workspace_id: str, spec: RuntimeSpec) -> Runtime:
        ...

    async def get(self, runtime_id: str) -> Runtime | None:
        ...

    async def start(self, runtime: Runtime) -> Runtime:
        ...

    async def stop(self, runtime: Runtime) -> Runtime:
        ...

    async def destroy(self, runtime: Runtime) -> None:
        ...

    async def exec(self, runtime: Runtime, request: ExecRequest) -> ExecResult:
        ...

    async def exec_stream(
        self, runtime: Runtime, request: ExecRequest
    ) -> AsyncIterator[ExecChunk]:
        ...
