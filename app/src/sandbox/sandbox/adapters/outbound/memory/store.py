"""In-memory sandbox metadata store."""

from __future__ import annotations

import asyncio

from sandbox.domain.models import Runtime, Workspace


class InMemorySandboxStore:
    def __init__(self) -> None:
        self._workspaces: dict[str, Workspace] = {}
        self._workspace_keys: dict[str, str] = {}
        self._runtimes: dict[str, Runtime] = {}
        self._lock = asyncio.Lock()

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        async with self._lock:
            return self._workspaces.get(workspace_id)

    async def find_workspace_by_key(self, key: str) -> Workspace | None:
        async with self._lock:
            workspace_id = self._workspace_keys.get(key)
            if workspace_id is None:
                return None
            return self._workspaces.get(workspace_id)

    async def save_workspace(self, workspace: Workspace) -> None:
        async with self._lock:
            self._workspaces[workspace.id] = workspace
            self._workspace_keys[workspace.key] = workspace.id

    async def delete_workspace(self, workspace_id: str) -> None:
        async with self._lock:
            workspace = self._workspaces.pop(workspace_id, None)
            if workspace is not None:
                self._workspace_keys.pop(workspace.key, None)

    async def list_workspaces(self) -> list[Workspace]:
        async with self._lock:
            return list(self._workspaces.values())

    async def get_runtime(self, runtime_id: str) -> Runtime | None:
        async with self._lock:
            return self._runtimes.get(runtime_id)

    async def find_runtime_by_workspace(
        self, workspace_id: str, backend_kind: str | None = None
    ) -> Runtime | None:
        async with self._lock:
            for runtime in self._runtimes.values():
                if runtime.workspace_id != workspace_id:
                    continue
                if backend_kind is not None and runtime.backend_kind != backend_kind:
                    continue
                return runtime
            return None

    async def save_runtime(self, runtime: Runtime) -> None:
        async with self._lock:
            self._runtimes[runtime.id] = runtime

    async def delete_runtime(self, runtime_id: str) -> None:
        async with self._lock:
            self._runtimes.pop(runtime_id, None)

    async def list_runtimes(self) -> list[Runtime]:
        async with self._lock:
            return list(self._runtimes.values())

