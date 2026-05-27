"""Persistence ports for sandbox metadata."""

from __future__ import annotations

from typing import Protocol

from sandbox.domain.models import RepoCache, Runtime, Workspace


class WorkspaceStore(Protocol):
    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        ...

    async def find_workspace_by_key(self, key: str) -> Workspace | None:
        ...

    async def save_workspace(self, workspace: Workspace) -> None:
        ...

    async def delete_workspace(self, workspace_id: str) -> None:
        ...

    async def list_workspaces(self) -> list[Workspace]:
        ...


class RuntimeStore(Protocol):
    async def get_runtime(self, runtime_id: str) -> Runtime | None:
        ...

    async def find_runtime_by_workspace(
        self, workspace_id: str, backend_kind: str | None = None
    ) -> Runtime | None:
        ...

    async def save_runtime(self, runtime: Runtime) -> None:
        ...

    async def delete_runtime(self, runtime_id: str) -> None:
        ...

    async def list_runtimes(self) -> list[Runtime]:
        ...


class RepoCacheStore(Protocol):
    async def get_repo_cache(self, cache_id: str) -> RepoCache | None:
        ...

    async def find_repo_cache_by_key(self, key: str) -> RepoCache | None:
        ...

    async def save_repo_cache(self, cache: RepoCache) -> None:
        ...

    async def delete_repo_cache(self, cache_id: str) -> None:
        ...

    async def list_repo_caches(self) -> list[RepoCache]:
        ...


class SandboxStore(WorkspaceStore, RuntimeStore, RepoCacheStore, Protocol):
    """Combined store used by SandboxService."""

