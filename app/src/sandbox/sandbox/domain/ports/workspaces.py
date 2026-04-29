"""Workspace provider port."""

from __future__ import annotations

from typing import Protocol

from sandbox.domain.models import Mount, Workspace, WorkspaceRequest


class WorkspaceProvider(Protocol):
    kind: str

    async def get_or_create_workspace(self, request: WorkspaceRequest) -> Workspace:
        ...

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        ...

    async def delete_workspace(self, workspace: Workspace) -> None:
        ...

    async def mount_for_runtime(self, workspace: Workspace, *, writable: bool) -> Mount:
        ...
