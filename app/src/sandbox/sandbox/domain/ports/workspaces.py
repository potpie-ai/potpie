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

    async def is_alive(self, workspace: Workspace) -> bool:
        """Cheap liveness probe.

        ``True`` ⇒ the workspace's backing storage still exists and the
        provider can serve operations against it. ``False`` ⇒ the
        workspace is gone (filesystem scrubbed, Daytona sandbox
        archived/deleted, etc.) and the caller should re-create.

        Adapters are expected to make this **cheap** — local checks the
        worktree path, Daytona probes the SDK by id without touching the
        runtime. Anything heavier than that defeats the purpose; the
        intended caller (``ProjectSandbox.health_check``) runs this on
        every conversation message.
        """
        ...
