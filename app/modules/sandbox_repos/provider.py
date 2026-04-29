"""WorkspaceProvider implementation backed by RepoManager.

Implements ``sandbox.domain.ports.workspaces.WorkspaceProvider`` (a Protocol)
without importing it — duck-typed registration via ``build_sandbox_container``
is enough. We delegate clone + worktree operations to RepoManager so:

* parsing and agent tooling share the same on-disk bare repo
* worktrees are tracked by RepoManager's metadata (visible to its eviction
  policy — agent worktrees age out via the same 30-day stale sweep)
* cleanup happens through RepoManager's ``cleanup_unique_worktree`` so
  metadata stays consistent
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from sandbox.adapters.outbound.local.auth import resolve_token
from sandbox.domain.errors import (
    InvalidWorkspacePath,
    RepoAuthFailed,
    RepoCacheUnavailable,
)
from sandbox.domain.models import (
    Mount,
    Workspace,
    WorkspaceLocation,
    WorkspaceMode,
    WorkspaceRequest,
    WorkspaceState,
    WorkspaceStorageKind,
    new_id,
    utc_now,
)

if TYPE_CHECKING:
    from app.modules.repo_manager.repo_manager import RepoManager


class RepoManagerWorkspaceProvider:
    """Sandbox workspace provider that delegates to RepoManager."""

    kind = "local_repo_manager"

    def __init__(self, repo_manager: "RepoManager") -> None:
        self._rm = repo_manager
        self._by_id: dict[str, Workspace] = {}
        self._by_key: dict[str, str] = {}

    async def get_or_create_workspace(self, request: WorkspaceRequest) -> Workspace:
        key = request.key()
        existing_id = self._by_key.get(key)
        if existing_id:
            existing = self._by_id.get(existing_id)
            if (
                existing
                and existing.location.local_path
                and Path(existing.location.local_path).exists()
            ):
                existing.last_used_at = utc_now()
                existing.updated_at = utc_now()
                return existing

        workspace = await asyncio.to_thread(self._create_sync, request)
        self._by_id[workspace.id] = workspace
        self._by_key[workspace.key] = workspace.id
        return workspace

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        return self._by_id.get(workspace_id)

    async def delete_workspace(self, workspace: Workspace) -> None:
        self._by_id.pop(workspace.id, None)
        self._by_key.pop(workspace.key, None)
        request = workspace.request
        # Analysis worktrees aren't conversation-scoped — they're shared across
        # callers and live until RepoManager's volume eviction trims them.
        if request.mode is WorkspaceMode.ANALYSIS:
            return
        scope = request.conversation_id or request.task_id
        if not scope:
            return
        await asyncio.to_thread(
            self._rm.cleanup_unique_worktree,
            request.repo.repo_name,
            request.user_id,
            scope,
        )

    async def mount_for_runtime(
        self, workspace: Workspace, *, writable: bool
    ) -> Mount:
        if workspace.location.local_path is None:
            raise InvalidWorkspacePath("Local workspace has no local_path")
        return Mount(
            source=str(Path(workspace.location.local_path).resolve()),
            target="/work",
            writable=writable,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _create_sync(self, request: WorkspaceRequest) -> Workspace:
        repo_name = request.repo.repo_name
        repo_url = request.repo.repo_url
        base_ref = request.base_ref
        token = request.auth_token or resolve_token(
            repo_name=repo_name, user_id=request.user_id
        )

        try:
            if request.mode is WorkspaceMode.ANALYSIS and not request.create_branch:
                worktree_path, branch = self._analysis_worktree(
                    request, repo_name, repo_url, base_ref, token
                )
            else:
                worktree_path, branch = self._edit_worktree(
                    request, repo_name, repo_url, base_ref, token
                )
        except (RepoCacheUnavailable, RepoAuthFailed):
            raise
        except RuntimeError as exc:
            # RepoManager raises bare RuntimeErrors for clone/worktree failures.
            message = str(exc)
            lower = message.lower()
            if "authentication" in lower or "permission denied" in lower:
                raise RepoAuthFailed(message) from exc
            raise RepoCacheUnavailable(message) from exc

        return Workspace(
            id=new_id("ws"),
            key=request.key(),
            repo_cache_id=None,
            request=request,
            location=WorkspaceLocation(
                kind=WorkspaceStorageKind.LOCAL_PATH,
                local_path=str(Path(worktree_path).resolve()),
            ),
            backend_kind=self.kind,
            state=WorkspaceState.READY,
            metadata={"branch": branch},
        )

    def _analysis_worktree(
        self,
        request: WorkspaceRequest,
        repo_name: str,
        repo_url: str | None,
        base_ref: str,
        token: str | None,
    ) -> tuple[Path, str]:
        """Read-only worktree at base_ref. Shared across callers."""
        self._rm.ensure_bare_repo(
            repo_name=repo_name,
            repo_url=repo_url,
            auth_token=token,
            ref=base_ref,
            user_id=request.user_id,
        )
        worktree_path = self._rm.create_worktree(
            repo_name=repo_name,
            ref=base_ref,
            auth_token=token,
            user_id=request.user_id,
            exists_ok=True,
        )
        return worktree_path, base_ref

    def _edit_worktree(
        self,
        request: WorkspaceRequest,
        repo_name: str,
        repo_url: str | None,
        base_ref: str,
        token: str | None,
    ) -> tuple[Path, str]:
        """Per-conversation/task worktree on a new branch off base_ref."""
        scope = request.conversation_id or request.task_id
        if not scope:
            raise ValueError(
                "EDIT/TASK workspace requires conversation_id or task_id"
            )
        new_branch_name = request.branch_name or _default_branch_name(request, scope)
        worktree_path = self._rm.create_worktree_with_new_branch(
            repo_name=repo_name,
            base_ref=base_ref,
            new_branch_name=new_branch_name,
            auth_token=token,
            user_id=request.user_id,
            unique_id=scope,
            exists_ok=True,
            repo_url=repo_url,
        )
        return worktree_path, new_branch_name


def _default_branch_name(request: WorkspaceRequest, scope: str) -> str:
    safe_scope = scope.replace("/", "-").replace("\\", "-").replace(" ", "-")
    if request.mode is WorkspaceMode.TASK:
        return f"agent/task-{safe_scope}"
    return f"agent/edits-{safe_scope}"
