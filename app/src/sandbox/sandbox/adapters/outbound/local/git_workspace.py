"""Local `.repos` workspace provider backed by git worktrees.

The bare-repo concern lives in `LocalRepoCacheProvider`. This adapter
forks worktrees off the cache and tracks them in memory; persistence is
the application service's job (via `SandboxStore`).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from sandbox.adapters.outbound.local._git_ops import (
    run,
    validate_ref,
    validate_repo_name,
)
from sandbox.adapters.outbound.local.repo_cache import LocalRepoCacheProvider
from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy
from sandbox.domain.errors import (
    InvalidWorkspacePath,
    RepoCacheUnavailable,
)
from sandbox.domain.models import (
    Capabilities,
    Mount,
    RepoCache,
    RepoCacheRequest,
    Workspace,
    WorkspaceLocation,
    WorkspaceMode,
    WorkspaceRequest,
    WorkspaceState,
    WorkspaceStorageKind,
    new_id,
    utc_now,
)
from sandbox.domain.ports.eviction import EvictionPolicy

if TYPE_CHECKING:
    from sandbox.domain.ports.repos import RepoCacheProvider


class LocalGitWorkspaceProvider:
    """Create durable local worktrees off a `LocalRepoCacheProvider`."""

    kind = "local"

    def __init__(
        self,
        repos_base_path: str | Path | None = None,
        *,
        eviction: EvictionPolicy | None = None,
        repo_cache_provider: "RepoCacheProvider | None" = None,
    ) -> None:
        raw_base = repos_base_path or os.getenv("SANDBOX_REPOS_BASE_PATH") or ".repos"
        self.repos_base_path = Path(raw_base).expanduser().resolve()
        self.repos_base_path.mkdir(parents=True, exist_ok=True)
        self._by_id: dict[str, Workspace] = {}
        self._by_key: dict[str, str] = {}
        self._eviction: EvictionPolicy = eviction or NoOpEvictionPolicy()
        # Auto-construct a sibling cache provider rooted at the same
        # `.repos` path. Tests and the bootstrap can override.
        self._repo_cache_provider: "RepoCacheProvider" = (
            repo_cache_provider or LocalRepoCacheProvider(self.repos_base_path)
        )

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

        # Cache miss: give the policy a chance to free space before we
        # clone or check out a worktree. NoOp by default; production
        # wires in a volume- or age-based policy.
        await self._eviction.evict_if_needed(user_id=request.user_id)

        # Ensure the bare repo exists; the workspace is forked off this
        # cache. The cache provider is idempotent — repeated calls just
        # re-fetch the requested ref.
        validate_repo_name(request.repo.repo_name)
        cache = await self._repo_cache_provider.ensure_cache(
            RepoCacheRequest(
                repo=request.repo,
                base_ref=request.base_ref,
                user_id=request.user_id,
                auth_token=request.auth_token,
            )
        )

        workspace = await asyncio.to_thread(
            self._create_worktree_sync, request, cache
        )
        self._by_id[workspace.id] = workspace
        self._by_key[workspace.key] = workspace.id
        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        return self._by_id.get(workspace_id)

    async def delete_workspace(self, workspace: Workspace) -> None:
        self._by_id.pop(workspace.id, None)
        self._by_key.pop(workspace.key, None)
        path = workspace.location.local_path
        if not path:
            return
        await asyncio.to_thread(self._remove_worktree_sync, Path(path))

    async def mount_for_runtime(self, workspace: Workspace, *, writable: bool) -> Mount:
        if workspace.location.local_path is None:
            raise InvalidWorkspacePath("Local workspace has no local_path")
        return Mount(
            source=str(Path(workspace.location.local_path).resolve()),
            target="/work",
            writable=writable,
        )

    async def is_alive(self, workspace: Workspace) -> bool:
        """For the local adapter, alive ⇔ the worktree still exists on disk.

        Cheap: a single ``Path.exists()``. The in-memory tracking
        dicts (``_by_id`` / ``_by_key``) are not authoritative here —
        a checkbox-only check would return True for a worktree the
        operator removed by hand, which is the exact case
        ``ProjectSandbox.health_check`` is meant to catch.
        """
        path = workspace.location.local_path
        if not path:
            return False
        return Path(path).exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _create_worktree_sync(
        self, request: WorkspaceRequest, cache: RepoCache
    ) -> Workspace:
        validate_ref(request.base_ref)
        if request.branch_name:
            validate_ref(request.branch_name)

        if cache.location.local_path is None:
            raise RepoCacheUnavailable("RepoCache has no local bare-repo path")
        bare_path = Path(cache.location.local_path)

        branch = request.branch_name or self._default_branch_name(request)
        worktree_path = self._worktree_path(request, branch)
        if worktree_path.exists():
            return self._workspace_from_path(request, worktree_path, branch, cache)

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        if request.create_branch or request.mode is not WorkspaceMode.ANALYSIS:
            result = run(
                [
                    "git",
                    "-C",
                    str(bare_path),
                    "worktree",
                    "add",
                    "-b",
                    branch,
                    "--",
                    str(worktree_path),
                    request.base_ref,
                ],
                timeout=120,
            )
            if result.returncode != 0 and "already exists" in result.stderr:
                # Branch already exists in the bare repo (e.g. a prior
                # conversation's edit branch); re-attach the worktree
                # without re-creating the branch.
                result = run(
                    [
                        "git",
                        "-C",
                        str(bare_path),
                        "worktree",
                        "add",
                        "--",
                        str(worktree_path),
                        branch,
                    ],
                    timeout=120,
                )
            self._raise_if_failed(result, "git worktree add")
        else:
            result = run(
                [
                    "git",
                    "-C",
                    str(bare_path),
                    "worktree",
                    "add",
                    "--detach",
                    "--",
                    str(worktree_path),
                    request.base_ref,
                ],
                timeout=120,
            )
            self._raise_if_failed(result, "git worktree add")
        return self._workspace_from_path(request, worktree_path, branch, cache)

    def _workspace_from_path(
        self,
        request: WorkspaceRequest,
        worktree_path: Path,
        branch: str,
        cache: RepoCache,
    ) -> Workspace:
        return Workspace(
            id=new_id("ws"),
            key=request.key(),
            repo_cache_id=cache.id,
            request=request,
            location=WorkspaceLocation(
                kind=WorkspaceStorageKind.LOCAL_PATH,
                local_path=str(worktree_path.resolve()),
            ),
            backend_kind=self.kind,
            state=WorkspaceState.READY,
            metadata={"branch": branch},
            capabilities=Capabilities.from_mode(request.mode),
        )

    @staticmethod
    def _remove_worktree_sync(path: Path) -> None:
        if not path.exists():
            return
        try:
            result = run(
                [
                    "git",
                    "-C",
                    str(path),
                    "worktree",
                    "remove",
                    "--force",
                    str(path),
                ],
                timeout=60,
            )
            if result.returncode == 0:
                return
        except Exception:
            pass
        shutil.rmtree(path, ignore_errors=True)

    def _worktrees_dir(self, repo_name: str) -> Path:
        return self.repos_base_path / repo_name / "worktrees"

    def _worktree_path(self, request: WorkspaceRequest, branch: str) -> Path:
        safe_user = self._safe_segment(request.user_id)
        scope = request.conversation_id or request.task_id or request.base_ref
        safe_scope = self._safe_segment(scope)
        safe_branch = self._safe_segment(branch)
        return (
            self._worktrees_dir(request.repo.repo_name)
            / f"{safe_user}_{safe_scope}_{safe_branch}"
        )

    def _default_branch_name(self, request: WorkspaceRequest) -> str:
        if request.mode is WorkspaceMode.ANALYSIS:
            return request.base_ref
        if request.mode is WorkspaceMode.TASK and request.task_id:
            return f"agent/task-{self._safe_branch_component(request.task_id)}"
        if request.conversation_id:
            return f"agent/edits-{self._safe_branch_component(request.conversation_id)}"
        return f"agent/workspace-{new_id('branch')}"

    @staticmethod
    def _raise_if_failed(
        result: subprocess.CompletedProcess[str], operation: str
    ) -> None:
        if result.returncode != 0:
            raise RepoCacheUnavailable(f"{operation} failed: {result.stderr.strip()}")

    @staticmethod
    def _safe_segment(value: str | None) -> str:
        raw = value or "default"
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in raw)

    @staticmethod
    def _safe_branch_component(value: str) -> str:
        return value.replace("/", "-").replace("\\", "-").replace(" ", "-")
