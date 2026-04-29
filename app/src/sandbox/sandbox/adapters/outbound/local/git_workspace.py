"""Local `.repos` workspace provider backed by git bare repos and worktrees."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse

from sandbox.adapters.outbound.local.auth import resolve_token
from sandbox.domain.errors import InvalidWorkspacePath, RepoAuthFailed, RepoCacheUnavailable
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


class LocalGitWorkspaceProvider:
    """Create durable local worktrees under a `.repos`-style directory."""

    kind = "local"

    def __init__(self, repos_base_path: str | Path | None = None) -> None:
        raw_base = repos_base_path or os.getenv("SANDBOX_REPOS_BASE_PATH") or ".repos"
        self.repos_base_path = Path(raw_base).expanduser().resolve()
        self.repos_base_path.mkdir(parents=True, exist_ok=True)
        self._by_id: dict[str, Workspace] = {}
        self._by_key: dict[str, str] = {}

    async def get_or_create_workspace(self, request: WorkspaceRequest) -> Workspace:
        key = request.key()
        existing_id = self._by_key.get(key)
        if existing_id:
            existing = self._by_id.get(existing_id)
            if existing and existing.location.local_path and Path(existing.location.local_path).exists():
                existing.last_used_at = utc_now()
                existing.updated_at = utc_now()
                return existing

        workspace = await asyncio.to_thread(self._create_workspace_sync, request)
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

    def _create_workspace_sync(self, request: WorkspaceRequest) -> Workspace:
        self._validate_repo_name(request.repo.repo_name)
        self._validate_ref(request.base_ref)
        if request.branch_name:
            self._validate_ref(request.branch_name)

        bare_path = self._bare_path(request.repo.repo_name)
        repo_url = request.repo.repo_url or self._default_github_url(request.repo.repo_name)
        # If the caller didn't pass an explicit token, fall through to the
        # adapter's resolver chain (env vars by default; production code
        # plugs in the richer GitHub-App / OAuth resolver via
        # `auth.set_token_resolver`).
        token = request.auth_token or resolve_token(
            repo_name=request.repo.repo_name, user_id=request.user_id
        )
        clone_url = self._authenticated_url(repo_url, token)
        self._ensure_bare_repo(bare_path, clone_url, request.base_ref)

        branch = request.branch_name or self._default_branch_name(request)
        worktree_path = self._worktree_path(request, branch)
        if worktree_path.exists():
            return self._workspace_from_path(request, worktree_path, branch)

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        if request.create_branch or request.mode is not WorkspaceMode.ANALYSIS:
            cmd = [
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
            ]
            result = self._run(cmd, timeout=120)
            if result.returncode != 0 and "already exists" in result.stderr:
                result = self._run(
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
            result = self._run(
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
        return self._workspace_from_path(request, worktree_path, branch)

    def _workspace_from_path(
        self, request: WorkspaceRequest, worktree_path: Path, branch: str
    ) -> Workspace:
        return Workspace(
            id=new_id("ws"),
            key=request.key(),
            repo_cache_id=None,
            request=request,
            location=WorkspaceLocation(
                kind=WorkspaceStorageKind.LOCAL_PATH,
                local_path=str(worktree_path.resolve()),
            ),
            backend_kind=self.kind,
            state=WorkspaceState.READY,
            metadata={"branch": branch},
        )

    def _ensure_bare_repo(self, bare_path: Path, clone_url: str, ref: str) -> None:
        if bare_path.exists() and (bare_path / "HEAD").exists():
            result = self._run(
                ["git", "-C", str(bare_path), "fetch", "origin", "--", ref],
                timeout=300,
            )
            if result.returncode != 0:
                raise RepoCacheUnavailable(
                    self._sanitize_git_error(f"git fetch failed: {result.stderr}")
                )
            return

        bare_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._run(
            ["git", "clone", "--bare", "--filter=blob:none", "--", clone_url, str(bare_path)],
            timeout=600,
        )
        if result.returncode != 0:
            message = self._sanitize_git_error(result.stderr)
            if "authentication" in message.lower() or "permission denied" in message.lower():
                raise RepoAuthFailed(message)
            raise RepoCacheUnavailable(f"git clone failed: {message}")
        fetch = self._run(["git", "-C", str(bare_path), "fetch", "origin", "--", ref], timeout=300)
        if fetch.returncode != 0:
            raise RepoCacheUnavailable(self._sanitize_git_error(fetch.stderr))

    def _remove_worktree_sync(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            result = self._run(["git", "-C", str(path), "worktree", "remove", "--force", str(path)], timeout=60)
            if result.returncode == 0:
                return
        except Exception:
            pass
        shutil.rmtree(path, ignore_errors=True)

    def _bare_path(self, repo_name: str) -> Path:
        return self.repos_base_path / repo_name / ".bare"

    def _worktrees_dir(self, repo_name: str) -> Path:
        return self.repos_base_path / repo_name / "worktrees"

    def _worktree_path(self, request: WorkspaceRequest, branch: str) -> Path:
        safe_user = self._safe_segment(request.user_id)
        scope = request.conversation_id or request.task_id or request.base_ref
        safe_scope = self._safe_segment(scope)
        safe_branch = self._safe_segment(branch)
        return self._worktrees_dir(request.repo.repo_name) / f"{safe_user}_{safe_scope}_{safe_branch}"

    def _default_branch_name(self, request: WorkspaceRequest) -> str:
        if request.mode is WorkspaceMode.ANALYSIS:
            return request.base_ref
        if request.mode is WorkspaceMode.TASK and request.task_id:
            return f"agent/task-{self._safe_branch_component(request.task_id)}"
        if request.conversation_id:
            return f"agent/edits-{self._safe_branch_component(request.conversation_id)}"
        return f"agent/workspace-{new_id('branch')}"

    @staticmethod
    def _run(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

    @staticmethod
    def _raise_if_failed(result: subprocess.CompletedProcess[str], operation: str) -> None:
        if result.returncode != 0:
            raise RepoCacheUnavailable(f"{operation} failed: {result.stderr.strip()}")

    @staticmethod
    def _default_github_url(repo_name: str) -> str:
        return f"https://github.com/{repo_name}.git"

    @staticmethod
    def _authenticated_url(repo_url: str, auth_token: str | None) -> str:
        if not auth_token:
            return repo_url
        parsed = urlparse(repo_url)
        if not parsed.scheme or not parsed.netloc:
            return repo_url
        host = parsed.hostname or parsed.netloc
        if parsed.port:
            host = f"{host}:{parsed.port}"
        netloc = f"x-access-token:{quote(auth_token, safe='')}@{host}"
        return parsed._replace(netloc=netloc).geturl()

    @staticmethod
    def _sanitize_git_error(message: str) -> str:
        return message.replace("x-access-token:", "x-access-token:***")

    @staticmethod
    def _validate_repo_name(repo_name: str) -> None:
        if not repo_name or "/" not in repo_name:
            raise ValueError("repo_name must be in owner/repo format")
        if repo_name.startswith("/") or "\\" in repo_name or ".." in repo_name:
            raise ValueError("repo_name contains unsafe path components")

    @staticmethod
    def _validate_ref(ref: str) -> None:
        if not ref:
            raise ValueError("git ref cannot be empty")
        if ".." in ref or "\n" in ref or "\r" in ref:
            raise ValueError("git ref contains unsafe characters")

    @staticmethod
    def _safe_segment(value: str | None) -> str:
        raw = value or "default"
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in raw)

    @staticmethod
    def _safe_branch_component(value: str) -> str:
        return value.replace("/", "-").replace("\\", "-").replace(" ", "-")
