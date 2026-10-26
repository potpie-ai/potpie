"""High-level client for the sandbox library.

Construct a client with :meth:`SandboxClient.from_env`, or share a container
between clients with :meth:`SandboxClient.from_container`. Callers acquire a
:class:`WorkspaceHandle` and use it for file, command, and Git operations.
File helpers use the host filesystem when the handle exposes a local path.
Other backends use their native file operations or command execution.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Mapping, Self

from sandbox.api.types import FileEntry, GitStatus, Hit, WorkspaceHandle

if TYPE_CHECKING:
    from sandbox.api.parser_wire import ParseArtifacts
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.bootstrap.container import SandboxContainer, build_sandbox_container
from sandbox.bootstrap.settings import SandboxSettings
from sandbox.domain.errors import InvalidWorkspacePath, SandboxCoreError
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    ExecResult,
    PullRequest,
    PullRequestComment,
    PullRequestCommentResult,
    PullRequestRequest,
    RepoCache,
    RepoCacheRequest,
    RepoIdentity,
    RuntimeState,
    Workspace,
    WorkspaceMode,
    WorkspaceRequest,
)


class SandboxOpError(SandboxCoreError):
    """Raised when an exec-based helper fails (non-zero exit, missing tool)."""

    def __init__(self, message: str, *, result: ExecResult | None = None) -> None:
        super().__init__(message)
        self.result = result


def _err_payload(result: ExecResult) -> str:
    """Return stderr, or stdout when the backend combines both streams.

    Daytona puts command output in stdout even when execution fails.
    """
    return _decode(result.stderr) or _decode(result.stdout)


class SandboxClient:
    """Process-wide entry point for sandbox operations.

    Use :meth:`from_container` to share providers, stores, and locks across clients.
    """

    def __init__(self, container: SandboxContainer) -> None:
        self._container = container
        self._service: SandboxService = container.service

    @classmethod
    def from_env(cls, *, settings: SandboxSettings | None = None) -> Self:
        """Build a client from environment variables (see `bootstrap.settings`)."""
        return cls(build_sandbox_container(settings))

    @classmethod
    def from_container(cls, container: SandboxContainer) -> Self:
        """Wrap an existing :class:`SandboxContainer` (tests, embedded use)."""
        return cls(container)

    @property
    def container(self) -> SandboxContainer:
        return self._container

    async def ensure_snapshot_ready(self) -> None:
        """Register the runtime image and wait for it to become active.

        Daytona looks up the configured snapshot and builds it from the bundled
        Dockerfile if missing. A local provider has no snapshot and returns immediately.
        Call during startup to finish any image build before serving tool requests.
        Raises ``RuntimeUnavailable`` if the snapshot fails or cannot become ready.
        """
        provider = self._container.workspace_provider
        ensure = getattr(provider, "_ensure_snapshot", None)
        if ensure is None:
            return
        # `_ensure_snapshot` is sync and may block on snapshot.create
        # (multi-minute image build); push it off the event loop.
        await asyncio.to_thread(ensure)

    # ------------------------------------------------------------------
    # Repo / workspace lifecycle
    # ------------------------------------------------------------------
    async def ensure_repo_cache(
        self,
        *,
        user_id: str | None = None,
        repo: str,
        base_ref: str,
        repo_url: str | None = None,
        auth_token: str | None = None,
    ) -> RepoCache:
        """Fetch ``base_ref`` into the repository's bare clone and save its metadata.

        Calls with the same provider host and repository reuse the clone. Prepare the
        cache before the first agent call so :meth:`get_workspace` can reuse it.
        Raises ``RuntimeError`` if the service has no ``RepoCacheProvider``.
        """
        request = RepoCacheRequest(
            repo=RepoIdentity(repo_name=repo, repo_url=repo_url),
            base_ref=base_ref,
            user_id=user_id,
            auth_token=auth_token,
        )
        return await self._service.ensure_repo_cache(request)

    async def get_workspace(
        self,
        *,
        user_id: str,
        project_id: str,
        repo: str,
        branch: str,
        base_ref: str | None = None,
        create_branch: bool = False,
        auth_token: str | None = None,
        mode: WorkspaceMode = WorkspaceMode.EDIT,
        conversation_id: str | None = None,
        task_id: str | None = None,
        repo_url: str | None = None,
    ) -> WorkspaceHandle:
        """Create or reuse a worktree on ``branch`` and return its handle.

        ``repo`` is the canonical ``owner/name``. Local adapters default ``repo_url``
        to ``https://github.com/<repo>.git``. Calls with the same user, project,
        repository, branch, mode, and scope reuse the workspace until release.
        """
        request = WorkspaceRequest(
            user_id=user_id,
            project_id=project_id,
            repo=RepoIdentity(repo_name=repo, repo_url=repo_url),
            base_ref=base_ref or branch,
            mode=mode,
            conversation_id=conversation_id,
            task_id=task_id,
            branch_name=branch,
            create_branch=create_branch,
            auth_token=auth_token,
        )
        workspace = await self._service.get_or_create_workspace(request)
        return _handle_from_workspace(workspace)

    async def acquire_session(
        self,
        *,
        user_id: str,
        project_id: str,
        repo: str,
        branch: str,
        base_ref: str | None = None,
        create_branch: bool = False,
        auth_token: str | None = None,
        mode: WorkspaceMode = WorkspaceMode.EDIT,
        conversation_id: str | None = None,
        task_id: str | None = None,
        repo_url: str | None = None,
    ) -> WorkspaceHandle:
        """Prepare the repository cache and acquire a workspace.

        Uses :meth:`SandboxService.acquire_session`. The workspace has the same
        identity and reuse rules as :meth:`get_workspace`.
        """
        request = WorkspaceRequest(
            user_id=user_id,
            project_id=project_id,
            repo=RepoIdentity(repo_name=repo, repo_url=repo_url),
            base_ref=base_ref or branch,
            mode=mode,
            conversation_id=conversation_id,
            task_id=task_id,
            branch_name=branch,
            create_branch=create_branch,
            auth_token=auth_token,
        )
        workspace = await self._service.acquire_session(request)
        return _handle_from_workspace(workspace)

    async def release_session(
        self, handle: WorkspaceHandle, *, destroy_runtime: bool = False
    ) -> None:
        """Hibernate the runtime and keep the worktree.

        Pass ``destroy_runtime=True`` to remove the runtime. The workspace survives
        until :meth:`destroy_workspace`.
        """
        await self._service.release_session(
            handle.workspace_id, destroy_runtime=destroy_runtime
        )

    async def release_workspace(self, handle: WorkspaceHandle) -> None:
        """Hibernate the runtime; keep the worktree.

        Use this between agent turns or when a conversation goes idle but may
        resume. The worktree (and the cloned repo) stay around so the next
        :meth:`get_workspace` is fast. To actually delete the worktree, call
        :meth:`destroy_workspace`.
        """
        runtime = await self._container.store.find_runtime_by_workspace(
            handle.workspace_id
        )
        if runtime is None or runtime.state is RuntimeState.STOPPED:
            return
        if runtime.state is RuntimeState.DELETED:
            return
        await self._service.hibernate_runtime(runtime.id)

    async def destroy_workspace(self, handle: WorkspaceHandle) -> None:
        """Remove the worktree (and its runtime). The repo cache survives."""
        await self._service.destroy_workspace(handle.workspace_id)

    async def is_alive(self, handle: WorkspaceHandle) -> bool:
        """Check whether the backing workspace still exists.

        Returns ``False`` for missing storage or transient backend errors.
        ``ProjectSandbox.health_check`` uses this result to recreate the workspace.
        Providers define the storage checks for each backend.
        """
        return await self._service.is_workspace_alive(handle.workspace_id)

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------
    async def exec(
        self,
        handle: WorkspaceHandle,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: int | None = None,
        command_kind: CommandKind = CommandKind.READ,
        max_output_bytes: int | None = None,
        stdin: bytes | None = None,
        shell: bool = False,
    ) -> ExecResult:
        """Run a command inside the workspace's runtime.

        `command_kind` tells the service whether to take the per-workspace
        write lock; pass `WRITE` for anything that mutates the tree, `READ`
        for queries. The default is conservative (`READ`) so callers must
        opt in to mutation.
        """
        request = ExecRequest(
            cmd=tuple(cmd),
            cwd=cwd,
            env=dict(env or {}),
            timeout_s=timeout_s,
            command_kind=command_kind,
            max_output_bytes=max_output_bytes,
            stdin=stdin,
            shell=shell,
        )
        return await self._service.exec(handle.workspace_id, request)

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    async def read_file(
        self,
        handle: WorkspaceHandle,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read a file relative to the workspace root.

        Three paths, in order of preference:
        1. Local-fs fast path when the backend exposes the worktree on
           the host (no exec round-trip).
        2. Runtime's native fs (Daytona's ``sandbox.fs.download_file``)
           when the runtime provider exposes ``read_bytes``.
        3. ``cat`` over exec for backends without native fs (Docker,
           future remote runtimes).
        """
        rel = _validate_relpath(path)
        if handle.local_path is not None:
            full = _safe_local_path(handle.local_path, rel)
            return await asyncio.to_thread(_read_bytes, full, max_bytes)
        target = _posix_join(handle.remote_path, rel)
        native = await self._service.fs_read_file(handle.workspace_id, target)
        if native is not None:
            if max_bytes is not None and len(native) > max_bytes:
                return native[:max_bytes]
            return native
        result = await self.exec(
            handle,
            ["cat", "--", target],
            command_kind=CommandKind.READ,
            max_output_bytes=max_bytes,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"read_file({path!r}) failed: {_err_payload(result)}",
                result=result,
            )
        return result.stdout

    async def write_file(
        self,
        handle: WorkspaceHandle,
        path: str,
        content: bytes | str,
    ) -> None:
        """Write a file relative to the workspace root and create parent directories.

        Local writes reload the workspace from the service. A destroyed workspace
        raises ``WorkspaceNotFound`` rather than recreating its directory from a stale
        handle. Local reads use the handle directly and fail if the file is gone.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        rel = _validate_relpath(path)
        if handle.local_path is not None:
            workspace = await self._service.get_workspace(handle.workspace_id)
            local_root = workspace.location.local_path
            if local_root is not None:
                full = _safe_local_path(local_root, rel)
                await asyncio.to_thread(_write_bytes, full, content)
                return
        target = _posix_join(handle.remote_path, rel)
        if await self._service.fs_write_file(handle.workspace_id, target, content):
            return
        # Generic fallback for backends without native fs. Daytona's
        # ``process.exec`` has no stdin parameter, so the previous
        # ``cat > path`` + stdin path silently wrote empty files. Base64
        # piped through ``base64 -d`` works on every shell image we
        # ship (busybox, coreutils, alpine) and survives binary content
        # because it never enters the shell as raw bytes.
        parent = str(PurePosixPath(target).parent)
        await self.exec(
            handle,
            ["mkdir", "-p", "--", parent],
            command_kind=CommandKind.WRITE,
        )
        encoded = base64.b64encode(content).decode("ascii")
        result = await self.exec(
            handle,
            [
                "sh",
                "-c",
                f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(target)}",
            ],
            command_kind=CommandKind.WRITE,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"write_file({path!r}) failed: {_err_payload(result)}",
                result=result,
            )

    async def list_dir(
        self, handle: WorkspaceHandle, path: str = "."
    ) -> list[FileEntry]:
        """List one directory level (no recursion).

        On local-fs backends `size` is populated; on exec-based backends it's
        `None` because we use `ls -1Ap` for portability across busybox and
        coreutils.
        """
        rel = _validate_relpath(path, allow_dot=True)
        if handle.local_path is not None:
            full = _safe_local_path(handle.local_path, rel)
            return await asyncio.to_thread(_scandir, full)
        target = _posix_join(handle.remote_path, rel)
        native = await self._service.fs_list_dir(handle.workspace_id, target)
        if native is not None:
            return [
                FileEntry(name=name, is_dir=is_dir, size=size)
                for (name, is_dir, size) in native
            ]
        result = await self.exec(
            handle,
            ["ls", "-1Ap", "--", target],
            command_kind=CommandKind.READ,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"list_dir({path!r}) failed: {_err_payload(result)}",
                result=result,
            )
        entries: list[FileEntry] = []
        for line in _decode(result.stdout).splitlines():
            if not line:
                continue
            is_dir = line.endswith("/")
            name = line[:-1] if is_dir else line
            entries.append(FileEntry(name=name, is_dir=is_dir))
        return entries

    async def search(
        self,
        handle: WorkspaceHandle,
        pattern: str,
        *,
        glob: str | None = None,
        case: bool = False,
        max_hits: int | None = 200,
        path: str | None = None,
    ) -> list[Hit]:
        """Ripgrep across the worktree.

        Requires `rg` on PATH inside the runtime (Phase 1 of the integration
        plan ships it preinstalled in the agent-sandbox image). Returns at
        most `max_hits` hits to bound LLM token use.
        """
        cmd = [
            "rg",
            "--json",
            "--no-heading",
            "--with-filename",
            "--line-number",
        ]
        if not case:
            cmd.append("--smart-case")
        else:
            cmd.append("--case-sensitive")
        if glob:
            cmd.extend(["--glob", glob])
        cmd.append("--")
        cmd.append(pattern)
        if path:
            rel = _validate_relpath(path, allow_dot=True)
            cmd.append(_posix_join(handle.remote_path, rel))
        result = await self.exec(handle, cmd, command_kind=CommandKind.READ)
        # rg exit codes: 0 = matches, 1 = no matches, 2 = error.
        if result.exit_code not in (0, 1):
            raise SandboxOpError(
                f"search({pattern!r}) failed: {_err_payload(result)}",
                result=result,
            )
        hits: list[Hit] = []
        for raw in _decode(result.stdout).splitlines():
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            line_no = int(data.get("line_number") or 0)
            file_path = (data.get("path") or {}).get("text") or ""
            text = (data.get("lines") or {}).get("text") or ""
            hits.append(Hit(path=file_path, line=line_no, snippet=text.rstrip("\n")))
            if max_hits is not None and len(hits) >= max_hits:
                break
        return hits

    # ------------------------------------------------------------------
    # In-sandbox parser
    # ------------------------------------------------------------------
    async def parse_repo(
        self,
        handle: WorkspaceHandle,
        *,
        repo_subdir: str | None = None,
        timeout_s: int = 600,
        max_output_bytes: int | None = None,
    ) -> "ParseArtifacts":
        """Run ``potpie-parse`` and decode its NDJSON output with ``sandbox.api.parser_wire``.

        ``repo_subdir`` selects a subtree. ``None`` parses the workspace root.
        ``max_output_bytes=None`` allows the complete stream. Backends with an output
        limit must set a large enough value because truncation corrupts the graph.
        Raises :class:`SandboxOpError` on timeout, truncation, a nonzero exit code,
        or malformed parser output.
        """
        # Load parser support only when a caller requests parsing.
        from sandbox.api.parser_wire import (
            WireFormatError,
            parse_stream,
        )

        target = "."
        if repo_subdir is not None:
            target = _validate_relpath(repo_subdir, allow_dot=True)

        result = await self.exec(
            handle,
            ["potpie-parse", target],
            command_kind=CommandKind.READ,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )
        if result.timed_out:
            raise SandboxOpError(
                f"potpie-parse timed out after {timeout_s}s "
                f"(workspace={handle.workspace_id}, target={target!r})",
                result=result,
            )
        if result.truncated:
            raise SandboxOpError(
                "potpie-parse output was truncated; bump max_output_bytes "
                "(parsing requires the full stream — partial NDJSON yields "
                "a corrupt graph)",
                result=result,
            )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"potpie-parse exited {result.exit_code}: {_err_payload(result)}",
                result=result,
            )

        try:
            artifacts: ParseArtifacts = parse_stream(
                _decode(result.stdout).splitlines()
            )
        except WireFormatError as exc:
            raise SandboxOpError(
                f"potpie-parse emitted malformed NDJSON: {exc}",
                result=result,
            ) from exc
        return artifacts

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------
    async def status(self, handle: WorkspaceHandle) -> GitStatus:
        result = await self.exec(
            handle,
            ["git", "status", "--porcelain=v1", "--branch"],
            command_kind=CommandKind.READ,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"git status failed: {_err_payload(result)}", result=result
            )
        return _parse_status(_decode(result.stdout), default_branch=handle.branch)

    async def diff(
        self,
        handle: WorkspaceHandle,
        *,
        base_ref: str | None = None,
        paths: list[str] | None = None,
    ) -> str:
        cmd = ["git", "diff"]
        if base_ref:
            cmd.append(base_ref)
        cmd.append("--")
        if paths:
            cmd.extend(_validate_relpath(p) for p in paths)
        result = await self.exec(handle, cmd, command_kind=CommandKind.READ)
        if result.exit_code != 0:
            raise SandboxOpError(
                f"git diff failed: {_err_payload(result)}", result=result
            )
        return _decode(result.stdout)

    async def commit(
        self,
        handle: WorkspaceHandle,
        message: str,
        *,
        paths: list[str] | None = None,
        author: tuple[str, str] | None = None,
    ) -> str:
        """Stage files and commit, returning the new commit's full SHA.

        With ``paths``, stage only those files. Otherwise, use ``git add -A``.
        An explicit ``author`` supplies the name and email. Otherwise, use the service's
        ``BotIdentityProvider`` if configured, then fall back to Git's configuration.
        Resolve the provider here as well as during runtime setup so callers that
        bypass setup still use the bot identity.
        Raises :class:`SandboxOpError` if Git cannot add or commit files.
        """
        if paths:
            for raw in paths:
                rel = _validate_relpath(raw)
                add = await self.exec(
                    handle,
                    ["git", "add", "--", rel],
                    command_kind=CommandKind.WRITE,
                )
                if add.exit_code != 0:
                    raise SandboxOpError(
                        f"git add {rel!r} failed: {_err_payload(add)}",
                        result=add,
                    )
        else:
            add = await self.exec(
                handle,
                ["git", "add", "-A"],
                command_kind=CommandKind.WRITE,
            )
            if add.exit_code != 0:
                raise SandboxOpError(
                    f"git add -A failed: {_err_payload(add)}", result=add
                )

        env: dict[str, str] = {}
        resolved = await self._resolve_author(handle, author)
        if resolved is not None:
            name, email = resolved
            env.update(
                {
                    "GIT_AUTHOR_NAME": name,
                    "GIT_AUTHOR_EMAIL": email,
                    "GIT_COMMITTER_NAME": name,
                    "GIT_COMMITTER_EMAIL": email,
                }
            )
        commit = await self.exec(
            handle,
            ["git", "commit", "-m", message],
            env=env,
            command_kind=CommandKind.WRITE,
        )
        if commit.exit_code != 0:
            raise SandboxOpError(
                f"git commit failed: {_err_payload(commit)}", result=commit
            )
        sha = await self.exec(
            handle,
            ["git", "rev-parse", "HEAD"],
            command_kind=CommandKind.READ,
        )
        if sha.exit_code != 0:
            raise SandboxOpError(
                f"git rev-parse failed: {_err_payload(sha)}", result=sha
            )
        return _decode(sha.stdout).strip()

    async def push(
        self,
        handle: WorkspaceHandle,
        *,
        remote: str = "origin",
        set_upstream: bool = True,
        force: bool = False,
    ) -> None:
        """Push the worktree branch to ``remote``.

        Resolve a fresh ``RemoteAuthProvider`` token for each call and pass its HTTP
        Basic header through Git's ``-c`` option. The username is ``x-access-token``.
        The token stays out of the shared bare clone's ``.git/config``.
        """
        # `-c http.<host>.extraheader=...` has to come BEFORE the `push`
        # subcommand for git to honour it. Keep the option list separate
        # so we can safely add more pre-flags later.
        pre_args = await self._auth_pre_args(handle)
        cmd = ["git", *pre_args, "push"]
        if force:
            cmd.append("--force-with-lease")
        if set_upstream:
            cmd.append("--set-upstream")
        cmd.extend([remote, f"HEAD:{handle.branch}"])
        result = await self.exec(handle, cmd, command_kind=CommandKind.WRITE)
        if result.exit_code != 0:
            raise SandboxOpError(
                f"git push failed: {_err_payload(result)}", result=result
            )

    # ------------------------------------------------------------------
    # Identity / auth helpers
    # ------------------------------------------------------------------
    async def _resolve_author(
        self,
        handle: WorkspaceHandle,
        explicit: tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        """Use the explicit author, then the bot identity, then Git's configuration."""
        if explicit is not None:
            return explicit
        workspace = await self._service.get_workspace(handle.workspace_id)
        identity = await self._service.bot_identity_for(
            workspace.request.repo, user_id=workspace.request.user_id
        )
        if identity is None:
            return None
        return (identity.name, identity.email)

    async def _auth_pre_args(self, handle: WorkspaceHandle) -> list[str]:
        """Build a host-scoped HTTP Basic header for this Git invocation.

        Resolve the token on each call because installation tokens expire. The header
        uses ``x-access-token`` as its username, matching GitHub's HTTPS clone scheme.
        Return no flags when the auth provider is absent or declines the request.
        """
        workspace = await self._service.get_workspace(handle.workspace_id)
        auth = await self._service.remote_auth_for(
            workspace.request.repo, user_id=workspace.request.user_id
        )
        if auth is None:
            return []
        host = workspace.request.repo.provider_host or "github.com"
        basic = base64.b64encode(f"x-access-token:{auth.token}".encode()).decode(
            "ascii"
        )
        header = f"AUTHORIZATION: basic {basic}"
        # Scope the header to this host and invocation so unrelated remotes
        # do not receive the token and Git does not save it to disk.
        return ["-c", f"http.https://{host}/.extraheader={header}"]

    # ------------------------------------------------------------------
    # Git platform (PRs / reviews / comments)
    # ------------------------------------------------------------------
    async def create_pull_request(
        self,
        handle: WorkspaceHandle,
        *,
        repo: str,
        title: str,
        body: str,
        base_branch: str,
        head_branch: str | None = None,
        repo_url: str | None = None,
        reviewers: list[str] | None = None,
        labels: list[str] | None = None,
        auth_token: str | None = None,
    ) -> PullRequest:
        """Open a pull request through the configured ``GitPlatformProvider``.

        ``head_branch`` defaults to the workspace branch. Requires a writable handle.
        Call :meth:`push` first to publish the branch.
        """
        if not handle.capabilities.writable:
            raise SandboxOpError(
                "create_pull_request requires a writable workspace; "
                "this handle was acquired with read-only capabilities."
            )
        request = PullRequestRequest(
            repo=RepoIdentity(repo_name=repo, repo_url=repo_url),
            title=title,
            body=body,
            head_branch=head_branch or handle.branch,
            base_branch=base_branch,
            reviewers=tuple(reviewers or ()),
            labels=tuple(labels or ()),
            auth_token=auth_token,
        )
        return await self._service.create_pull_request(request)

    async def comment_on_pull_request(
        self,
        *,
        repo: str,
        pr_number: int,
        body: str,
        path: str | None = None,
        line: int | None = None,
        commit_id: str | None = None,
        repo_url: str | None = None,
        auth_token: str | None = None,
    ) -> PullRequestCommentResult:
        """Post a conversation or inline comment through ``GitPlatformProvider``.

        Set both ``path`` and ``line`` for an inline review comment. ``commit_id`` can
        pin it to a commit. Omit both to post a conversation comment.
        No workspace handle is required. The provider's authentication determines
        the comment's author.
        """
        if (path is None) != (line is None):
            raise SandboxOpError(
                "comment_on_pull_request: pass both `path` and `line` "
                "(inline comment) or neither (top-level comment)"
            )
        request = PullRequestComment(
            repo=RepoIdentity(repo_name=repo, repo_url=repo_url),
            pr_number=pr_number,
            body=body,
            path=path,
            line=line,
            commit_id=commit_id,
            auth_token=auth_token,
        )
        return await self._service.comment_on_pull_request(request)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
def _handle_from_workspace(ws: Workspace) -> WorkspaceHandle:
    branch = ws.metadata.get("branch") or ws.request.branch_name or ws.request.base_ref
    return WorkspaceHandle(
        workspace_id=ws.id,
        branch=branch,
        backend_kind=ws.backend_kind,
        local_path=ws.location.local_path,
        remote_path=ws.location.remote_path,
        capabilities=ws.capabilities,
    )


def _validate_relpath(path: str, *, allow_dot: bool = False) -> str:
    """Normalize separators and reject absolute paths or ``..`` traversal.

    This client check also applies to backends that execute commands remotely.
    """
    if path is None:
        raise InvalidWorkspacePath("path is required")
    norm = path.replace("\\", "/")
    if norm in ("", "."):
        if allow_dot:
            return "."
        raise InvalidWorkspacePath("path is required")
    if os.path.isabs(norm):
        raise InvalidWorkspacePath(f"path must be relative: {path!r}")
    parts = PurePosixPath(norm).parts
    if any(p == ".." for p in parts):
        raise InvalidWorkspacePath(f"path escapes workspace: {path!r}")
    cleaned = "/".join(p for p in parts if p not in ("", "."))
    if not cleaned:
        if allow_dot:
            return "."
        raise InvalidWorkspacePath(f"path is empty: {path!r}")
    return cleaned


def _posix_join(base: str | None, rel: str) -> str:
    if base is None:
        # Without a remote root, the runtime already runs in the worktree.
        return rel
    if rel == ".":
        return base
    return f"{base.rstrip('/')}/{rel}"


def _safe_local_path(root: str, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` and reject symlinks that escape it.

    ``_validate_relpath`` rejects absolute paths and ``..`` first. Resolving both
    paths catches symlinks that point outside the worktree. ``Path.resolve`` also
    accepts nonexistent leaf components so writes can create new files.
    """
    root_path = Path(root).resolve()
    target = (root_path / rel).resolve()
    if target != root_path:
        try:
            target.relative_to(root_path)
        except ValueError:
            raise InvalidWorkspacePath(f"path {rel!r} resolves outside workspace root")
    return target


def _read_bytes(path: Path, max_bytes: int | None) -> bytes:
    if not path.exists():
        raise SandboxOpError(f"file not found: {path}")
    if path.is_dir():
        raise SandboxOpError(f"not a file: {path}")
    if max_bytes is None:
        return path.read_bytes()
    with path.open("rb") as f:
        return f.read(max_bytes)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _scandir(root: Path) -> list[FileEntry]:
    if not root.exists():
        raise SandboxOpError(f"directory not found: {root}")
    if not root.is_dir():
        raise SandboxOpError(f"not a directory: {root}")
    out: list[FileEntry] = []
    for entry in os.scandir(root):
        try:
            stat = entry.stat(follow_symlinks=False)
            size = stat.st_size if entry.is_file() else None
        except OSError:
            size = None
        out.append(FileEntry(name=entry.name, is_dir=entry.is_dir(), size=size))
    out.sort(key=lambda e: (not e.is_dir, e.name))
    return out


def _parse_status(text: str, *, default_branch: str) -> GitStatus:
    """Parse `git status --porcelain=v1 --branch` output.

    Lines:
      `## main...origin/main`            → header
      `M  path`                          → staged modify
      ` M path`                          → unstaged modify
      `MM path`                          → both
      `?? path`                          → untracked
    """
    branch = default_branch
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("## "):
            header = line[3:]
            # `branch...remote` or `branch (no commits)` etc.
            for token in (header.split("...", 1)[0], header.split(" ", 1)[0]):
                if token and token != "HEAD":
                    branch = token
                    break
            continue
        if len(line) < 3:
            continue
        idx = line[:2]
        path = line[3:]
        if idx == "??":
            untracked.append(path)
            continue
        if idx[0] not in (" ", "?"):
            staged.append(path)
        if idx[1] not in (" ", "?"):
            unstaged.append(path)
    return GitStatus(
        branch=branch,
        is_clean=not (staged or unstaged or untracked),
        staged=tuple(staged),
        unstaged=tuple(unstaged),
        untracked=tuple(untracked),
    )


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace") if data else ""


__all__ = [
    "SandboxClient",
    "SandboxOpError",
]
