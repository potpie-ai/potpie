"""High-level client over :class:`SandboxService`.

The library's intended import. Construct one per process (or per worker) with
:meth:`SandboxClient.from_env`; every other entry point in potpie should go
through it. The client never exposes the underlying providers — callers ask
for a :class:`WorkspaceHandle` and operate on it.

Helpers (`read_file`, `write_file`, `list_dir`, `search`, `commit`, `push`,
`status`, `diff`) take a fast path through the host filesystem when the
backend exposes one (`handle.local_path is not None`) and otherwise dispatch
through `exec`. That preserves backend portability without paying for a
sub-process when we already have direct fs access.
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
    """Pick the most useful error text from an :class:`ExecResult`.

    Daytona's ``process.exec`` collapses stdout and stderr into one
    ``result`` field (per SDK; the ExecutionArtifacts ``stdout`` mirrors
    ``result``), so ``ExecResult.stderr`` comes back empty even when the
    command actually failed. Falling back to ``stdout`` on those backends
    gives the LLM a real error message instead of the empty string the
    formatters used to emit. Local / Docker backends still split the
    streams correctly, so the stderr branch wins when it's populated.
    """
    return _decode(result.stderr) or _decode(result.stdout)


class SandboxClient:
    """Process-wide entry point for the sandbox library.

    Construction is cheap; the underlying providers / store / locks are built
    once via :func:`build_sandbox_container`. Multiple clients in a single
    process should share a container — use :meth:`from_container` for that.
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
        """Materialize the bare repo for ``repo`` and persist a `RepoCache`.

        Idempotent on `(provider_host, repo)` — repeat calls fetch the
        requested ref into the existing bare. Use this from parsing's
        READY hook (Phase 5 in the roadmap) so the cache row is in the
        store before the first agent call; subsequent
        :meth:`get_workspace` calls reuse the same on-disk bare without
        re-cloning.

        Raises ``RuntimeError`` if the underlying service was built
        without a `RepoCacheProvider` (e.g. Daytona-only mode today).
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
        """Materialise a worktree on `branch` and return a stable handle.

        `repo` is the canonical `owner/name` string; `repo_url` is a hint for
        adapters that don't already know how to derive the URL (the local-fs
        adapter falls back to `https://github.com/<repo>.git` when omitted).
        Idempotent on `(user, project, repo, branch, mode, scope)` — repeated
        calls return the same workspace until it's released.
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
        """High-level session entry point: cache + workspace in one call.

        Wraps :meth:`SandboxService.acquire_session` — ensures the
        parent `RepoCache` exists, then forks the workspace. Idempotent
        on the same `(user, project, repo, branch, mode, scope)` tuple,
        same as :meth:`get_workspace`.
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
        """Hibernate the runtime; keep the worktree.

        Same semantics as :meth:`release_workspace` but symmetric with
        :meth:`acquire_session`. Pass ``destroy_runtime=True`` to free
        the runtime fully — the workspace still survives until
        :meth:`destroy_workspace`.
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
        """Cheap liveness probe — does the backing workspace still exist?

        ``False`` ⇒ the underlying storage is gone (Daytona sandbox
        archived/deleted, local worktree removed by hand, store record
        purged). The intended caller (``ProjectSandbox.health_check``)
        runs this on every conversation message and re-creates via
        ``ensure()`` when it returns ``False``.

        Suppresses transient backend errors and reports them as ``False``
        so the caller's recovery path takes over instead of having to
        handle exceptions inline. See provider docstrings for details
        on what "alive" means per backend.
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
        """Write a file relative to the workspace root.

        Creates parent directories. Uses direct fs access on local-fs backends.

        Mutations re-resolve the workspace through the service so a stale
        handle whose workspace was destroyed raises ``WorkspaceNotFound``
        instead of silently recreating the directory. Reads keep the
        zero-roundtrip fast path because a stale read just fails on the
        underlying ``_read_bytes`` (file gone) — only writes can do
        damage by recreating a destroyed worktree on disk.
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
            hits.append(
                Hit(path=file_path, line=line_no, snippet=text.rstrip("\n"))
            )
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
        """Run the in-sandbox parser and return the reconstructed graph payload.

        Invokes ``potpie-parse`` (installed by the agent-sandbox image)
        inside the workspace; the runner streams NDJSON to stdout which
        we decode via :mod:`app.modules.intelligence.tools.sandbox.parser_wire`.

        ``repo_subdir`` lets a caller parse a subtree (rare — most parses
        operate on the whole worktree); ``None`` means parse the
        workspace root.

        ``max_output_bytes`` caps the parser's output size; the default
        of ``None`` means "no truncation," which is what parsing wants
        — a truncated stream would yield a corrupt graph. Adapters that
        cannot stream unbounded output should bump it explicitly.

        Raises :class:`SandboxOpError` if the parser exits non-zero or
        the runner's wire format is malformed.
        """
        # Imported lazily so the parser_wire module (stdlib-only by design)
        # doesn't need to be loaded for every SandboxClient — only callers
        # that actually parse pay the cost.
        from sandbox.api.parser_wire import (
            ParseArtifacts,
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
        """Stage and commit. Returns the new commit's full SHA.

        Without `paths`, stages everything (`git add -A`). With `paths`, only
        the listed files.

        ``author`` is ``(name, email)`` if you want to override; when
        omitted, the configured :class:`BotIdentityProvider` (if any)
        supplies the default so every agent commit lands under the bot
        identity. Without a provider AND without an explicit ``author``,
        the commit falls through to git's own ``user.name``/``user.email``
        config (the runtime spec already injected the bot env vars at
        ``get_or_create_runtime`` time, so this only matters for tests
        and embedded usage that bypass the service).

        Raises :class:`SandboxOpError` if there's nothing to commit.
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
            stderr = _decode(commit.stderr)
            stdout = _decode(commit.stdout)
            raise SandboxOpError(
                f"git commit failed: {stderr or stdout}", result=commit
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

        Auth is injected per-call from the configured
        :class:`RemoteAuthProvider`: a freshly resolved token (App
        installation token in production) is passed via
        ``-c http.<host>.extraheader='AUTHORIZATION: bearer …'`` so it
        never lands in ``.git/config`` and never persists past this
        invocation. The bare clone's ``origin`` URL was scrubbed at
        clone time on purpose (caches are shared across users), so
        without this re-injection, push to a private remote would fail.
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
        result = await self.exec(
            handle, cmd, command_kind=CommandKind.WRITE
        )
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
        """Pick the commit identity for `commit()`.

        Priority: explicit caller arg > BotIdentityProvider > None.
        Returning None lets git fall back to its configured user.name /
        user.email (which the service already populated into the runtime
        env from the same provider). The double-pass is intentional —
        callers that hold a SandboxClient but bypass the runtime spec
        (rare, but tests do it) still get the bot identity stamped.
        """
        if explicit is not None:
            return explicit
        identity = await self._service_bot_identity(handle)
        if identity is None:
            return None
        return (identity.name, identity.email)

    async def _service_bot_identity(self, handle: WorkspaceHandle):
        workspace = await self._service.get_workspace(handle.workspace_id)
        return await self._service.bot_identity_for(
            workspace.request.repo, user_id=workspace.request.user_id
        )

    async def _auth_pre_args(self, handle: WorkspaceHandle) -> list[str]:
        """Build the ``-c http.<host>.extraheader=…`` flags for `git push/fetch`.

        Returns an empty list when no :class:`RemoteAuthProvider` is
        wired or the provider declines (returns ``None``); the caller's
        git command then runs unauthenticated, which is the right
        behaviour for public repos and dev fixtures.

        The token is freshly resolved on every call — installation
        tokens expire in 1h, so caching at acquire-session time would
        silently break long-running conversations.
        """
        workspace = await self._service.get_workspace(handle.workspace_id)
        auth = await self._service.remote_auth_for(
            workspace.request.repo, user_id=workspace.request.user_id
        )
        if auth is None:
            return []
        host = workspace.request.repo.provider_host or "github.com"
        header = f"AUTHORIZATION: bearer {auth.token}"
        # `-c` overrides config for this invocation only; the token
        # never lands on disk. The host scope is important — a global
        # extraheader would match every HTTPS git operation in the
        # process, including unrelated remotes.
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
        """Open a PR from ``head_branch`` (defaults to ``handle.branch``)
        into ``base_branch`` via the configured `GitPlatformProvider`.

        Refuses to run if the workspace is not writable — opening a PR
        on an analysis workspace makes no sense and would surface a
        confusing platform error. Push the branch (`SandboxClient.push`)
        before calling this.
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
        """Post a comment on an existing PR via the configured `GitPlatformProvider`.

        Two shapes:

        * **Top-level** — only ``body`` and ``pr_number``. Posts a
          conversation comment on the PR.
        * **Inline** — set ``path`` and ``line`` (and optionally
          ``commit_id`` to pin the anchor). Posts a review comment at
          that file/line.

        Unlike :meth:`create_pull_request`, this does NOT require a
        writable workspace handle — review comments are commonly issued
        from analysis flows (the ``review-pr`` agent that has no
        worktree at all). The platform provider's auth chain is what
        determines attribution.
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
    """Reject absolute paths and `..` traversal; normalize separators.

    Returns the cleaned relative path. The runtime side does its own
    sandboxing — this is a defence-in-depth check at the client surface so
    callers can't accidentally read `/etc/passwd` even on an exec backend.
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
        # exec-based backend without a remote root means the runtime workdir
        # is already the worktree — fall through with the relative path.
        return rel if rel != "." else "."
    if rel == ".":
        return base
    return f"{base.rstrip('/')}/{rel}"


def _safe_local_path(root: str, rel: str) -> Path:
    """Resolve `rel` under `root`, rejecting symlink escapes.

    `_validate_relpath` already rejects `..` and absolute paths at the
    string level. This is the second line of defence for local-fs
    backends: a symlink inside the worktree could still resolve outside it
    (e.g. `link -> /etc`), and a direct `Path(root) / rel` read would
    happily follow it. Resolving both root and target and checking that
    the target stays underneath blocks that escape vector.

    `Path.resolve()` tolerates non-existent leaf components, so this works
    for writes that create new files — only the existing portion of the
    path is followed for symlinks.
    """
    root_path = Path(root).resolve()
    target = (root_path / rel).resolve()
    if target != root_path:
        try:
            target.relative_to(root_path)
        except ValueError:
            raise InvalidWorkspacePath(
                f"path {rel!r} resolves outside workspace root"
            )
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
