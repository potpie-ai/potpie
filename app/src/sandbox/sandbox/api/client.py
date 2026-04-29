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
import json
import os
import shlex
from pathlib import Path, PurePosixPath
from typing import Mapping, Self

from sandbox.api.types import FileEntry, GitStatus, Hit, WorkspaceHandle
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.bootstrap.container import SandboxContainer, build_sandbox_container
from sandbox.bootstrap.settings import SandboxSettings
from sandbox.domain.errors import InvalidWorkspacePath, SandboxCoreError
from sandbox.domain.models import (
    CommandKind,
    ExecRequest,
    ExecResult,
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

        Uses direct fs access on local-fs backends; `cat` over exec elsewhere.
        """
        rel = _validate_relpath(path)
        if handle.local_path is not None:
            full = _safe_local_path(handle.local_path, rel)
            return await asyncio.to_thread(_read_bytes, full, max_bytes)
        result = await self.exec(
            handle,
            ["cat", "--", _posix_join(handle.remote_path, rel)],
            command_kind=CommandKind.READ,
            max_output_bytes=max_bytes,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"read_file({path!r}) failed: {_decode(result.stderr)}",
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
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        rel = _validate_relpath(path)
        if handle.local_path is not None:
            full = _safe_local_path(handle.local_path, rel)
            await asyncio.to_thread(_write_bytes, full, content)
            return
        target = _posix_join(handle.remote_path, rel)
        # Ensure parent dir then stream content via stdin to `tee`.
        parent = str(PurePosixPath(target).parent)
        await self.exec(
            handle,
            ["mkdir", "-p", "--", parent],
            command_kind=CommandKind.WRITE,
        )
        result = await self.exec(
            handle,
            ["sh", "-c", f"cat > {shlex.quote(target)}"],
            stdin=content,
            command_kind=CommandKind.WRITE,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"write_file({path!r}) failed: {_decode(result.stderr)}",
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
        result = await self.exec(
            handle,
            ["ls", "-1Ap", "--", target],
            command_kind=CommandKind.READ,
        )
        if result.exit_code != 0:
            raise SandboxOpError(
                f"list_dir({path!r}) failed: {_decode(result.stderr)}",
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
            cmd.append(_validate_relpath(path, allow_dot=True))
        result = await self.exec(handle, cmd, command_kind=CommandKind.READ)
        # rg exit codes: 0 = matches, 1 = no matches, 2 = error.
        if result.exit_code not in (0, 1):
            raise SandboxOpError(
                f"search({pattern!r}) failed: {_decode(result.stderr)}",
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
                f"git status failed: {_decode(result.stderr)}", result=result
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
                f"git diff failed: {_decode(result.stderr)}", result=result
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
        the listed files. `author` is `(name, email)` if you want to override.
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
                        f"git add {rel!r} failed: {_decode(add.stderr)}",
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
                    f"git add -A failed: {_decode(add.stderr)}", result=add
                )

        env: dict[str, str] = {}
        if author is not None:
            name, email = author
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
                f"git rev-parse failed: {_decode(sha.stderr)}", result=sha
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
        cmd = ["git", "push"]
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
                f"git push failed: {_decode(result.stderr)}", result=result
            )


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
