"""Value types returned by `SandboxClient`.

These are deliberately small and provider-agnostic. The full domain models
(`Workspace`, `Runtime`, …) live behind the client and are not part of the
import surface for downstream callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WorkspaceHandle:
    """Opaque handle for an active workspace.

    Internally it carries the IDs the client needs to talk to the service. The
    handle is the stable thing returned to callers — the client looks up the
    live :class:`Workspace` on every method call so we never serve stale data.

    `local_path` is populated only when the backend exposes the worktree on the
    host filesystem (the local-fs adapter). Daytona / docker backends leave it
    as `None` — callers that need to walk the tree directly should branch on
    this field rather than assume a host path exists.
    """

    workspace_id: str
    branch: str
    backend_kind: str
    local_path: str | None = None
    remote_path: str | None = None


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One entry from `SandboxClient.list_dir`.

    `size` is `None` when the backend does not cheaply report it (e.g. when the
    listing is produced from `ls -1Ap` over an exec call). Callers that need
    accurate sizes should `read_file` and measure, or branch on backend.
    """

    name: str
    is_dir: bool
    size: int | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    """One match from `SandboxClient.search`."""

    path: str
    line: int
    snippet: str


@dataclass(frozen=True, slots=True)
class GitStatus:
    """Parsed `git status --porcelain=v1` for a workspace."""

    branch: str
    is_clean: bool
    staged: tuple[str, ...] = field(default_factory=tuple)
    unstaged: tuple[str, ...] = field(default_factory=tuple)
    untracked: tuple[str, ...] = field(default_factory=tuple)
