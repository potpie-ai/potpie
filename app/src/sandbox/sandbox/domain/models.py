"""Domain models for sandbox workspace and runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class WorkspaceMode(str, Enum):
    ANALYSIS = "analysis"
    EDIT = "edit"
    TASK = "task"


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Explicit per-workspace capability flags.

    Replaces the implicit branching on ``WorkspaceMode`` that used to be
    duplicated across adapters. Construct via ``Capabilities.from_mode``
    so the mapping stays in one place; adapters and the application
    service read these flags instead of re-deriving from the enum.

    * ``writable`` — workspace permits mutating operations (file writes,
      ``git commit`` etc.). Read-only callers (parsing, analysis tools)
      should hold a workspace with ``writable=False`` so the runtime
      provider can refuse write commands at the boundary.
    * ``isolated`` — workspace lives on its own branch and can't stomp
      another caller working on the same base branch.
    * ``persistent`` — workspace state survives runtime destruction;
      false would mean ephemeral (e.g. a future "scratch" mode).
    """

    writable: bool = False
    isolated: bool = False
    persistent: bool = True

    @classmethod
    def from_mode(cls, mode: "WorkspaceMode") -> "Capabilities":
        if mode is WorkspaceMode.ANALYSIS:
            return cls(writable=False, isolated=False, persistent=True)
        # EDIT and TASK both fork their own branch; the runtime is
        # write-capable. Any future mode should be added here explicitly.
        return cls(writable=True, isolated=True, persistent=True)


class WorkspaceState(str, Enum):
    CREATING = "creating"
    READY = "ready"
    DELETED = "deleted"
    ERROR = "error"


class WorkspaceStorageKind(str, Enum):
    LOCAL_PATH = "local_path"
    DOCKER_VOLUME = "docker_volume"
    DAYTONA_SANDBOX = "daytona_sandbox"
    REMOTE_PATH = "remote_path"


class RuntimeState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    DELETED = "deleted"
    ERROR = "error"


class NetworkMode(str, Enum):
    NONE = "none"
    LIMITED = "limited"
    DEFAULT = "default"


class CommandKind(str, Enum):
    READ = "read"
    WRITE = "write"
    INSTALL = "install"
    TEST = "test"

    @property
    def mutates_workspace(self) -> bool:
        return self is not CommandKind.READ


@dataclass(frozen=True, slots=True)
class RepoIdentity:
    repo_name: str
    repo_url: str | None = None
    provider: str = "github"
    provider_host: str = "github.com"


@dataclass(frozen=True, slots=True)
class WorkspaceRequest:
    user_id: str
    project_id: str
    repo: RepoIdentity
    base_ref: str
    mode: WorkspaceMode = WorkspaceMode.EDIT
    conversation_id: str | None = None
    task_id: str | None = None
    branch_name: str | None = None
    create_branch: bool = False
    auth_token: str | None = field(default=None, repr=False, compare=False)
    pinned: bool = False

    def key(self) -> str:
        if self.mode is WorkspaceMode.ANALYSIS:
            scope = self.base_ref
        elif self.mode is WorkspaceMode.EDIT:
            scope = self.conversation_id
        else:
            scope = self.task_id
        return "|".join(
            [
                self.user_id,
                self.project_id,
                self.repo.provider_host,
                self.repo.repo_name,
                self.mode.value,
                scope or "",
            ]
        )


@dataclass(frozen=True, slots=True)
class WorkspaceLocation:
    kind: WorkspaceStorageKind
    local_path: str | None = None
    remote_path: str | None = None
    docker_volume: str | None = None
    backend_workspace_id: str | None = None


@dataclass(frozen=True, slots=True)
class RepoCacheRequest:
    """Request to ensure a durable bare repo for ``repo`` is up to date.

    A `RepoCache` is shared across users for the same `(provider_host,
    repo_name)`; auth and per-user access control happen at the
    `WorkspaceRequest` level. `base_ref` tells the provider which ref to
    fetch into the existing bare repo so worktrees can be created off it
    later.
    """

    repo: RepoIdentity
    base_ref: str
    user_id: str | None = None
    auth_token: str | None = field(default=None, repr=False, compare=False)

    def key(self) -> str:
        return "|".join([self.repo.provider_host, self.repo.repo_name])


@dataclass(slots=True)
class RepoCache:
    """Durable bare git mirror shared across workspaces.

    `key` is the canonical identifier the store uses for lookup
    (`<host>|<repo_name>`); see `RepoCacheRequest.key()`. Multiple
    workspaces can fork worktrees off the same cache without re-cloning.
    """

    id: str
    key: str
    repo: RepoIdentity
    location: WorkspaceLocation
    backend_kind: str
    state: WorkspaceState = WorkspaceState.READY
    last_fetched_at: datetime | None = None
    last_used_at: datetime = field(default_factory=utc_now)
    size_bytes: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Workspace:
    id: str
    key: str
    repo_cache_id: str | None
    request: WorkspaceRequest
    location: WorkspaceLocation
    backend_kind: str
    state: WorkspaceState = WorkspaceState.READY
    dirty: bool = False
    pinned_until: datetime | None = None
    last_used_at: datetime = field(default_factory=utc_now)
    size_bytes: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, str] = field(default_factory=dict)
    capabilities: Capabilities = field(default_factory=Capabilities)


@dataclass(frozen=True, slots=True)
class Mount:
    source: str
    target: str
    writable: bool = False


@dataclass(frozen=True, slots=True)
class ResourceHints:
    cpu: float | None = None
    memory_mb: int | None = None
    disk_gb: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    image: str
    workdir: str
    mounts: tuple[Mount, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    resources: ResourceHints | None = None
    network: NetworkMode = NetworkMode.LIMITED
    placement: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    workspace_id: str
    image: str = "python:3.12-slim"
    env: Mapping[str, str] = field(default_factory=dict)
    writable: bool = True
    network: NetworkMode = NetworkMode.LIMITED
    timeout_s: int | None = None
    resources: ResourceHints | None = None


@dataclass(slots=True)
class Runtime:
    id: str
    workspace_id: str
    backend_kind: str
    backend_runtime_id: str | None
    spec: RuntimeSpec
    state: RuntimeState = RuntimeState.RUNNING
    last_started_at: datetime | None = field(default_factory=utc_now)
    last_used_at: datetime = field(default_factory=utc_now)
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ExecRequest:
    cmd: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: int | None = None
    stdin: bytes | None = None
    max_output_bytes: int | None = None
    command_kind: CommandKind = CommandKind.READ
    shell: bool = False


@dataclass(frozen=True, slots=True)
class ExecResult:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class ExecChunk:
    stream: str
    data: bytes


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    snapshot: bool = False
    preview_url: bool = False
    interactive_session: bool = False


@dataclass(frozen=True, slots=True)
class PullRequestRequest:
    """Request to open a pull request from the agent's worktree branch."""

    repo: RepoIdentity
    title: str
    body: str
    head_branch: str
    base_branch: str
    reviewers: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    auth_token: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PullRequest:
    """Result of a successful PR creation.

    `id` is the platform-side PR id (numeric on GitHub/GitLab); `url`
    is the human-readable URL the agent can surface to the user.
    """

    id: int | str
    url: str
    title: str
    head_branch: str
    base_branch: str
    backend_kind: str

