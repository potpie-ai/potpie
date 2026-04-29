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


@dataclass(slots=True)
class RepoCache:
    id: str
    repo: RepoIdentity
    location: WorkspaceLocation
    backend_kind: str
    state: WorkspaceState = WorkspaceState.READY
    last_fetched_at: datetime | None = None
    last_used_at: datetime = field(default_factory=utc_now)
    size_bytes: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


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

