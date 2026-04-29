# Sandbox Core Setup

## Context

Potpie agents need an isolated place to inspect and change user repositories:
clone once, reuse the repo later, create worktrees for new work, run commands,
edit files, run tests, and resume the same state in a later agent session.

The important distinction is that "sandbox" is not one thing:

- A **repo cache** is the durable git mirror. It should avoid recloning.
- A **workspace** is the durable working tree or worktree where the agent sees
  files and can make changes.
- A **runtime** is the compute environment that runs commands against a
  workspace. It may be stopped, recreated, or moved without deleting the
  workspace.

The current codebase already has part of this model:

- `app/modules/repo_manager/repo_manager.py` manages bare repositories and
  worktrees under `.repos/`.
- `app/modules/repo_manager/sync_helper.py` has the GitHub App -> OAuth -> env
  token auth chain for cloning/fetching.
- `checkout_worktree_branch`, `apply_changes`, `git_commit`, and `bash_command`
  already expect a repo-manager-backed worktree.
- `app/modules/utils/gvisor_runner.py` is useful for read-only analysis, but it
  is not the right foundation for write-capable agent execution because it is
  read-only by design and may fall back to regular subprocess execution.

This design keeps the repo/worktree lifecycle separate from the execution
backend. The application layer asks for a workspace, then attaches a runtime to
it.

---

## Goals

- Clone a user repo once and reuse it across agent sessions.
- Keep durable agent work in a workspace/worktree, especially for feature work.
- Let multiple execution backends run against the same workspace abstraction:
  local container, Docker, Daytona, E2B, or future providers.
- Allow the compute runtime to be hibernated or destroyed without losing the
  repo cache or workspace state.
- Support explicit eviction policies for runtimes, workspaces, and repo caches.
- Keep application code backend-agnostic without hiding important persistence
  semantics.

## Non-goals

- Building a human developer environment like VS Code Remote, Coder, or devpod.
- Making every backend expose identical networking, region, quota, snapshot, or
  preview behavior.
- Replacing the existing read-only `gvisor_runner.py` path immediately.
- Treating snapshots as the only persistence mechanism. Git worktrees and
  volumes are first-class persistence.

---

## Core Model

### RepoCache

A `RepoCache` is a durable local or remote git mirror for a repository.

In local mode this maps naturally to the existing bare repo:

```text
.repos/<owner>/<repo>/.bare/
```

Responsibilities:

- Normalize repo identity.
- Clone/fetch using the existing auth chain.
- Store metadata needed for eviction and debugging.
- Never hold uncommitted agent edits.

Key rule: repo cache is shared infrastructure. It is not the agent's mutable
working area.

### Workspace

A `Workspace` is a durable checkout/worktree created from a repo cache.

Typical examples:

- Read-only analysis workspace for `project_id + base_ref`.
- Edit workspace for `project_id + conversation_id`.
- Task workspace for `project_id + task_id`.

In local mode this maps to a git worktree:

```text
.repos/<owner>/<repo>/worktrees/<user>_<conversation>_<branch>/
```

Responsibilities:

- Provide a stable filesystem root for an agent session.
- Preserve uncommitted edits, generated files, dependency installs, test
  artifacts, and branch state until explicitly cleaned or evicted.
- Track metadata: owner user, project, repo, base ref, branch, dirty state,
  last used time, pinned status, and runtime attachment.

Key rule: persistent agent work belongs to a workspace, not to the runtime.

### SandboxRuntime

A `SandboxRuntime` is the compute environment that executes commands against a
workspace.

Examples:

- Local Docker container with workspace mounted.
- Daytona workspace runtime.
- E2B microVM.
- Local read-only gVisor subprocess path for analysis only.

Responsibilities:

- Start, stop, destroy, and report state.
- Execute commands.
- Stream long-running command output.
- Expose optional backend capabilities such as preview URLs or snapshots.

Key rule: destroying a runtime must not delete a workspace unless the caller
explicitly asks for workspace cleanup.

---

## Layered Architecture

```text
Agent / Tool layer
    |
    v
SandboxService
    - resolve repo identity
    - get/create repo cache
    - get/create workspace
    - attach/resume runtime
    - enforce locks, TTL, eviction
    |
    +--> RepoCacheStore / WorkspaceStore / RuntimeStore
    |
    +--> RepoCacheManager
    |       local implementation wraps existing RepoManager
    |
    +--> WorkspaceManager
    |       local implementation creates git worktrees
    |
    +--> RuntimeProvider port
            DockerProvider / DaytonaProvider / E2BProvider / LocalReadOnlyProvider
```

The repo cache and workspace layers own persistence. The runtime provider owns
execution.

---

## Public Service API

Tools and agents should call `SandboxService`, not backend adapters directly.

```python
# app/modules/sandbox/service.py
from dataclasses import dataclass
from typing import Literal, Optional


WorkspaceMode = Literal["analysis", "edit", "task"]


@dataclass(frozen=True)
class WorkspaceRequest:
    user_id: str
    project_id: str
    repo_name: str                 # "owner/repo"
    repo_url: Optional[str]
    base_ref: str                  # branch or commit
    mode: WorkspaceMode
    conversation_id: Optional[str] = None
    task_id: Optional[str] = None
    branch_name: Optional[str] = None
    create_branch: bool = False


@dataclass(frozen=True)
class RuntimeRequest:
    workspace_id: str
    image: str
    env: dict[str, str]
    writable: bool = True
    network: Literal["none", "limited", "default"] = "limited"
    timeout_s: Optional[int] = None


class SandboxService:
    async def get_or_create_workspace(
        self, request: WorkspaceRequest
    ) -> "Workspace":
        ...

    async def get_or_create_runtime(
        self, request: RuntimeRequest
    ) -> "SandboxRuntime":
        ...

    async def exec(
        self,
        workspace_id: str,
        request: "ExecRequest",
    ) -> "ExecResult":
        ...

    async def hibernate_runtime(self, runtime_id: str) -> None:
        ...

    async def destroy_runtime(self, runtime_id: str) -> None:
        ...

    async def destroy_workspace(self, workspace_id: str) -> None:
        ...
```

Convenience methods can exist for common tool flows:

```python
async def get_or_create_analysis_workspace(user_id, project_id) -> Workspace: ...
async def get_or_create_edit_workspace(user_id, project_id, conversation_id) -> Workspace: ...
async def run_command(workspace_id, command) -> ExecResult: ...
```

---

## Runtime Port

The provider port should be about compute, not repo cloning policy.

```python
# app/modules/sandbox/port.py
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional, Protocol

RuntimeId = str
RuntimeState = Literal["starting", "running", "stopped", "error"]


@dataclass(frozen=True)
class Mount:
    source: str
    target: str
    writable: bool


@dataclass(frozen=True)
class ResourceHints:
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_gb: Optional[int] = None


@dataclass(frozen=True)
class RuntimeSpec:
    image: str
    workdir: str
    mounts: list[Mount]
    env: dict[str, str]
    resources: Optional[ResourceHints] = None
    network: Literal["none", "limited", "default"] = "limited"
    placement: Optional[dict] = None
    labels: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class ExecRequest:
    cmd: list[str]
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None
    timeout_s: Optional[int] = None
    stdin: Optional[bytes] = None
    max_output_bytes: Optional[int] = None


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class ExecChunk:
    stream: Literal["stdout", "stderr"]
    data: bytes


class SandboxRuntime(Protocol):
    id: RuntimeId
    state: RuntimeState

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def exec(self, req: ExecRequest) -> ExecResult: ...
    async def exec_stream(self, req: ExecRequest) -> AsyncIterator[ExecChunk]: ...


class RuntimeProvider(Protocol):
    kind: Literal["local_readonly", "docker", "daytona", "e2b"]
    capabilities: "RuntimeCapabilities"

    async def create(self, spec: RuntimeSpec) -> SandboxRuntime: ...
    async def get(self, id: RuntimeId) -> SandboxRuntime: ...
    async def list(self, labels: Optional[dict[str, str]] = None) -> list["RuntimeSummary"]: ...
    async def destroy(self, id: RuntimeId) -> None: ...
```

### Optional Capabilities

Do not rely on `isinstance(provider, SomeProtocol)` unless the protocol is
runtime-checkable. Prefer explicit optional capability accessors.

```python
@dataclass(frozen=True)
class RuntimeCapabilities:
    snapshot: bool = False
    preview_url: bool = False
    interactive_session: bool = False


class SnapshotCapability(Protocol):
    async def snapshot_runtime(self, runtime_id: RuntimeId, label: str | None = None) -> str: ...
    async def restore_runtime(self, snapshot_id: str) -> SandboxRuntime: ...


class PreviewURLCapability(Protocol):
    async def expose_port(self, runtime_id: RuntimeId, port: int) -> str: ...


class SessionCapability(Protocol):
    async def open_session(self, runtime_id: RuntimeId) -> "Session": ...
```

Provider construction can expose these as attributes:

```python
provider.snapshotter: SnapshotCapability | None
provider.preview_urls: PreviewURLCapability | None
provider.sessions: SessionCapability | None
```

---

## Storage Model

Persist this in Postgres, even if the first implementation only supports local
filesystem paths.

### `sandbox_repo_caches`

One row per durable repo mirror.

Recommended columns:

- `id`
- `repo_name`
- `repo_url`
- `provider`
- `provider_host`
- `local_path`
- `auth_scope_hash`
- `last_fetched_at`
- `last_used_at`
- `size_bytes`
- `state`
- `created_at`
- `updated_at`

Local implementation can mirror existing `.repos/.meta` data, then migrate more
fully later.

### `sandbox_workspaces`

One row per durable worktree/workspace.

Recommended columns:

- `id`
- `repo_cache_id`
- `project_id`
- `user_id`
- `mode`
- `conversation_id`
- `task_id`
- `base_ref`
- `branch_name`
- `local_path`
- `state`
- `dirty`
- `pinned_until`
- `last_used_at`
- `size_bytes`
- `created_at`
- `updated_at`

Suggested uniqueness:

- Analysis workspace: `(user_id, project_id, base_ref, mode)` where
  `mode = "analysis"`.
- Edit workspace: `(user_id, project_id, conversation_id)` where
  `mode = "edit"`.
- Task workspace: `(user_id, project_id, task_id)` where `mode = "task"`.

### `sandbox_runtimes`

One row per runtime attachment.

Recommended columns:

- `id`
- `workspace_id`
- `backend_kind`
- `backend_runtime_id`
- `image`
- `state`
- `last_started_at`
- `last_used_at`
- `expires_at`
- `created_at`
- `updated_at`

Runtime records can be deleted aggressively. Workspace records should survive
runtime destruction.

---

## Workspace Lifecycle

### Analysis Flow

Use this for read-only code search and exploration.

1. Resolve `project_id` to repo details.
2. Ensure repo cache exists using `RepoManager.ensure_bare_repo` or
   `prepare_for_parsing`.
3. Get or create a clean worktree for the requested branch/commit.
4. Attach a read-only runtime when possible.
5. Run whitelisted read-only commands.

This can continue using `bash_command` plus `gvisor_runner.py` initially, but
the tool should treat gVisor fallback as reduced isolation and keep commands
read-only.

### Edit Flow

Use this for feature work, bug fixes, tests, and PR creation.

1. Resolve `project_id` to repo details.
2. Ensure repo cache exists.
3. Create or reuse an edit workspace keyed by `conversation_id`.
4. Create branch `agent/edits-{conversation_id}` from the project base branch.
5. Attach a write-capable runtime with the workspace mounted writable.
6. Run commands and edits inside that runtime.
7. Commit/push/PR from the same workspace.

This should replace the current split where generated changes live in Redis and
then get applied to the worktree. Redis can remain a UI/change-tracking cache,
but the workspace should become the source of truth for actual files.

### Task Flow

Use this for multi-agent or background work where one conversation may spawn
multiple independent implementation attempts.

1. Key workspace by `task_id`.
2. Create branch `agent/task-{task_id}` or a child branch from the edit branch.
3. Keep each task workspace isolated.
4. Merge/cherry-pick/diff back into the conversation edit workspace if accepted.

---

## Eviction Policy

Eviction should be tiered. Compute is cheap to recreate; repo clones are
expensive.

### Runtime Eviction

Default behavior:

- Stop idle runtimes after 15-30 minutes.
- Destroy stopped runtimes after 1-6 hours.
- Never delete the workspace as part of runtime eviction.

Triggers:

- Idle timeout.
- Backend quota pressure.
- Explicit user/project cleanup.
- Failed runtime health checks.

### Workspace Eviction

Default behavior:

- Keep edit workspaces for longer than runtimes, for example 7-30 days.
- Keep dirty or recently active workspaces longer.
- Never evict pinned workspaces.
- Prefer evicting clean analysis workspaces before dirty edit workspaces.

Suggested priority:

1. Clean analysis workspaces older than TTL.
2. Clean task workspaces older than TTL.
3. Clean edit workspaces older than TTL.
4. Dirty unpinned task workspaces under disk pressure.
5. Dirty unpinned edit workspaces only under severe disk pressure.

Before deleting a dirty workspace, consider one of:

- Create a patch artifact.
- Commit to an internal branch.
- Mark it as `eviction_blocked` and alert/admin-log.

### Repo Cache Eviction

Default behavior:

- Keep repo caches longest.
- Evict only when disk pressure remains after runtime and workspace eviction.
- Use LRU with size awareness.
- Do not evict a repo cache with active workspaces.

The existing `RepoManager` already has tiered eviction hooks:

- Worktree threshold: `_WORKTREE_EVICTION_THRESHOLD_PERCENTAGE`
- Repo threshold: `_REPO_EVICTION_TARGET_PERCENTAGE`
- Stale worktree age: `_STALE_WORKTREE_MAX_AGE_DAYS`

The sandbox service should reuse this behavior first, then move metadata into
Postgres when needed.

---

## Concurrency and Locking

Concurrency needs to be explicit because git worktrees and package managers are
not safe under arbitrary parallel mutation.

Use DB uniqueness plus advisory locks:

- Lock repo cache creation by normalized repo identity.
- Lock workspace creation by workspace uniqueness key.
- Lock mutating commands per workspace.
- Allow concurrent read-only commands when no mutating command is running.

Command classes:

- `read`: `rg`, `grep`, `cat`, `ls`, static inspection.
- `write`: file edits, `git checkout`, `git add`, `git commit`, formatters.
- `install`: package manager commands, dependency resolution.
- `test`: usually read-mostly but often writes build artifacts; treat as
  mutating unless mounted with a separate cache/output directory.

For the first implementation, a conservative per-workspace async lock around
all write-capable runtime commands is acceptable.

---

## Security Rules

- Do not pass host environment variables into runtimes by default.
- Inject only scoped credentials that the command actually needs.
- Keep git credentials out of remotes after clone/fetch.
- Do not mount host repo paths broadly; mount only the selected workspace.
- For write-capable work, do not silently fall back to host subprocess.
- Default network to `limited` or `none`; allow broader network only for
  explicit workflows like dependency install.
- Validate all file paths against the workspace root.
- Treat user-provided repo names, branches, and paths as untrusted input.

Important local-mode rule:

`gvisor_runner.py` is acceptable for the existing read-only analysis path. It
should not be used as the write-capable sandbox runtime because it mounts
read-only and has regular-subprocess fallback behavior.

---

## Adapter Responsibilities

| Adapter | Runtime backing | Workspace persistence | Good first use | Notes |
| --- | --- | --- | --- | --- |
| `LocalReadOnlyProvider` | existing gVisor runner | local worktree | read-only analysis | No write workflows; no silent trust in fallback |
| `DockerProvider` | long-lived container | local worktree or Docker volume | first write-capable backend | Best first implementation target |
| `DaytonaProvider` | Daytona workspace/runtime | native or synced workspace | managed sandbox backend | Use native snapshots/previews when useful |
| `E2BProvider` | Firecracker microVM | native or synced workspace | fast ephemeral runtime | Validate persistence semantics carefully |

Local Daytona development notes:

- One-shot setup: `app/src/sandbox/scripts/setup-daytona-local.sh` brings the
  Daytona compose stack up. The script layers
  `scripts/daytona-overrides/docker-compose.override.yaml` on top of the
  upstream compose so the dashboard host port is **3010** by default (port
  3000 is reserved for the potpie frontend); the override also mounts a dex
  config that whitelists `http://localhost:3010` as an OIDC redirect URI.
  Set `DAYTONA_DASHBOARD_PORT=...` before running the script to remap to
  another port.

  After the stack starts the script waits for `/api/health`, mints a dev API key for
  the bundled `dev@daytona.io / password` user, sets a default region if
  needed, writes `app/src/sandbox/.env.daytona.local`, and prints the URL of
  every observability dashboard the compose file already ships:
  - Daytona dashboard: `http://localhost:3010`
  - Sandbox snapshots: `http://localhost:3010/dashboard/snapshots`
  - Active sandboxes: `http://localhost:3010/dashboard/sandboxes`
  - Swagger API: `http://localhost:3010/api`
  - Jaeger traces: `http://localhost:16686`
  - pgAdmin: `http://localhost:5050`
  - Container registry UI: `http://localhost:5100`
  - MinIO console: `http://localhost:9001` (`minioadmin` / `minioadmin`)
  - MailDev: `http://localhost:1080`
- Defaults to the Daytona checkout at `/Users/nandan/Desktop/Dev/daytona`;
  override with `DAYTONA_REPO_PATH=/path/to/daytona`.
- Do not pass `--project-directory` to docker compose. Volume binds in the
  Daytona compose file are relative to the compose file's own directory and
  silently auto-create stub directories at the wrong location otherwise.
- Cleanup: `app/src/sandbox/scripts/teardown-daytona-local.sh` deletes only
  sandboxes labelled `managed-by=potpie`. Pass `--stack` to also bring the
  compose stack down with `-v`.
- The bundled `daytonaio/sandbox:0.5.0-slim` snapshot does **not** ship the
  `git` CLI. The Daytona adapter handles this by using the toolbox `git.clone`
  / `git.create_branch` / `git.checkout_branch` endpoints and verifying
  outcomes by reading `.git/HEAD` directly (the toolbox SDK occasionally
  raises `DaytonaValidationError("...: ")` on operations that actually
  succeeded).
- Tests resolve `proxy.localhost` to 127.0.0.1 in-process (see
  `tests/e2e/conftest.py`) so neither sudo nor the `setup-proxy-dns.sh`
  dnsmasq script is required for E2E runs. Long-lived shell access still
  needs the dnsmasq setup or an `/etc/hosts` entry.
- Adapter env vars (sourced from `.env.daytona.local`):
  `SANDBOX_WORKSPACE_PROVIDER=daytona`, `SANDBOX_RUNTIME_PROVIDER=daytona`,
  `DAYTONA_API_URL`, `DAYTONA_API_KEY`, optional `DAYTONA_SNAPSHOT`,
  `DAYTONA_WORKSPACE_ROOT`.

The adapter must:

- Translate runtime lifecycle and exec calls.
- Map backend errors into typed sandbox errors.
- Enforce its declared mount/network behavior.
- Report capabilities truthfully.

The adapter must not:

- Decide which user/repo/workspace should be used.
- Own the clone-on-create policy.
- Reach into Postgres workspace mappings directly.

---

## Typed Errors

Every adapter should map backend-specific failures into a small error set.

```python
class SandboxError(Exception): ...
class SandboxNotFound(SandboxError): ...
class SandboxUnauthorized(SandboxError): ...
class SandboxTimeout(SandboxError): ...
class SandboxConflict(SandboxError): ...
class SandboxUnavailable(SandboxError): ...
class SandboxResourceLimit(SandboxError): ...
class SandboxCommandRejected(SandboxError): ...
```

Service-level repo/workspace errors should be separate:

```python
class WorkspaceError(Exception): ...
class WorkspaceNotFound(WorkspaceError): ...
class WorkspaceLocked(WorkspaceError): ...
class WorkspaceDirty(WorkspaceError): ...
class RepoCacheUnavailable(WorkspaceError): ...
class RepoAuthFailed(WorkspaceError): ...
```

---

## First Implementation Plan

### Phase 1: Model the service without changing behavior

- Add `app/src/sandbox/` with service, models, and runtime port.
- Add a local `WorkspaceManager` that wraps the existing `RepoManager`.
- Store workspace/runtime metadata in Postgres, but allow local path lookup from
  existing repo manager metadata during migration.
- Keep `bash_command` on the current read-only path.

### Phase 2: Route edit workspaces through SandboxService

- Make `checkout_worktree_branch` call `SandboxService.get_or_create_workspace`.
- Keep the branch naming convention: `agent/edits-{conversation_id}`.
- Ensure `apply_changes`, `git_commit`, and PR creation resolve the same
  workspace record.
- Add per-workspace locks around branch creation and mutating file/git
  operations.

### Phase 3: Add DockerProvider for write-capable exec

- Mount the workspace writable at `/work`.
- Disable silent host fallback.
- Add resource hints and timeout handling.
- Stream command output for long-running commands.
- Validate persistence across runtime stop/destroy/recreate.
- Keep a local Docker Compose smoke test that mounts the repo at the same
  absolute host path and forwards `/var/run/docker.sock`, so nested Docker bind
  mounts point at host-visible worktree paths.

### Phase 4: Move command tools onto SandboxService

- Split tools by intent:
  - read-only command tool can keep stricter whitelist.
  - write-capable command tool is available only in edit/task workspaces.
- Run tests and package commands through DockerProvider, not gVisor.
- Keep output limits and streaming to avoid API process OOMs.

### Phase 5: Add managed backend

- Add Daytona after Docker proves the port.
- Map Daytona native workspace/snapshot/preview features through capabilities.
- Only add E2B if cold start or isolation requirements justify it.

---

## Open Questions

- Should edit workspace persistence be keyed by `conversation_id`, by an
  explicit "agent task/session id", or both?
- When a dirty workspace is old enough to evict, should Potpie auto-commit to
  an internal branch, save a patch artifact, or block eviction?
- Should dependency caches be per-user, per-repo, or per-workspace?
- How much network access should test/install commands get by default?
- Should local Docker workspaces mount host worktrees directly, or should they
  copy/sync into a Docker volume for stronger host isolation?
