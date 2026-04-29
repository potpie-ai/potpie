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

The hexagonal sandbox module now lives at `app/src/sandbox/sandbox/` (domain
ports, application service, API client, outbound adapters). The legacy
`app/modules/repo_manager/repo_manager.py` still owns the `.repos/`
filesystem layout, bare-repo cloning, and tiered eviction; the bridge at
`app/modules/sandbox_repos/provider.py` adapts it to the sandbox
`WorkspaceProvider` port for the production wiring. See **Current State** below
for what is in place and **Known Gaps** for what still needs to be moved out
of the legacy module.

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

## Current State (Implemented)

The hexagonal layout described above is in place. References point at code
that is shipped today.

**Domain** (`app/src/sandbox/sandbox/domain/`)
- `models.py` — `Workspace`, `WorkspaceRequest`, `RepoIdentity`, `RepoCache`
  (declared but not yet wired through), `Runtime`, `RuntimeSpec`,
  `ExecRequest`, `ExecResult`, `Mount`, `WorkspaceMode` (`ANALYSIS`/`EDIT`/
  `TASK`).
- `errors.py` — `WorkspaceError` family (`WorkspaceNotFound`,
  `RepoCacheUnavailable`, `RepoAuthFailed`, `InvalidWorkspacePath`, …) and
  `RuntimeErrorBase` family (`RuntimeUnavailable`, `RuntimeTimeout`, …).
- `ports/workspaces.py`, `ports/runtimes.py`, `ports/stores.py`,
  `ports/locks.py` — the four ports.

**Application** (`app/src/sandbox/sandbox/application/services/sandbox_service.py`)
- `SandboxService` owns `get_or_create_workspace`,
  `get_or_create_runtime`, `exec`, `hibernate_runtime`,
  `destroy_runtime`, `destroy_workspace`. Persists workspaces and runtimes
  via the store, takes per-workspace locks for mutating commands.

**API** (`app/src/sandbox/sandbox/api/client.py`)
- `SandboxClient` is the public façade. Helpers: `read_file`,
  `write_file`, `list_dir`, `search`, `status`, `diff`, `commit`, `push`.
  Local-fs fast paths when `handle.local_path is not None`; exec-based
  fallbacks otherwise.

**Bootstrap** (`app/src/sandbox/sandbox/bootstrap/`)
- `settings.py` reads `SANDBOX_WORKSPACE_PROVIDER` /
  `SANDBOX_RUNTIME_PROVIDER` / `SANDBOX_REPOS_BASE_PATH` /
  `SANDBOX_METADATA_PATH`.
- `container.py` — `build_sandbox_container` selects providers and store.

**Adapters**
- `adapters/outbound/local/git_workspace.py` — `LocalGitWorkspaceProvider`.
  Bare-repo + worktree creation under `.repos/`, with
  `<user>_<scope>_<branch>` worktree paths to isolate per-conversation
  edits. Currently **not used in production** — the bridge below overrides
  the wiring.
- `adapters/outbound/local/subprocess_runtime.py` —
  `LocalSubprocessRuntimeProvider` for local exec.
- `adapters/outbound/daytona/provider.py` — `DaytonaWorkspaceProvider` and
  `DaytonaRuntimeProvider`. One Daytona sandbox per `(user, project)`,
  branch-named worktrees inside it.
- `adapters/outbound/docker/runtime.py` — `DockerRuntimeProvider`.
- `adapters/outbound/file/json_store.py` — `JsonSandboxStore` (durable
  metadata; default for local mode).
- `adapters/outbound/memory/store.py`, `memory/locks.py` — in-memory
  fallbacks.

**Bridge to legacy `RepoManager`**
- `app/modules/sandbox_repos/provider.py` — `RepoManagerWorkspaceProvider`.
  In production wiring (`app/modules/intelligence/tools/sandbox/client.py`)
  this is swapped in for `LocalGitWorkspaceProvider` so that parsing,
  agent tooling, and eviction all flow through the legacy
  `app.modules.repo_manager.RepoManager`. Keeps an in-memory cache only —
  no rows in `JsonSandboxStore`.

**Agent tool surface** (`app/modules/intelligence/tools/sandbox/`)
- `client.py` — process-wide `SandboxClient` accessor (`get_sandbox_client`)
  and `resolve_workspace(...)` helper.
- `context.py` — contextvars (`user_id`, `conversation_id`, `branch`,
  `auth_token`) the agent harness sets at run start.
- `tools.py` exports `create_sandbox_tools()` returning four tools:
  `sandbox_text_editor`, `sandbox_shell`, `sandbox_search`, `sandbox_git`.
- `tool_functions.py` — each tool calls `_resolve(project_id, mode=...)`
  to get a `WorkspaceHandle` per call.

---

## Known Gaps (post-revamp)

Findings from the architectural audit; tracked by the **Implementation
Roadmap** below.

1. **Two parallel local providers, one is dead code.**
   `LocalGitWorkspaceProvider` is the canonical hexagonal local adapter,
   but the production wiring substitutes `RepoManagerWorkspaceProvider`
   so `LocalGitWorkspaceProvider` is never called outside tests. The two
   have drifted (different worktree path layouts, different metadata
   schemas).

2. **Dual unsynchronized persistence.**
   `RepoManager` writes
   `.repos/.meta/<owner>/<repo>/branch__commit.json`; `JsonSandboxStore`
   writes `Workspace` records. They never reconcile. Eviction in one
   leaves stale rows in the other.

3. **Eviction lives outside the sandbox application layer.**
   Volume tracking and tiered eviction (worktrees first at 80%, full
   repos at 90%) are inside `RepoManager`. The sandbox module has no
   `EvictionPolicy` port and no eviction logic of its own.

4. **Parsing bypasses the sandbox abstraction.**
   `parsing_service.py:298` calls `parse_helper.clone_or_copy_repository`
   which goes directly to `RepoManager`. There is a feature-flagged
   `SandboxClient` path (`SANDBOX_PARSING_ENABLED`), but it is not the
   default and does not persist a `Workspace` past the call. After
   `update_project_status(..., READY)` at `parsing_service.py:575` no
   workspace record exists in the sandbox store.

5. **No first-class `RepoCache` entity.** `RepoCache` is declared in
   `models.py:115` but `Workspace.repo_cache_id` is always `None`. There
   is no `RepoCacheProvider` port; the bare-repo concept is implicit
   inside each adapter.

6. **`WorkspaceMode` overloads four concerns:** read-only vs writable,
   branch creation, sharing, and keying. Branch-creation logic is
   duplicated across `LocalGitWorkspaceProvider._create_workspace_sync`
   and `daytona/provider.py:_ensure_worktree`. Capabilities should be
   explicit on the workspace, derived from mode at construction.

7. **Daytona adapter leaks SDK types.** `Any`-typed sandbox handles
   (`_client_factory`, `_sandboxes`) flow through the application layer.
   `DaytonaRuntimeProvider.exec` reaches into `sandbox.process.exec`
   directly. Worktree paths use only `branch_name` (vs. local's
   `<user>_<scope>_<branch>`), which weakens isolation guarantees.

8. **No git-platform port.** PR creation is not represented in the
   sandbox layer. `commit` and `push` exist on `SandboxClient`, but
   creating a PR has to bypass the sandbox abstraction.

9. **Tool surface uses contextvars instead of explicit dependencies.**
   `tool_functions._resolve` reads `user_id` from a contextvar and looks
   up `repo_name` via DB. The agent harness sets the contextvar at
   `pydantic_agent.py:613-621`. A toolset factory should take
   `(client, run_context)` explicitly.

10. **No eviction port / no per-conversation cleanup hook.**
    `SandboxStore` lacks `list_workspaces_by_repo`,
    `find_workspaces_for_eviction`, etc. EDIT-mode worktrees are only
    reaped by `RepoManager`'s age-based eviction; no
    `release_session(conversation_id)` exists.

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

## Implementation Roadmap

The original Phase 1-5 plan is **complete** — the sandbox module, the
runtime port, the Daytona adapter, the agent tool surface, and the
Postgres-ready store are all shipped. The post-revamp phases (P1-P8)
each address a numbered gap from the audit. Status as of this
session:

### P1 — Unify the local provider — DONE (cutover is opt-in)

Closes gaps 1, 2, 3.

* `EvictionPolicy` port at `domain/ports/eviction.py` with
  `EvictionResult` value type; `NoOpEvictionPolicy` default in
  `adapters/outbound/memory/eviction.py`.
* `LocalGitWorkspaceProvider` accepts `eviction=` kwarg, calls
  `evict_if_needed` on cache miss.
* Production cutover is **gated** by env var:
  ``SANDBOX_USE_CANONICAL_LOCAL=true`` switches
  `intelligence/tools/sandbox/client.py` from the bridge to
  `LocalGitWorkspaceProvider`. Default false to preserve operator
  control over the migration window. `RepoManagerWorkspaceProvider` is
  marked deprecated; it stays in tree until all environments have
  flipped the flag.
* **Operator note:** existing on-disk worktrees from the bridge
  (layout: `<branch>` for shared, `<user>_<unique>_<branch>` for
  conversation-scoped) are NOT migrated. The canonical adapter creates
  fresh worktrees in the `<user>_<scope>_<branch>` layout. Bridge
  worktrees with uncommitted state should be committed/pushed before
  flipping the flag, or accept that they become inaccessible to
  canonical-mode tools.
* Volume-aware policy (`VolumeBasedEvictionPolicy`) replacing the
  `NoOp` default is a P1 follow-up; today eviction in the legacy path
  flows through `RepoManager`'s thresholds and the canonical path runs
  unbounded until the policy is wired.

### P2 — Promote `RepoCache` to first-class — DONE

Closes gap 5.

* `RepoCacheRequest` and `RepoCache` (with stable `key`) in
  `domain/models.py`.
* `RepoCacheProvider` port at `domain/ports/repos.py`.
* `RepoCacheStore` mixin on `SandboxStore`; both `InMemorySandboxStore`
  and `JsonSandboxStore` implement it (rows survive restart).
* `LocalRepoCacheProvider` at
  `adapters/outbound/local/repo_cache.py` owns bare-repo creation;
  `LocalGitWorkspaceProvider` depends on the cache port and sets
  `Workspace.repo_cache_id` on every workspace it builds.
* `SandboxService.ensure_repo_cache(request)` keys by repo identity,
  takes a per-key lock, persists the row.

### P3 — Capabilities split and `acquire_session` API — DONE

Closes gaps 6 and 10.

* `Capabilities(writable, isolated, persistent)` value object;
  `Capabilities.from_mode` is the single source of truth.
* `Workspace.capabilities` populated by every adapter (local, bridge,
  Daytona). Round-trips through `JsonSandboxStore`.
* `SandboxService.acquire_session(request)` orchestrates ensure-cache
  + workspace creation atomically.
* `SandboxService.release_session(workspace_id, *, destroy_runtime)`
  hibernates the runtime by default; workspace survives.
* `SandboxClient.acquire_session` / `release_session` symmetric public
  API.

### P4 — Daytona hardening — partial

Addresses gap 7 (correctness portion).

**Done:**
* `_validate_ref` in `daytona/provider.py` rejects newlines / `..` in
  base_ref and branch_name before they hit shell-style exec calls
  (parity with the local adapter).
* Worktree path now `<user>_<scope>_<branch>` so two conversations on
  the same branch get distinct worktrees (no silent collision).

**Deferred (code-quality follow-up):** The Daytona SDK still types as
`Any` on `_client_factory`/`_sandboxes`; the full `DaytonaApi` Protocol
abstraction is a 500+ line refactor that doesn't fix any correctness
issue. Track as a P4.5 cleanup.

### P5 — Provision-on-parse — DONE

Closes gap 4.

* `SandboxClient.ensure_repo_cache(...)` thin wrapper over the service.
* `provision_repo_cache` helper at
  `app/modules/intelligence/tools/sandbox/client.py`.
* `parsing_service.py` calls `_provision_repo_cache_safe` from BOTH
  READY transitions (eager-return short-circuit and the normal
  post-`analyze_directory` exit). Failures are logged and swallowed —
  cache provisioning is an optimization, not a parsing prerequisite.

### P6 — Tool surface refactor — DONE

Closes gap 9.

* `create_sandbox_tools(client=..., handle=...)` — explicit-handle
  factory mode. Tools dispatch through pre-bound `(client, handle)`
  closures with input schemas that omit `project_id`. Capability
  gating drops write tools when the handle is read-only
  (`enforce_capabilities=True` default).
* The legacy zero-arg `create_sandbox_tools()` form keeps working —
  required for back-compat with the existing harness wiring at
  `multi_agent/agent_factory.py` and `pydantic_agent.py`.
* Helper functions extracted in `tool_functions.py`:
  `_exec_text_editor`, `_exec_shell`, `_exec_search`, `_exec_git`,
  `_exec_pull_request`. Both factory modes funnel through the same
  helpers.
* `WorkspaceHandle.capabilities` carries the gating signal.
* Contextvar machinery in `tools/sandbox/context.py` is **kept** for
  the legacy form — full removal happens once harness callers migrate
  to the explicit form (P6.5 cleanup).

### P7 — `GitPlatformProvider` and PR tool — DONE

Closes gap 8.

* `GitPlatformProvider` port at `domain/ports/git_platform.py`;
  `PullRequestRequest` / `PullRequest` value objects in
  `domain/models.py`; `PullRequestFailed` /
  `GitPlatformNotConfigured` errors.
* `GitHubGitPlatformProvider` bridge adapter at
  `app/modules/sandbox_repos/git_platform.py` wraps the existing
  `code_provider.github.GitHubProvider` so auth chain stays put.
* `SandboxService.create_pull_request(request)` enforces "platform
  configured" precondition; `SandboxClient.create_pull_request(handle,
  ...)` enforces "writable workspace" precondition.
* `sandbox_pr` agent tool — included in the explicit toolset only when
  the harness passes ``pr_repo_name=...``; capability-gated on
  writable handles. Push the branch via `sandbox_git push` first; the
  PR tool is the platform-side step only.
* **Production wiring of the platform provider is a follow-up.** The
  per-call user resolution (auth tokens scoped to the calling user)
  doesn't fit the process-wide `SandboxClient` cleanly; needs a small
  request-scoped factory before it can run in prod.

### P8 — Postgres store and multi-worker locks — DEFERRED

Closes the implicit single-node assumption in `JsonSandboxStore` and
`InMemoryLockManager`. The schema sketch is already documented above
(`sandbox_repo_caches`, `sandbox_workspaces`, `sandbox_runtimes`).
Implementation requires a Postgres connection and migration tooling
that aren't in this session's scope. Adapter signatures will mirror
the existing in-memory and JSON ones, so the swap is a bootstrap-only
change.

**Track as a follow-up.** The current `JsonSandboxStore` is suitable
for single-node deployments; multi-worker / multi-host setups should
not flip to canonical-local until the Postgres adapter ships, since
the JSON store's flush model assumes a single writer.

### Outstanding follow-ups (small)

* P1 follow-up — `VolumeBasedEvictionPolicy` reading from
  `SandboxStore` so canonical-local has bounded disk use.
* P4 follow-up — `DaytonaApi` Protocol port to remove `Any` typing.
* P6 follow-up — migrate harness callers
  (`multi_agent/agent_factory.py`, `pydantic_agent.py`) to the
  explicit-handle form, then delete the contextvar plumbing.
* P7 follow-up — request-scoped `GitPlatformProvider` factory so the
  per-user auth chain works in production.
* P8 — Postgres adapters once the DB story lands.

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
