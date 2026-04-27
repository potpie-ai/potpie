# Sandbox — Core Setup

## Context

Potpie agents need a place to do real work on user repositories: clone the repo, run commands, edit files, run tests, and resume that environment in a later session. Today, command execution happens through `app/modules/utils/gvisor_runner.py` against local worktrees, restricted to a whitelist of read-only commands. That is fine for analysis tools (`grep`, `find`, `cat`) but does not cover write-capable agent workflows that need persistence across sessions.

We want one **sandbox port** with multiple **backend adapters** (local, Docker, Daytona, E2B) so the application layer never depends on a specific backend. This doc defines the port, the application layer that sits on top of it, and what each adapter is responsible for.

---

## Goals

- Agent does work on a user repo inside an isolated environment.
- Environment **persists** across agent sessions — same user + same repo resumes the same sandbox state.
- Application code is backend-agnostic. Switching from local Docker to Daytona is a config change.
- Backend-specific features (snapshots, preview URLs) are exposed through **capability interfaces**, not forced into the core port.

## Non-goals

- A general-purpose dev environment for humans (use VS Code remote, Coder, devpod for that).
- Unifying networking, regions, or resource quotas across backends — too divergent.
- Replacing `gvisor_runner.py` for the read-only analysis path. That stays as the `local` adapter's fast path.

---

## Layered architecture

```
┌──────────────────────────────────────────────┐
│  Agent / Tool layer                          │  langchain tools, pydantic-ai agents
└─────────────────────┬────────────────────────┘
                      │
┌─────────────────────▼────────────────────────┐
│  SandboxService (application)                │  user→sandbox mapping,
│  - getOrCreateForRepo(user, repo)            │  TTL/hibernation policy,
│  - hibernate / resume / destroy              │  clone-on-create workflow
└─────────────────────┬────────────────────────┘
                      │
┌─────────────────────▼────────────────────────┐
│  SandboxProvider (port)                      │  backend-agnostic interface
└─────────────────────┬────────────────────────┘
        ┌─────────────┼──────────────┬──────────────┐
        ▼             ▼              ▼              ▼
   LocalProvider  DockerProvider  DaytonaProvider  E2BProvider
   (gvisor +      (long-lived     (managed or      (managed or
    worktree)      container)      self-hosted)     self-hosted)
```

The **port** is dumb: lifecycle, exec, files. The **service** holds policy: which sandbox belongs to which user/repo, when to hibernate, how to resume.

---

## The port

### Core interfaces

```python
# app/modules/sandbox/port.py
from typing import Protocol, AsyncIterator, Literal, Optional
from dataclasses import dataclass

SandboxId = str
SandboxState = Literal["starting", "running", "stopped", "error"]


@dataclass(frozen=True)
class SandboxSpec:
    image: str                          # OCI image or backend-specific identifier
    env: dict[str, str]
    workdir: str = "/work"
    resources: Optional["ResourceHints"] = None
    placement: Optional[dict] = None    # opaque; backend-specific (region, runner)


@dataclass(frozen=True)
class ResourceHints:
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None
    disk_gb: Optional[int] = None


@dataclass(frozen=True)
class ExecRequest:
    cmd: list[str]
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None
    timeout_s: Optional[int] = None
    stdin: Optional[bytes] = None


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class ExecChunk:
    stream: Literal["stdout", "stderr"]
    data: bytes


class FileSystem(Protocol):
    async def read(self, path: str) -> bytes: ...
    async def write(self, path: str, data: bytes) -> None: ...
    async def list(self, path: str) -> list[str]: ...
    async def upload(self, local_path: str, remote_path: str) -> None: ...
    async def download(self, remote_path: str, local_path: str) -> None: ...


class Sandbox(Protocol):
    id: SandboxId
    state: SandboxState
    fs: FileSystem

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def exec(self, req: ExecRequest) -> ExecResult: ...
    async def exec_stream(self, req: ExecRequest) -> AsyncIterator[ExecChunk]: ...


class SandboxProvider(Protocol):
    kind: Literal["local", "docker", "daytona", "e2b"]
    capabilities: "Capabilities"

    async def create(self, spec: SandboxSpec) -> Sandbox: ...
    async def get(self, id: SandboxId) -> Sandbox: ...
    async def list(self, filter: Optional[dict] = None) -> list["SandboxSummary"]: ...
    async def destroy(self, id: SandboxId) -> None: ...
```

### Capability interfaces (optional)

Don't pollute the core with methods only some backends support. Expose these as separate protocols and let the service check `provider.capabilities` before using them.

```python
@dataclass(frozen=True)
class Capabilities:
    snapshot: bool = False
    preview_url: bool = False
    interactive_session: bool = False


class SnapshotCapability(Protocol):
    async def snapshot(self, id: SandboxId, label: Optional[str] = None) -> str: ...
    async def restore(self, snapshot_id: str) -> Sandbox: ...


class PreviewURLCapability(Protocol):
    async def expose_port(self, id: SandboxId, port: int) -> str: ...


class SessionCapability(Protocol):
    """Long-lived PTY for shells the agent keeps open across calls."""
    async def open_session(self, id: SandboxId) -> "Session": ...
```

### Typed errors

Every adapter maps backend errors into this small set. Without this, the service layer ends up with `if "not found" in str(err)` per backend.

```python
class SandboxError(Exception): ...
class SandboxNotFound(SandboxError): ...
class SandboxUnauthorized(SandboxError): ...
class SandboxTimeout(SandboxError): ...
class SandboxConflict(SandboxError): ...
class SandboxUnavailable(SandboxError): ...   # backend down / quota / transient
```

---

## The application layer

`SandboxService` is what tools and agents actually call. It owns:

- The mapping from `(user_id, repo_url)` → `SandboxId` (persisted in Postgres alongside the rest of potpie's state).
- The clone-on-create workflow.
- Hibernation policy (when to `stop`, when to snapshot, when to destroy).
- Resume semantics (cold start vs. snapshot restore).

```python
# app/modules/sandbox/service.py
class SandboxService:
    def __init__(self, provider: SandboxProvider, store: SandboxStore):
        self._provider = provider
        self._store = store

    async def get_or_create_for_repo(
        self, user_id: str, repo_url: str, *, image: str
    ) -> Sandbox:
        record = await self._store.find(user_id=user_id, repo_url=repo_url)

        if record is not None:
            sb = await self._provider.get(record.sandbox_id)
            if sb.state == "stopped":
                await sb.start()
            return sb

        sb = await self._provider.create(
            SandboxSpec(image=image, env={"REPO_URL": repo_url})
        )
        await sb.exec(ExecRequest(cmd=["git", "clone", repo_url, "/work"]))
        await self._store.save(
            user_id=user_id, repo_url=repo_url, sandbox_id=sb.id
        )
        return sb

    async def hibernate(self, sandbox_id: SandboxId) -> None:
        sb = await self._provider.get(sandbox_id)
        if isinstance(self._provider, SnapshotCapability):
            await self._provider.snapshot(sandbox_id, label="hibernate")
        await sb.stop()
```

`SandboxStore` is a thin Postgres-backed table — `(user_id, repo_url, sandbox_id, last_used_at, backend_kind)`. Adapters never touch it.

---

## Adapter responsibilities

Each adapter is responsible for translating the port to its backend, mapping errors into the typed set, and declaring its capabilities truthfully.

| Adapter | Backing | Persistence | Snapshots | Preview URLs | Status |
|---------|---------|-------------|-----------|--------------|--------|
| `LocalProvider` | gVisor + worktree | filesystem | no | no | wraps existing `gvisor_runner.py` for the analysis path; new code path for write-capable exec |
| `DockerProvider` | long-lived container | docker volume | via `docker commit` | port publish | first cross-machine target; covers self-host without external service |
| `DaytonaProvider` | Daytona API | native | yes | yes | for managed or self-hosted Daytona; uses `daytona` Python SDK |
| `E2BProvider` | E2B Firecracker microVMs | native | yes | yes | optional; lighter than Daytona, faster cold start |

Local-mode caveats:
- `LocalProvider` should not pretend to support snapshots. Set `capabilities.snapshot = False`.
- The existing whitelist in `bash_command_tool.py` belongs in the **tool**, not the adapter. The adapter executes whatever it's told; the tool decides what's safe.

---

## What we deliberately don't abstract

- **Auth.** Each provider takes its own config object at construction. No unified credential interface.
- **Regions / runners.** Daytona-specific. Pass through `SandboxSpec.placement` as opaque dict.
- **Resource limits.** Hints only. Adapters that can't enforce them ignore them.
- **Network policy.** Backend-level config, not a runtime API.
- **Snapshots.** Not in the core port. Capability interface only.

---

## Three rules that prevent pain

1. **Async everything, even local.** The instant any adapter is remote, sync APIs become a lie. `gvisor_runner.run_command_isolated` is sync today; the local adapter wraps it with `asyncio.to_thread` (consistent with the direction in `docs/async-migration-plan.md`).
2. **Stream long-running exec.** `exec` returns buffered bytes — fine for `ls`. Anything that produces large output (`pytest`, `npm install`, `git clone` of a big repo) goes through `exec_stream`. Buffering a 200MB test log will OOM the API process.
3. **Typed errors at the adapter boundary.** Backend exceptions never leak past the adapter. Service layer only sees `SandboxError` subclasses.

---

## Directory layout

```
app/modules/sandbox/
    __init__.py
    port.py                  # protocols, dataclasses, errors
    service.py               # SandboxService, SandboxStore protocol
    store_postgres.py        # concrete store
    adapters/
        local.py             # wraps gvisor_runner
        docker.py
        daytona.py
        e2b.py
```

---

## Migration from today

1. Land `port.py` and `service.py` with no behavior change. Add `LocalProvider` wrapping `gvisor_runner.py`.
2. Move `bash_command_tool.py` to call `SandboxService` instead of `run_command_isolated` directly. Whitelist enforcement stays in the tool.
3. Add `DockerProvider` behind a feature flag. Validate clone + exec + persistence across restarts.
4. Add `DaytonaProvider` once Docker proves the abstraction holds. If anything in the port has to change to fit Daytona, it was the wrong abstraction — fix the port, not the adapter.
5. `E2BProvider` only if Daytona's snapshot/cold-start latency becomes a real bottleneck.

---

## Open questions

- **Where does repo state live during hibernation** — sandbox-internal disk, external volume, or rehydrated from git on resume? Affects snapshot size and resume latency. Probably differs per backend (Daytona snapshot vs. Docker volume vs. fresh clone).
- **Cleanup policy** — TTL based on `last_used_at`, or tied to user session? Whichever, the policy belongs in `SandboxService`, not the adapter.
- **Concurrency on a single sandbox** — can two agent calls `exec` on the same sandbox in parallel? Most backends say yes, but command-level locking may be needed at the service layer for repo-mutating commands (`git`, file writes).
