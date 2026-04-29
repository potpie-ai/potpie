from __future__ import annotations

import threading
import time

import pytest

from sandbox.adapters.outbound.daytona.provider import (
    DaytonaRuntimeProvider,
    DaytonaWorkspaceProvider,
)
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.models import (
    ExecRequest,
    RepoIdentity,
    RuntimeRequest,
    WorkspaceMode,
    WorkspaceRequest,
)


class FakeResponse:
    def __init__(self, exit_code: int = 0, result: str = "") -> None:
        self.exit_code = exit_code
        self.result = result


class FakeGit:
    def __init__(self) -> None:
        self.clones: list[dict[str, object]] = []

    def clone(self, **kwargs: object) -> None:
        self.clones.append(kwargs)


class FakeProcess:
    """Minimal process emulator.

    Tracks every command and lets tests script per-command exit codes. The
    default returns 0 with `result="ok"`. `pwd` is special-cased so the
    runtime test can assert which workdir was used.

    The fake also synthesises filesystem state for the probes the provider
    relies on:

    * ``git clone --bare ... <path>`` → marks ``<path>/HEAD`` as existing.
    * ``git ... worktree add ... -- <path> ...`` → marks ``<path>/.git``.
    * ``test -d|-e|-f <path>`` → exit 0 iff ``<path>`` was marked above.
    """

    def __init__(self) -> None:
        self.commands: list[dict[str, object]] = []
        self.fake_paths: set[str] = set()

    def exec(self, command: str, **kwargs: object) -> FakeResponse:
        self.commands.append({"command": command, **kwargs})
        if command == "pwd":
            return FakeResponse(result=str(kwargs.get("cwd", "")))
        if command.startswith(("test -d ", "test -e ", "test -f ")):
            target = command.split(" ", 2)[2].strip("'\"")
            return FakeResponse(
                exit_code=0 if target in self.fake_paths else 1,
                result="",
            )
        if command.startswith("git clone "):
            parts = command.split()
            path = parts[-1].strip("'\"")
            if "--bare" in parts:
                self.fake_paths.add(f"{path}/HEAD")
            else:
                self.fake_paths.add(f"{path}/.git")
        elif "worktree add" in command:
            parts = command.split()
            try:
                tree_idx = parts.index("--") + 1
                tree_path = parts[tree_idx].strip("'\"")
                self.fake_paths.add(f"{tree_path}/.git")
            except (ValueError, IndexError):
                pass
        return FakeResponse(result="ok")

    def mark_bare(self, path: str) -> None:
        """Test helper: pretend a bare clone exists at `path`."""
        self.fake_paths.add(f"{path}/HEAD")


class FakeSandbox:
    def __init__(
        self,
        sandbox_id: str,
        *,
        labels: dict[str, str] | None = None,
        state: str = "started",
        created_at: str = "",
    ) -> None:
        self.id = sandbox_id
        self.git = FakeGit()
        self.process = FakeProcess()
        self.labels = labels or {}
        self.state = state
        self.created_at = created_at
        self.started = False
        self.stopped = False
        self.deleted = False

    def start(self) -> None:
        self.started = True
        self.state = "started"

    def stop(self) -> None:
        self.stopped = True

    def delete(self) -> None:
        self.deleted = True


class _FakeSnapshot:
    """Lightweight stand-in for daytona.common.snapshot.Snapshot."""

    def __init__(self, state: str) -> None:
        self.state = state


class FakeSnapshotService:
    """Tracks snapshot.get/create calls and lets tests script lookup misses.

    Default behaviour: ``get`` returns a present snapshot in the ``active``
    state so tests that don't care about provisioning don't have to wire up
    dockerfiles. Set ``existing=False`` to simulate a fresh Daytona where the
    snapshot has to be built; ``create`` then flips ``existing`` so the next
    ``get`` hits with state ``active``.

    Pass a list of states via ``states`` to script a sequence — useful for
    the ``wait_for_active`` path (e.g. ``["pending", "building", "active"]``).
    """

    def __init__(
        self,
        existing: bool = True,
        states: list[str] | None = None,
    ) -> None:
        self.existing = existing
        self._states: list[str] = list(states) if states else []
        self._default_state = "active"
        self.get_calls: list[str] = []
        self.create_calls: list[object] = []

    def get(self, name: str) -> _FakeSnapshot:
        self.get_calls.append(name)
        if not self.existing:
            raise RuntimeError(f"snapshot {name} not found")
        if self._states:
            state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        else:
            state = self._default_state
        return _FakeSnapshot(state)

    def create(
        self,
        params: object,
        *,
        on_logs: object | None = None,
        timeout: float = 0,
    ) -> _FakeSnapshot:
        self.create_calls.append(params)
        self.existing = True
        return _FakeSnapshot("active")


class FakeDaytonaClient:
    """Mints sequential sandbox ids and records every `create()` call.

    ``list(labels=...)`` mirrors the Daytona SDK so we can exercise the
    cross-restart sandbox-recovery path. ``register`` lets tests pre-seed
    sandboxes that look like they survived from a previous worker session.
    """

    def __init__(self, snapshot_exists: bool = True) -> None:
        self._sandboxes: dict[str, FakeSandbox] = {}
        self.create_calls = 0
        self.list_calls: list[dict[str, str] | None] = []
        self.snapshot = FakeSnapshotService(existing=snapshot_exists)

    def create(self, params: object | None = None) -> FakeSandbox:
        self.create_calls += 1
        sandbox_id = f"sbx_{self.create_calls}"
        labels = dict(getattr(params, "labels", None) or {}) if params else {}
        sandbox = FakeSandbox(sandbox_id, labels=labels, state="started")
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    def get(self, sandbox_id: str) -> FakeSandbox:
        return self._sandboxes[sandbox_id]

    def list(self, labels: dict[str, str] | None = None) -> list[FakeSandbox]:
        self.list_calls.append(labels)
        if not labels:
            return list(self._sandboxes.values())
        return [
            s for s in self._sandboxes.values()
            if all(s.labels.get(k) == v for k, v in labels.items())
        ]

    def register(self, sandbox: FakeSandbox) -> None:
        """Pre-seed a sandbox so tests can simulate a Daytona-side survivor."""
        self._sandboxes[sandbox.id] = sandbox


@pytest.mark.asyncio
async def test_daytona_workspace_bare_clones_and_creates_worktree() -> None:
    client = FakeDaytonaClient()
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    runtime_provider = DaytonaRuntimeProvider(workspace_provider)
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )

    bare_dir = "/home/daytona/workspace/owner_private/.bare"
    worktree_dir = "/home/daytona/workspace/owner_private/worktrees/agent_edits-c1"

    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(
                repo_name="owner/private",
                repo_url="https://github.com/owner/private.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
            auth_token="secret-token",
        )
    )

    sandbox = client._sandboxes["sbx_1"]
    assert workspace.location.remote_path == worktree_dir
    assert workspace.location.backend_workspace_id == "sbx_1"

    # Bare-clone via shell; auth embedded in URL.
    clone_cmds = [
        str(c["command"]) for c in sandbox.process.commands
        if "git clone --bare" in str(c.get("command"))
    ]
    assert len(clone_cmds) == 1
    assert "x-access-token:secret-token@github.com" in clone_cmds[0]
    assert bare_dir in clone_cmds[0]
    # Provider must NOT call the SDK's git.clone — it doesn't support --bare.
    assert sandbox.git.clones == []

    worktree_cmds = [
        str(c["command"]) for c in sandbox.process.commands
        if "worktree add" in str(c.get("command"))
    ]
    # Uses lowercase `-b` (fresh branch from base_ref) — never `-B` (which
    # would reset the branch and discard prior agent commits on re-runs).
    assert any(
        bare_dir in cmd and worktree_dir in cmd and "-b" in cmd.split()
        and "agent/edits-c1" in cmd
        for cmd in worktree_cmds
    )

    runtime = await service.get_or_create_runtime(RuntimeRequest(workspace.id))
    result = await service.exec(workspace.id, ExecRequest(cmd=("pwd",)))
    assert runtime.backend_runtime_id == "sbx_1"
    assert result.stdout == worktree_dir.encode()


@pytest.mark.asyncio
async def test_two_branches_share_one_sandbox() -> None:
    """Two workspace requests for the same project on different branches must
    target the same Daytona sandbox; the second one only adds a worktree."""
    client = FakeDaytonaClient()
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=DaytonaRuntimeProvider(workspace_provider),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )
    repo = RepoIdentity(
        repo_name="owner/private",
        repo_url="https://github.com/owner/private.git",
    )
    ws_a = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=repo,
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c-a",
            create_branch=True,
        )
    )
    ws_b = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=repo,
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c-b",
            create_branch=True,
        )
    )

    assert ws_a.id != ws_b.id
    assert ws_a.location.backend_workspace_id == ws_b.location.backend_workspace_id
    # One sandbox + one bare clone shared across the two workspaces.
    assert client.create_calls == 1
    sandbox = client._sandboxes["sbx_1"]
    clone_cmds = [
        c for c in sandbox.process.commands
        if "git clone --bare" in str(c.get("command"))
    ]
    assert len(clone_cmds) == 1
    # Each branch got its own worktree path.
    assert ws_a.location.remote_path != ws_b.location.remote_path
    worktree_cmds = [
        str(c["command"]) for c in sandbox.process.commands
        if "worktree add" in str(c.get("command"))
    ]
    assert sum(1 for cmd in worktree_cmds if "agent/edits-c-a" in cmd) == 1
    assert sum(1 for cmd in worktree_cmds if "agent/edits-c-b" in cmd) == 1


@pytest.mark.asyncio
async def test_separate_projects_get_separate_sandboxes() -> None:
    client = FakeDaytonaClient()
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=DaytonaRuntimeProvider(workspace_provider),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )

    repo = RepoIdentity(repo_name="owner/private", repo_url="https://example/x.git")
    ws_a = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=repo,
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )
    ws_b = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p2",
            repo=repo,
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )
    assert client.create_calls == 2
    assert ws_a.location.backend_workspace_id != ws_b.location.backend_workspace_id


@pytest.mark.asyncio
async def test_delete_workspace_keeps_sandbox_alive() -> None:
    client = FakeDaytonaClient()
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    runtime_provider = DaytonaRuntimeProvider(workspace_provider)
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=runtime_provider,
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )
    request = WorkspaceRequest(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(
            repo_name="owner/private", repo_url="https://x/.git"
        ),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
        create_branch=True,
    )
    workspace = await service.get_or_create_workspace(request)
    sandbox = client._sandboxes["sbx_1"]
    await service.destroy_workspace(workspace.id)
    assert sandbox.deleted is False
    # `worktree remove` and `rm -rf` were issued.
    assert any("worktree remove" in str(c["command"]) for c in sandbox.process.commands)
    assert any(str(c["command"]).startswith("rm -rf") for c in sandbox.process.commands)


@pytest.mark.asyncio
async def test_missing_snapshot_is_built_on_first_sandbox_creation(
    tmp_path,
) -> None:
    """First sandbox-create on a fresh Daytona auto-builds the snapshot.

    Operator no longer has to remember `make daytona-up` before the worker
    starts — the provider notices the missing snapshot, builds it from the
    bundled Dockerfile, then proceeds with sandbox creation.
    """
    client = FakeDaytonaClient(snapshot_exists=False)

    # Stand in for the bundled `images/agent-sandbox/Dockerfile`. Image.from_dockerfile
    # only needs the path to exist; the fake snapshot service doesn't read it.
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM busybox\n")

    # Stub the SDK Image.from_dockerfile so we don't need the real daytona
    # SDK installed in this test env.
    import sys
    import types

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    # `_create_sandbox` falls through to `CreateSandboxFromSnapshotParams` after
    # ensure-snapshot returns. Stub it so the test doesn't need the real SDK.
    fake_daytona.CreateSandboxFromSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type("DaytonaNotFoundError", (Exception,), {})  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    try:
        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/daytona/workspace",
            snapshot_dockerfile=str(dockerfile),
        )
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo", repo_url="https://github.com/owner/repo.git"
                ),
                base_ref="main",
                mode=WorkspaceMode.EDIT,
                conversation_id="c1",
                create_branch=True,
            )
        )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # Provider checked once, missed, then built the snapshot.
    assert client.snapshot.get_calls == ["potpie/agent-sandbox:0.1.0"]
    assert len(client.snapshot.create_calls) == 1
    # Ensure-snapshot is one-shot: a second sandbox-create in the same process
    # must not retry the build.
    await provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u2",
            project_id="p2",
            repo=RepoIdentity(
                repo_name="owner/repo", repo_url="https://github.com/owner/repo.git"
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c2",
            create_branch=True,
        )
    )
    assert len(client.snapshot.create_calls) == 1
    assert client.snapshot.get_calls == ["potpie/agent-sandbox:0.1.0"]


@pytest.mark.asyncio
async def test_snapshot_build_timeout_surfaces_runtime_unavailable(
    tmp_path,
) -> None:
    """A timed-out snapshot build raises RuntimeUnavailable, not a hang.

    Also: heartbeat must be cancelled and the ensured-flag must NOT latch, so
    a retry will re-check (the build may have finished out-of-band).
    """
    from sandbox.domain.errors import RuntimeUnavailable

    client = FakeDaytonaClient(snapshot_exists=False)

    # Override snapshot.create so it raises a timeout-shaped error after a
    # short delay, simulating the SDK's `@with_timeout()` firing.
    def slow_create(params: object, *, on_logs: object | None = None, timeout: float = 0):
        time.sleep(0.05)
        raise RuntimeError("Function 'create' exceeded timeout of 0.05 seconds.")

    client.snapshot.create = slow_create  # type: ignore[assignment]

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM busybox\n")

    import sys
    import types

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_daytona.CreateSandboxFromSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type("DaytonaNotFoundError", (Exception,), {})  # type: ignore[attr-defined]
    fake_errors_mod.DaytonaTimeoutError = type("DaytonaTimeoutError", (Exception,), {})  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    try:
        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/daytona/workspace",
            snapshot_dockerfile=str(dockerfile),
            snapshot_build_timeout_s=0.05,
            snapshot_heartbeat_s=0.01,
        )
        with pytest.raises(RuntimeUnavailable, match="timed out"):
            await provider.get_or_create_workspace(
                WorkspaceRequest(
                    user_id="u1",
                    project_id="p1",
                    repo=RepoIdentity(
                        repo_name="owner/repo", repo_url="https://github.com/owner/repo.git"
                    ),
                    base_ref="main",
                    mode=WorkspaceMode.EDIT,
                    conversation_id="c1",
                    create_branch=True,
                )
            )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # The build should not be marked ensured — a retry must re-check.
    assert provider._snapshot_ensured is False
    # No leaked heartbeat threads.
    leaked = [
        t for t in threading.enumerate() if t.name == "daytona-snapshot-heartbeat"
    ]
    assert leaked == []


@pytest.mark.asyncio
async def test_existing_pending_snapshot_is_waited_until_active() -> None:
    """Worker restart hits a half-built snapshot — wait, don't fail.

    Regression: previously `_ensure_snapshot` accepted any successful `get()`
    as ready. Daytona then refused sandbox creation with `Snapshot ... is
    pending`. Now we poll until state==active and only then proceed.
    """
    client = FakeDaytonaClient()
    # Sequence: pending → building → active. The third get() returns active.
    client.snapshot = FakeSnapshotService(
        existing=True, states=["pending", "building", "active"]
    )

    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        snapshot="potpie/agent-sandbox:0.1.0",
        snapshot_build_timeout_s=5,
        snapshot_heartbeat_s=0.05,
    )
    # Drive _ensure_snapshot directly — sandbox creation isn't what we're
    # testing here, and the FakeDaytonaClient's create() is fine to leave
    # untouched. Patch time.sleep so the 2s poll interval doesn't slow CI.
    import sandbox.adapters.outbound.daytona.provider as provider_module

    real_sleep = provider_module.time.sleep
    provider_module.time.sleep = lambda _s: real_sleep(0)
    try:
        provider._ensure_snapshot()
    finally:
        provider_module.time.sleep = real_sleep

    # Saw three get() calls: the initial check + two polls before active.
    assert client.snapshot.get_calls == ["potpie/agent-sandbox:0.1.0"] * 3
    # Did NOT call create — the snapshot already exists.
    assert client.snapshot.create_calls == []
    assert provider._snapshot_ensured is True


@pytest.mark.asyncio
async def test_existing_failed_snapshot_raises_immediately() -> None:
    """A snapshot stuck in error/build_failed must surface a clear error."""
    from sandbox.domain.errors import RuntimeUnavailable

    client = FakeDaytonaClient()
    client.snapshot = FakeSnapshotService(
        existing=True, states=["build_failed"]
    )
    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        snapshot="potpie/agent-sandbox:0.1.0",
    )
    with pytest.raises(RuntimeUnavailable, match="build_failed"):
        provider._ensure_snapshot()
    assert provider._snapshot_ensured is False


@pytest.mark.asyncio
async def test_recovers_existing_sandbox_across_provider_restart() -> None:
    """A fresh provider with no in-memory state must adopt an existing
    Daytona-side sandbox for the same (user, project), not orphan it.

    Regression: prior to label-based recovery, a worker restart (in-memory
    `_project_sandbox_ids` wiped, but the Daytona sandbox + worktree still
    on disk) caused the next request to ``_create_sandbox`` a fresh box
    and lose every commit the agent had made.
    """
    client = FakeDaytonaClient()
    # Pre-seed: this is what `client.list()` will return — a sandbox left
    # behind by the previous worker run.
    survivor = FakeSandbox(
        "sbx_survivor",
        labels={
            "managed-by": "potpie",
            "component": "sandbox-core",
            "potpie-user": "u1",
            "potpie-project": "p1",
        },
        state="started",
        created_at="2026-04-28T12:00:00Z",
    )
    survivor.process.fake_paths.add(
        "/home/daytona/workspace/owner_repo/.bare/HEAD"
    )
    client.register(survivor)

    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=DaytonaRuntimeProvider(workspace_provider),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )

    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(
                repo_name="owner/repo", repo_url="https://x/.git"
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )

    # Adopted the existing sandbox; no new sandbox spawned.
    assert workspace.location.backend_workspace_id == "sbx_survivor"
    assert client.create_calls == 0
    # Bare clone was already on disk on the survivor — no re-clone happened.
    bare_clone_cmds = [
        c for c in survivor.process.commands
        if "git clone --bare" in str(c.get("command"))
    ]
    assert bare_clone_cmds == []
    # And the lookup went through the labelled list filter.
    assert client.list_calls and client.list_calls[0] == {
        "managed-by": "potpie",
        "potpie-user": "u1",
        "potpie-project": "p1",
    }


@pytest.mark.asyncio
async def test_recovery_starts_stopped_sandbox() -> None:
    """A stopped survivor still gets adopted — provider calls .start() first."""
    client = FakeDaytonaClient()
    survivor = FakeSandbox(
        "sbx_stopped",
        labels={
            "managed-by": "potpie",
            "potpie-user": "u1",
            "potpie-project": "p1",
        },
        state="stopped",
    )
    client.register(survivor)

    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    sandbox = provider._ensure_sandbox(("u1", "p1"))

    assert sandbox.id == "sbx_stopped"
    assert survivor.started is True
    assert client.create_calls == 0


@pytest.mark.asyncio
async def test_recovery_skips_archived_sandbox() -> None:
    """Archived survivors have no volume — must not be adopted."""
    client = FakeDaytonaClient()
    archived = FakeSandbox(
        "sbx_archived",
        labels={
            "managed-by": "potpie",
            "potpie-user": "u1",
            "potpie-project": "p1",
        },
        state="archived",
    )
    client.register(archived)

    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    sandbox = provider._ensure_sandbox(("u1", "p1"))

    # Skipped the archived one and minted a new sandbox.
    assert sandbox.id != "sbx_archived"
    assert client.create_calls == 1
    assert archived.started is False


@pytest.mark.asyncio
async def test_recovery_handles_paginated_list_response() -> None:
    """Real Daytona SDK returns ``PaginatedSandboxes`` with ``.items``, not
    a plain list. Recovery must unwrap it — otherwise it silently bails out
    and creates a fresh sandbox, losing the agent's prior commits.

    Regression: the first cut of recovery used ``isinstance(candidates, list)``
    which is False for ``PaginatedSandboxes``, so every restart created a
    new sandbox instead of adopting the existing one.
    """

    class _Paginated:
        """Stand-in for ``daytona._sync.sandbox.PaginatedSandboxes``."""

        def __init__(self, items: list[FakeSandbox]) -> None:
            self.items = items

    survivor = FakeSandbox(
        "sbx_paginated",
        labels={
            "managed-by": "potpie",
            "potpie-user": "u1",
            "potpie-project": "p1",
        },
        state="started",
    )

    class PaginatedClient(FakeDaytonaClient):
        def list(self, labels: dict[str, str] | None = None) -> _Paginated:  # type: ignore[override]
            self.list_calls.append(labels)
            base = list(self._sandboxes.values())
            if labels:
                base = [
                    s for s in base
                    if all(s.labels.get(k) == v for k, v in labels.items())
                ]
            return _Paginated(base)

    client = PaginatedClient()
    client.register(survivor)

    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    sandbox = provider._ensure_sandbox(("u1", "p1"))

    assert sandbox.id == "sbx_paginated"
    assert client.create_calls == 0
