from __future__ import annotations

import threading
import time
import types

import pytest

pytestmark = pytest.mark.unit

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


class FakeFs:
    """Records sandbox.fs.* calls for tests asserting native fs use.

    Mirrors the SDK's surface narrowly — ``upload_file`` /
    ``download_file`` / ``list_files`` / ``create_folder`` are the only
    methods the provider touches today.
    """

    def __init__(self) -> None:
        self.uploads: list[tuple[bytes, str]] = []
        self.downloads: list[str] = []
        self.lists: list[str] = []
        self.folders: list[tuple[str, str]] = []
        self.files: dict[str, bytes] = {}
        self.dirs: dict[str, list[tuple[str, bool, int | None]]] = {}

    def upload_file(self, content: bytes, remote_path: str, timeout: int = 0) -> None:
        self.uploads.append((content, remote_path))
        self.files[remote_path] = content

    def download_file(self, remote_path: str, timeout: int = 0) -> bytes:
        self.downloads.append(remote_path)
        return self.files.get(remote_path, b"")

    def create_folder(self, path: str, mode: str) -> None:
        self.folders.append((path, mode))

    def list_files(self, path: str) -> list[object]:
        self.lists.append(path)
        entries = self.dirs.get(path, [])
        return [
            types.SimpleNamespace(name=n, is_dir=d, size=s)
            for (n, d, s) in entries
        ]


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
        self.fs = FakeFs()
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


class FakeVolumeService:
    """Records volume.get / volume.create calls.

    Mirrors the SDK's idempotent ``get(name, create=True)`` shape — a
    miss with ``create=True`` mints a new ``Volume`` and caches it; a
    subsequent ``get`` returns the same volume.
    """

    def __init__(self) -> None:
        self._volumes: dict[str, types.SimpleNamespace] = {}
        self.get_calls: list[tuple[str, bool]] = []
        self.create_calls: list[str] = []
        self._next_id = 0

    def get(self, name: str, create: bool = False) -> types.SimpleNamespace:
        self.get_calls.append((name, create))
        if name in self._volumes:
            return self._volumes[name]
        if not create:
            raise RuntimeError(f"volume {name!r} not found")
        return self.create(name)

    def create(self, name: str) -> types.SimpleNamespace:
        self.create_calls.append(name)
        self._next_id += 1
        volume = types.SimpleNamespace(
            id=f"vol_{self._next_id}", name=name, state="active"
        )
        self._volumes[name] = volume
        return volume


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
        self.volume = FakeVolumeService()

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
    # P4 hardening: worktree path is `<user>_<scope>_<branch>` so two
    # conversations on the same branch never share a worktree.
    worktree_dir = (
        "/home/daytona/workspace/owner_private/worktrees/u1_c1_agent_edits-c1"
    )

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
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: kw  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
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
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: kw  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
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


# ----------------------------------------------------------------------
# P4 hardening
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_distinct_conversations_on_same_branch_get_distinct_worktrees() -> None:
    """Two conversations forking the same agent branch must NOT share a
    worktree path (gap 7 in the audit). Without user/scope encoding the
    second run would land on the first run's edits."""
    client = FakeDaytonaClient()
    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )

    branch = "feat/x"
    ws_a = await provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="conv-A",
            branch_name=branch,
            create_branch=True,
        )
    )
    ws_b = await provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="conv-B",
            branch_name=branch,
            create_branch=True,
        )
    )
    assert ws_a.location.remote_path != ws_b.location.remote_path
    assert "conv-A" in ws_a.location.remote_path
    assert "conv-B" in ws_b.location.remote_path


@pytest.mark.asyncio
async def test_invalid_base_ref_rejected() -> None:
    """Newlines / `..` in the ref must be rejected before the SDK call."""
    client = FakeDaytonaClient()
    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    with pytest.raises(ValueError, match="unsafe"):
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
                ),
                base_ref="main\nrm -rf /",
                mode=WorkspaceMode.EDIT,
                conversation_id="c1",
            )
        )


@pytest.mark.asyncio
async def test_invalid_branch_name_rejected() -> None:
    """Same threat model for branch_name."""
    client = FakeDaytonaClient()
    provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/daytona/workspace",
    )
    with pytest.raises(ValueError, match="unsafe"):
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
                ),
                base_ref="main",
                branch_name="main\n--evil",
                mode=WorkspaceMode.EDIT,
                conversation_id="c1",
                create_branch=True,
            )
        )


# ----------------------------------------------------------------------
# Recovery: dead-sandbox detection at exec time.
#
# Scenario: a workspace + runtime were persisted in a previous worker
# session. The Daytona-side sandbox got destroyed (TTL eviction, manual
# cleanup, infra outage). On the next exec we'd otherwise bubble a
# "Sandbox with ID … not found" all the way to the agent's tool result.
# The recovery path should adopt the project's surviving sandbox (or
# create a fresh one) and retry transparently.
# ----------------------------------------------------------------------


class _DeadThenLiveProcess(FakeProcess):
    """Process that fails the first exec call with a 404, then succeeds.

    Models the Daytona quirk where ``client.get(id)`` returns a stale
    handle (so ``_lookup_sandbox_by_id`` succeeds) but the actual
    ``sandbox.process.exec`` discovers the sandbox is gone server-side.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dead = True

    def exec(self, command: str, **kwargs: object):
        if self.dead:
            self.dead = False
            raise RuntimeError(
                "bad request: failed to get runner info: "
                "Sandbox with ID dead-id not found"
            )
        return super().exec(command, **kwargs)


@pytest.mark.asyncio
async def test_exec_recovers_when_persisted_sandbox_id_is_dead() -> None:
    """Stale persisted sandbox id → exec recovers via labels and retries."""
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

    # 1) Acquire workspace + runtime so the store + caches are populated.
    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u-recovery",
            project_id="p-recovery",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="conv1",
            create_branch=True,
        )
    )
    await service.get_or_create_runtime(RuntimeRequest(workspace.id))

    # 2) Simulate the live Daytona sandbox being destroyed: drop it from
    #    the SDK fake. The persisted runtime still references its id.
    live = client._sandboxes.pop("sbx_1")
    # Re-register a survivor with the recovery labels the production
    # ``_recover_project_sandbox`` filters on. The default test fake
    # creates sandboxes with empty labels (because no ``snapshot`` is
    # configured on the provider, so ``_create_sandbox`` falls through
    # to the no-args ``client.create()`` path) — we hand-set them here
    # so the label-based lookup works exactly like production.
    survivor_labels = {
        "managed-by": "potpie",
        "component": "sandbox-core",
        "potpie-user": "u-recovery",
        "potpie-project": "p-recovery",
    }
    survivor = FakeSandbox(
        "sbx_survivor",
        labels=survivor_labels,
        state="started",
    )
    survivor.process.fake_paths = set(live.process.fake_paths)
    client.register(survivor)

    # 3) Make the existing (dead) sandbox's process raise the 404 on its
    #    next exec. The provider's in-memory cache still holds the dead
    #    handle from creation time, so this simulates the actual
    #    production failure mode.
    workspace_provider._sandboxes["sbx_1"] = live  # ensure cache hit
    live.process = _DeadThenLiveProcess()
    live.process.fake_paths = set()  # legitimately "dead" sandbox

    # 4) Drop the in-memory project_sandbox_ids → forces label recovery.
    workspace_provider._project_sandbox_ids.clear()

    # Force a fresh runtime fetch from the store so its
    # ``backend_runtime_id`` is the dead "sbx_1" — exactly what would
    # happen on a worker restart.
    runtime_provider._runtimes.clear()

    initial_list_calls = len(client.list_calls)
    # 5) The exec should NOT raise; it must recover via labels.
    result = await service.exec(workspace.id, ExecRequest(cmd=("echo", "ok")))
    assert result.exit_code == 0

    # The label-based recovery path was exercised — at least one
    # subsequent ``list(labels=…)`` call was made to find the survivor.
    assert len(client.list_calls) > initial_list_calls

    # Runtime now points at the live sandbox, not the dead one.
    runtime = await service._store.find_runtime_by_workspace(workspace.id)
    assert runtime is not None
    assert runtime.backend_runtime_id == "sbx_survivor"


@pytest.mark.asyncio
async def test_exec_recovery_creates_fresh_sandbox_when_no_survivor() -> None:
    """No labelled survivor → recovery creates a brand-new sandbox."""
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

    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u-fresh",
            project_id="p-fresh",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="conv1",
            create_branch=True,
        )
    )
    await service.get_or_create_runtime(RuntimeRequest(workspace.id))

    # Wipe out the only sandbox so no labelled survivor exists. Recovery
    # must fall through to ``_create_sandbox``.
    dead = client._sandboxes.pop("sbx_1")
    dead.process = _DeadThenLiveProcess()
    workspace_provider._sandboxes["sbx_1"] = dead
    workspace_provider._project_sandbox_ids.clear()
    runtime_provider._runtimes.clear()

    initial_creates = client.create_calls
    result = await service.exec(workspace.id, ExecRequest(cmd=("echo", "fresh")))
    assert result.exit_code == 0
    # Recovery did create a brand-new sandbox.
    assert client.create_calls == initial_creates + 1


@pytest.mark.asyncio
async def test_exec_recovery_no_labels_propagates_error() -> None:
    """Without (user_id, project_id) labels, recovery can't safely route."""
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

    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u-no-label",
            project_id="p-no-label",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="conv1",
            create_branch=True,
        )
    )
    runtime = await service.get_or_create_runtime(RuntimeRequest(workspace.id))

    # Strip the recovery labels off the runtime spec so the adapter
    # has nothing to route on (simulates an old persisted runtime row
    # from before the label stamping landed). Use dataclasses.replace
    # since RuntimeSpec is frozen.
    from dataclasses import replace as _dc_replace
    runtime.spec = _dc_replace(runtime.spec, labels={})
    await service._store.save_runtime(runtime)

    # Simulate the dead sandbox.
    dead = client._sandboxes.pop("sbx_1")
    dead.process = _DeadThenLiveProcess()
    workspace_provider._sandboxes["sbx_1"] = dead
    workspace_provider._project_sandbox_ids.clear()
    runtime_provider._runtimes.clear()

    with pytest.raises(RuntimeError, match="Sandbox with ID dead-id not found"):
        await service.exec(workspace.id, ExecRequest(cmd=("echo", "x")))


# ----------------------------------------------------------------------------
# Native fs path: writes / reads / list_dir on Daytona must go through
# ``sandbox.fs.*`` rather than the broken ``sandbox.process.exec`` stdin pipe.
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_daytona_runtime_write_bytes_uses_fs_upload() -> None:
    """``DaytonaRuntimeProvider.write_bytes`` must hit ``sandbox.fs.upload_file``.

    Pre-fix the same call routed through ``process.exec`` with a ``cat > path``
    command. Daytona's SDK doesn't pipe ``stdin`` into ``process.exec``, so
    that path silently produced empty files. This test pins the new
    behaviour: every byte goes through ``upload_file``, the parent dir is
    created via ``create_folder``, and ``process.exec`` is NOT consulted.
    """
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
    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )
    runtime = await service.get_or_create_runtime(RuntimeRequest(workspace.id))
    sandbox = client.get(runtime.backend_runtime_id)
    process_calls_before = len(sandbox.process.commands)

    ok = await service.fs_write_file(
        workspace.id, "/home/daytona/workspace/owner_private/x.txt", b"hello"
    )
    assert ok is True
    assert sandbox.fs.uploads == [(b"hello", "/home/daytona/workspace/owner_private/x.txt")]
    # Parent dir created via fs.create_folder, not via shell mkdir.
    assert sandbox.fs.folders == [("/home/daytona/workspace/owner_private", "755")]
    # ``process.exec`` must not have run during the write — that was the
    # broken path.
    assert len(sandbox.process.commands) == process_calls_before


@pytest.mark.asyncio
async def test_daytona_runtime_read_and_list_use_fs() -> None:
    """``read_bytes`` / ``list_dir_native`` round-trip through ``sandbox.fs``."""
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
    workspace = await service.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )
    runtime = await service.get_or_create_runtime(RuntimeRequest(workspace.id))
    sandbox = client.get(runtime.backend_runtime_id)
    sandbox.fs.files["/home/daytona/workspace/owner_private/x.txt"] = b"hi"
    sandbox.fs.dirs["/home/daytona/workspace/owner_private"] = [
        ("x.txt", False, 2), ("subdir", True, None),
    ]

    body = await service.fs_read_file(
        workspace.id, "/home/daytona/workspace/owner_private/x.txt"
    )
    assert body == b"hi"

    listing = await service.fs_list_dir(
        workspace.id, "/home/daytona/workspace/owner_private"
    )
    assert listing == [("x.txt", False, 2), ("subdir", True, None)]


# ----------------------------------------------------------------------------
# Sandbox creation knobs: name, auto_delete, network policy.
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_sandbox_passes_name_and_network_params() -> None:
    """Custom name + network knobs reach ``CreateSandboxFromSnapshotParams``.

    Captures the kwargs the SDK sees so a future regression (e.g. dropping a
    field on a refactor) shows up here instead of as a missing-name complaint
    from ops or an open-egress incident from security.
    """
    captured: list[dict[str, object]] = []

    class _CapturingParams:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.labels = kwargs.get("labels", {})

    import sys

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSandboxFromSnapshotParams = _CapturingParams  # type: ignore[attr-defined]
    fake_daytona.CreateSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type(  # type: ignore[attr-defined]
        "DaytonaNotFoundError", (Exception,), {}
    )
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: kw  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    try:
        client = FakeDaytonaClient()
        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/daytona/workspace",
            sandbox_name_prefix="potpie",
            auto_delete_minutes=60,
            network_allow_list="192.0.2.0/24,198.51.100.0/24",
        )
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="user-12345678abcd",
                project_id="proj-87654321zyxw",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
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

    assert len(captured) == 1
    kwargs = captured[0]
    # Name encodes the (user, project) pair so the dashboard groups
    # related sandboxes; truncated to 8 chars per segment for readability.
    assert kwargs["name"] == "potpie-user-123-proj-876"
    assert kwargs["auto_delete_interval"] == 60
    assert kwargs["network_allow_list"] == "192.0.2.0/24,198.51.100.0/24"
    # network_block_all is mutually exclusive with allow_list — neither
    # set together.
    assert "network_block_all" not in kwargs


@pytest.mark.asyncio
async def test_create_sandbox_block_all_takes_precedence() -> None:
    """``network_block_all=True`` clears the allow-list (Daytona requires this).

    The Daytona SDK rejects passing both — block_all wins per docs.
    """
    captured: list[dict[str, object]] = []

    class _CapturingParams:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.labels = kwargs.get("labels", {})

    import sys

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSandboxFromSnapshotParams = _CapturingParams  # type: ignore[attr-defined]
    fake_daytona.CreateSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type(  # type: ignore[attr-defined]
        "DaytonaNotFoundError", (Exception,), {}
    )
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: kw  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    try:
        client = FakeDaytonaClient()
        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/daytona/workspace",
            network_block_all=True,
            network_allow_list="ignored.if.block_all/0",
        )
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
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

    kwargs = captured[0]
    assert kwargs["network_block_all"] is True
    assert "network_allow_list" not in kwargs


@pytest.mark.asyncio
async def test_snapshot_build_passes_resources(tmp_path) -> None:
    """``CreateSnapshotParams`` carries the configured Resources.

    Snapshot resources are immutable post-build, so this is the only
    place we get to size the sandbox. Defaults for the agent runtime
    must beat the SDK's 1/1/3 floor.
    """
    captured: list[dict[str, object]] = []

    def _capturing_create_snapshot(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return kwargs

    import sys

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSnapshotParams = _capturing_create_snapshot  # type: ignore[attr-defined]
    fake_daytona.CreateSandboxFromSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type(  # type: ignore[attr-defined]
        "DaytonaNotFoundError", (Exception,), {}
    )
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: ("Resources", kw)  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM busybox\n")
    try:
        client = FakeDaytonaClient(snapshot_exists=False)
        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/daytona/workspace",
            snapshot_dockerfile=str(dockerfile),
            snapshot_cpu=4,
            snapshot_memory_gb=8,
            snapshot_disk_gb=20,
        )
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
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

    assert len(captured) == 1
    snap_kwargs = captured[0]
    assert snap_kwargs["resources"] == ("Resources", {"cpu": 4, "memory": 8, "disk": 20})


# ----------------------------------------------------------------------------
# Stderr fallback: Daytona's process.exec collapses stdout+stderr into one
# stream, so callers that only render ``result.stderr`` get empty error
# messages. The tool-level wrapper now falls back to stdout on failures.
# ----------------------------------------------------------------------------
def test_err_payload_prefers_stderr_falls_back_to_stdout() -> None:
    """``_err_payload`` masks the Daytona "no stderr" wart.

    Daytona's process.exec collapses stdout+stderr into ``result``; the
    ``ExecResult.stderr`` we surface to callers is always empty on that
    backend. The error formatters call ``_err_payload`` so the LLM sees
    the actual diagnostic instead of an empty string. Backends that
    split the streams (local, Docker) keep stderr precedence.
    """
    from sandbox.api.client import _err_payload
    from sandbox.domain.models import ExecResult

    daytona_shape = ExecResult(exit_code=1, stdout=b"fatal: bad ref", stderr=b"")
    assert _err_payload(daytona_shape) == "fatal: bad ref"

    split_shape = ExecResult(
        exit_code=2, stdout=b"some progress", stderr=b"real error"
    )
    assert _err_payload(split_shape) == "real error"

    empty = ExecResult(exit_code=0, stdout=b"", stderr=b"")
    assert _err_payload(empty) == ""


# ----------------------------------------------------------------------------
# Bare-cache volume: opt-in mount that survives sandbox destruction.
# ----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_volume_off_uses_local_fs_bare_path() -> None:
    """Default (flag off) keeps the existing local-fs bare layout.

    The volume code is opt-in; deploys that haven't flipped the flag
    must see ``volume.get`` never called and bare paths under
    ``<workspace_root>/<repo>/.bare`` exactly as before.
    """
    client = FakeDaytonaClient()
    workspace_provider = DaytonaWorkspaceProvider(
        client_factory=lambda: client,
        workspace_root="/home/agent/work",
        # use_volume_for_bare defaults to False
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
                repo_name="owner/repo",
                repo_url="https://github.com/owner/repo.git",
            ),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
            create_branch=True,
        )
    )

    assert workspace_provider._bare_path("owner/repo") == "/home/agent/work/owner_repo/.bare"
    assert client.volume.get_calls == []
    assert client.volume.create_calls == []


@pytest.mark.asyncio
async def test_volume_on_get_or_creates_per_user_volume() -> None:
    """Flag on: per-user volume is get-or-created exactly once.

    Two projects for the same user share one volume — that's what
    keeps us under Daytona's 100-volume-per-org cap. Per-project
    isolation comes from the ``subpath`` mount option, asserted in
    the next test.
    """
    captured: list[dict[str, object]] = []

    class _CapturingParams:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.labels = kwargs.get("labels", {})

    import sys

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSandboxFromSnapshotParams = _CapturingParams  # type: ignore[attr-defined]
    fake_daytona.CreateSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type(  # type: ignore[attr-defined]
        "DaytonaNotFoundError", (Exception,), {}
    )
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: kw  # type: ignore[attr-defined]
    fake_volume_mod = types.ModuleType("daytona.common.volume")
    fake_volume_mod.VolumeMount = lambda **kw: ("VolumeMount", kw)  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
        "daytona.common.volume": fake_volume_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    try:
        client = FakeDaytonaClient()
        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/agent/work",
            use_volume_for_bare=True,
            volume_name_prefix="potpie-bare",
            volume_mount_path="/home/agent/work/.bare-cache",
        )
        # First project for user u1 — creates the volume.
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
                ),
                base_ref="main",
                mode=WorkspaceMode.EDIT,
                conversation_id="c1",
                create_branch=True,
            )
        )
        # Second project for the same user — reuses the volume.
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p2",
                repo=RepoIdentity(
                    repo_name="owner/other",
                    repo_url="https://github.com/owner/other.git",
                ),
                base_ref="main",
                mode=WorkspaceMode.EDIT,
                conversation_id="c2",
                create_branch=True,
            )
        )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # Two sandboxes (one per project), one volume.
    assert len(captured) == 2
    assert client.volume.create_calls == ["potpie-bare-u1"]
    # Both sandboxes mount the same volume; subpath differs.
    mount_a = captured[0]["volumes"][0]
    mount_b = captured[1]["volumes"][0]
    assert mount_a == ("VolumeMount", {
        "volume_id": "vol_1",
        "mount_path": "/home/agent/work/.bare-cache",
        "subpath": "p1",
    })
    assert mount_b == ("VolumeMount", {
        "volume_id": "vol_1",
        "mount_path": "/home/agent/work/.bare-cache",
        "subpath": "p2",
    })
    # Bare path lives inside the mount, not under <repo>/.bare.
    assert provider._bare_path("owner/repo") == "/home/agent/work/.bare-cache/.bare"


@pytest.mark.asyncio
async def test_volume_failure_falls_back_to_local_fs() -> None:
    """A volume-service hiccup must not break sandbox creation.

    The bare clone falls back to local fs (the original behaviour) —
    losing the cross-restart cache, but keeping the sandbox flow
    working. Critical for resilience: a volume API outage shouldn't
    take agent execution down.
    """
    captured: list[dict[str, object]] = []

    class _CapturingParams:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.labels = kwargs.get("labels", {})

    import sys

    fake_daytona = types.ModuleType("daytona")
    fake_daytona.CreateSandboxFromSnapshotParams = _CapturingParams  # type: ignore[attr-defined]
    fake_daytona.CreateSnapshotParams = lambda **kw: kw  # type: ignore[attr-defined]
    fake_image_mod = types.ModuleType("daytona.common.image")
    fake_image_mod.Image = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_dockerfile=lambda path: {"dockerfile": str(path)}
    )
    fake_errors_mod = types.ModuleType("daytona.common.errors")
    fake_errors_mod.DaytonaNotFoundError = type(  # type: ignore[attr-defined]
        "DaytonaNotFoundError", (Exception,), {}
    )
    fake_sandbox_mod = types.ModuleType("daytona.common.sandbox")
    fake_sandbox_mod.Resources = lambda **kw: kw  # type: ignore[attr-defined]
    fake_volume_mod = types.ModuleType("daytona.common.volume")
    fake_volume_mod.VolumeMount = lambda **kw: ("VolumeMount", kw)  # type: ignore[attr-defined]
    monkeyed = {
        "daytona": fake_daytona,
        "daytona.common.image": fake_image_mod,
        "daytona.common.errors": fake_errors_mod,
        "daytona.common.sandbox": fake_sandbox_mod,
        "daytona.common.volume": fake_volume_mod,
    }
    saved = {k: sys.modules.get(k) for k in monkeyed}
    sys.modules.update(monkeyed)
    try:
        client = FakeDaytonaClient()

        def _broken_get(name: str, create: bool = False):
            raise RuntimeError("volume API down")

        client.volume.get = _broken_get  # type: ignore[assignment]

        provider = DaytonaWorkspaceProvider(
            client_factory=lambda: client,
            snapshot="potpie/agent-sandbox:0.1.0",
            workspace_root="/home/agent/work",
            use_volume_for_bare=True,
        )
        await provider.get_or_create_workspace(
            WorkspaceRequest(
                user_id="u1",
                project_id="p1",
                repo=RepoIdentity(
                    repo_name="owner/repo",
                    repo_url="https://github.com/owner/repo.git",
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

    # Sandbox got created without a mount because the volume call failed.
    assert len(captured) == 1
    assert "volumes" not in captured[0]
