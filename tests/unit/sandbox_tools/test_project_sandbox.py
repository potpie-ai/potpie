"""Unit tests for the ``ProjectSandbox`` lifecycle facade.

These exercise the behaviors the parsing pipeline and conversation_service
will rely on in phases 3 and 4: idempotent ensure, self-healing recovery
when the backing storage is gone, and the cheap health-check / parse
wrappers. A FakeSandboxClient stands in for the real backend so the
tests don't need a live Daytona or filesystem worktree.

The contract surface tested here is the same one
``ProjectSandbox.ensure`` calls into — ``acquire_session``, ``is_alive``,
``destroy_workspace``, ``parse_repo``. The fake mirrors those exactly so
a future change to the real client surfaces here as a test failure
rather than a silent skew.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from app.modules.intelligence.tools.sandbox import project_sandbox
from app.modules.intelligence.tools.sandbox.project_sandbox import (
    ProjectRef,
    ProjectSandbox,
    get_project_sandbox,
    set_project_sandbox,
)
from sandbox import WorkspaceHandle
from sandbox.api.parser_wire import ParseArtifacts


@dataclass
class _Call:
    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeSandboxClient:
    """In-memory client tracking calls + scripted responses.

    ``alive_for`` controls what ``is_alive`` returns per workspace_id; a
    sequence semantics — pop the next answer when called, default to True.
    Lets tests drive the recovery path deterministically without timing
    or thread tricks.
    """

    def __init__(self) -> None:
        self.calls: list[_Call] = []
        self.alive_for: dict[str, list[bool]] = {}
        self.acquire_call_count = 0
        self.next_workspace_id = "ws_1"
        self.parse_artifacts: ParseArtifacts | None = None
        self.destroy_should_raise: Exception | None = None

    def _next_handle(self) -> WorkspaceHandle:
        wid = f"ws_{self.acquire_call_count}"
        return WorkspaceHandle(
            workspace_id=wid,
            branch="main",
            backend_kind="local",
            local_path=f"/tmp/fake/{wid}",
            remote_path=None,
        )

    async def acquire_session(self, **kwargs: Any) -> WorkspaceHandle:
        self.acquire_call_count += 1
        self.calls.append(_Call("acquire_session", kwargs=kwargs))
        return self._next_handle()

    async def is_alive(self, handle: WorkspaceHandle) -> bool:
        self.calls.append(
            _Call("is_alive", kwargs={"workspace_id": handle.workspace_id})
        )
        sequence = self.alive_for.get(handle.workspace_id)
        if sequence:
            return sequence.pop(0)
        return True

    async def destroy_workspace(self, handle: WorkspaceHandle) -> None:
        self.calls.append(
            _Call("destroy_workspace", kwargs={"workspace_id": handle.workspace_id})
        )
        if self.destroy_should_raise is not None:
            raise self.destroy_should_raise

    async def parse_repo(
        self,
        handle: WorkspaceHandle,
        *,
        repo_subdir: str | None = None,
        timeout_s: int = 600,
    ) -> ParseArtifacts:
        self.calls.append(
            _Call(
                "parse_repo",
                kwargs={
                    "workspace_id": handle.workspace_id,
                    "repo_subdir": repo_subdir,
                    "timeout_s": timeout_s,
                },
            )
        )
        return self.parse_artifacts or ParseArtifacts()


@pytest.fixture
def fake_client() -> Iterator[FakeSandboxClient]:
    client = FakeSandboxClient()
    yield client
    set_project_sandbox(None)


@pytest.fixture
def stub_auth():
    """Pin ``_resolve_auth_token`` to a known string so tests don't depend
    on env vars or the real auth chain."""
    with patch.object(
        project_sandbox, "_resolve_auth_token", return_value="stub-token"
    ):
        yield


REPO = ProjectRef(repo_name="owner/repo", base_ref="main",
                  repo_url="https://github.com/owner/repo.git")


# ---------------------------------------------------------------------------
# ensure() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_happy_path_returns_handle(fake_client, stub_auth) -> None:
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    handle = await facade.ensure(user_id="u1", project_id="p1", repo=REPO)
    assert handle.workspace_id == "ws_1"

    # One acquire, one is_alive probe — no recovery roundtrip.
    methods = [c.method for c in fake_client.calls]
    assert methods == ["acquire_session", "is_alive"]


@pytest.mark.asyncio
async def test_ensure_uses_analysis_mode_with_base_ref_as_branch(
    fake_client, stub_auth
) -> None:
    """The Daytona provider keys workspaces on (user, project, repo,
    mode, base_ref). ANALYSIS mode + branch=base_ref ⇒ the parsing
    workspace, not the per-conversation EDIT workspace. Pin both so a
    refactor that flips them shows up here loudly."""
    from sandbox import WorkspaceMode

    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    await facade.ensure(user_id="u1", project_id="p1", repo=REPO)
    acquire = next(c for c in fake_client.calls if c.method == "acquire_session")
    assert acquire.kwargs["mode"] is WorkspaceMode.ANALYSIS
    assert acquire.kwargs["branch"] == "main"
    assert acquire.kwargs["base_ref"] == "main"
    assert acquire.kwargs["repo"] == "owner/repo"
    assert acquire.kwargs["repo_url"] == "https://github.com/owner/repo.git"


@pytest.mark.asyncio
async def test_ensure_passes_explicit_auth_token_through(
    fake_client, stub_auth
) -> None:
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    await facade.ensure(
        user_id="u1", project_id="p1", repo=REPO, auth_token="explicit-token"
    )
    acquire = next(c for c in fake_client.calls if c.method == "acquire_session")
    # Explicit caller-supplied token wins over the resolver.
    assert acquire.kwargs["auth_token"] == "explicit-token"


@pytest.mark.asyncio
async def test_ensure_falls_back_to_resolver_when_token_missing(
    fake_client, stub_auth
) -> None:
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    await facade.ensure(user_id="u1", project_id="p1", repo=REPO)
    acquire = next(c for c in fake_client.calls if c.method == "acquire_session")
    # The stub_auth fixture pins the resolver to "stub-token".
    assert acquire.kwargs["auth_token"] == "stub-token"


# ---------------------------------------------------------------------------
# ensure() — recovery path when is_alive fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_recovers_when_first_handle_is_dead(
    fake_client, stub_auth
) -> None:
    """The persisted store thinks the sandbox lives but the backend
    has archived/deleted it. ensure() must detect that on the cheap
    is_alive probe and re-acquire. Otherwise the very next exec would
    fail and the conversation would crash."""
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    # First handle is dead, second is alive.
    fake_client.alive_for["ws_1"] = [False]
    fake_client.alive_for["ws_2"] = [True]

    handle = await facade.ensure(user_id="u1", project_id="p1", repo=REPO)
    assert handle.workspace_id == "ws_2"

    methods = [c.method for c in fake_client.calls]
    # acquire → is_alive(False) → destroy → acquire (no second is_alive
    # — the recovery path trusts the second acquire's freshness rather
    # than burning another probe round-trip).
    assert methods == ["acquire_session", "is_alive", "destroy_workspace",
                       "acquire_session"]


@pytest.mark.asyncio
async def test_ensure_swallows_destroy_failure_during_recovery(
    fake_client, stub_auth
) -> None:
    """If destroy_workspace itself fails (because the backing sandbox
    is already gone — exactly the case we're recovering from), we
    proceed to re-acquire anyway. Otherwise the recovery path would
    fail closed and the conversation couldn't proceed."""
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    fake_client.alive_for["ws_1"] = [False]
    fake_client.destroy_should_raise = RuntimeError("sandbox not found")

    handle = await facade.ensure(user_id="u1", project_id="p1", repo=REPO)
    assert handle.workspace_id == "ws_2"


# ---------------------------------------------------------------------------
# health_check / parse — thin wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_delegates_to_is_alive(fake_client) -> None:
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    handle = WorkspaceHandle(
        workspace_id="ws_x", branch="main", backend_kind="local",
        local_path="/tmp/x",
    )
    fake_client.alive_for["ws_x"] = [False]
    assert await facade.health_check(handle) is False
    assert fake_client.calls[-1].method == "is_alive"


@pytest.mark.asyncio
async def test_parse_delegates_to_parse_repo(fake_client) -> None:
    facade = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    handle = WorkspaceHandle(
        workspace_id="ws_x", branch="main", backend_kind="local",
        local_path="/tmp/x",
    )
    fake_artifacts = ParseArtifacts()
    fake_client.parse_artifacts = fake_artifacts
    out = await facade.parse(handle, repo_subdir="src", timeout_s=42)
    assert out is fake_artifacts

    parse_call = fake_client.calls[-1]
    assert parse_call.method == "parse_repo"
    assert parse_call.kwargs["repo_subdir"] == "src"
    assert parse_call.kwargs["timeout_s"] == 42


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------


def test_get_project_sandbox_singleton(fake_client) -> None:
    """The accessor caches a single facade per process so callers
    don't pay setup cost on each invocation."""
    set_project_sandbox(None)
    facade_a = get_project_sandbox()
    facade_b = get_project_sandbox()
    assert facade_a is facade_b


def test_set_project_sandbox_overrides_accessor(fake_client) -> None:
    set_project_sandbox(None)
    custom = ProjectSandbox(client=fake_client)  # type: ignore[arg-type]
    set_project_sandbox(custom)
    assert get_project_sandbox() is custom
