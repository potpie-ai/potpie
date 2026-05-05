from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy
from sandbox.domain.models import (
    RepoIdentity,
    WorkspaceMode,
    WorkspaceRequest,
)
from sandbox.domain.ports.eviction import EvictionResult


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _run(["git", "init", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test User"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], repo)
    _run(["git", "commit", "-m", "initial"], repo)
    return repo


class _RecordingEvictionPolicy:
    """Test policy that records every evict_if_needed call."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def evict_if_needed(
        self, *, user_id: str | None = None
    ) -> EvictionResult:
        self.calls.append(user_id)
        return EvictionResult()


@pytest.mark.asyncio
async def test_noop_policy_returns_empty_result() -> None:
    policy = NoOpEvictionPolicy()
    result = await policy.evict_if_needed(user_id="u1")
    assert result == EvictionResult()
    assert result.is_empty


@pytest.mark.asyncio
async def test_local_provider_calls_evict_on_create_only(tmp_path: Path) -> None:
    """First call (cache miss) runs the policy; second call (cache hit) skips it."""
    source = _make_repo(tmp_path)
    policy = _RecordingEvictionPolicy()
    provider = LocalGitWorkspaceProvider(tmp_path / ".repos", eviction=policy)

    request = WorkspaceRequest(
        user_id="u1",
        project_id="p1",
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        mode=WorkspaceMode.EDIT,
        conversation_id="c1",
    )

    await provider.get_or_create_workspace(request)
    assert policy.calls == ["u1"]

    # Same key again — in-memory cache hit, eviction must NOT be called
    # again. Otherwise we'd thrash the policy on every read.
    await provider.get_or_create_workspace(request)
    assert policy.calls == ["u1"]


@pytest.mark.asyncio
async def test_local_provider_defaults_to_noop_eviction(tmp_path: Path) -> None:
    """No explicit policy ⇒ NoOp. Workspace creation still succeeds."""
    source = _make_repo(tmp_path)
    provider = LocalGitWorkspaceProvider(tmp_path / ".repos")

    workspace = await provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    assert workspace.location.local_path is not None
    assert isinstance(provider._eviction, NoOpEvictionPolicy)
