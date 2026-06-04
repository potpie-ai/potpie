"""Tests for the GitPlatformProvider port and SandboxClient PR surface.

Covers: service-level dispatch through a fake provider, the writable-
capability guard on `SandboxClient.create_pull_request`, and the typed
`PullRequestFailed` error path.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from sandbox import (
    Capabilities,
    PullRequest,
    PullRequestRequest,
    RepoIdentity,
    SandboxClient,
    SandboxContainer,
    SandboxService,
    WorkspaceHandle,
    WorkspaceMode,
)
from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.repo_cache import LocalRepoCacheProvider
from sandbox.adapters.outbound.local.subprocess_runtime import (
    LocalSubprocessRuntimeProvider,
)
from sandbox.adapters.outbound.memory.eviction import NoOpEvictionPolicy
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.api.client import SandboxOpError
from sandbox.domain.errors import GitPlatformNotConfigured, PullRequestFailed


class FakeGitPlatformProvider:
    """Records every PR request; returns a canned PullRequest."""

    kind = "fake-github"

    def __init__(self) -> None:
        self.calls: list[PullRequestRequest] = []
        self.next_result: PullRequest | None = None
        self.raise_for_next: BaseException | None = None

    async def create_pull_request(
        self, request: PullRequestRequest
    ) -> PullRequest:
        self.calls.append(request)
        if self.raise_for_next is not None:
            exc = self.raise_for_next
            self.raise_for_next = None
            raise exc
        if self.next_result is not None:
            result = self.next_result
            self.next_result = None
            return result
        return PullRequest(
            id=42,
            url="https://example.test/pulls/42",
            title=request.title,
            head_branch=request.head_branch,
            base_branch=request.base_branch,
            backend_kind=self.kind,
        )


def _build(tmp_path, *, git_platform=None) -> SandboxClient:
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    workspace_provider = LocalGitWorkspaceProvider(
        tmp_path / ".repos", repo_cache_provider=cache_provider
    )
    store = InMemorySandboxStore()
    locks = InMemoryLockManager()
    service = SandboxService(
        workspace_provider=workspace_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=store,
        locks=locks,
        repo_cache_provider=cache_provider,
        git_platform_provider=git_platform,
    )
    container = SandboxContainer(
        workspace_provider=workspace_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=store,
        locks=locks,
        service=service,
        eviction=NoOpEvictionPolicy(),
        repo_cache_provider=cache_provider,
        git_platform_provider=git_platform,
    )
    return SandboxClient.from_container(container)


def _writable_handle() -> WorkspaceHandle:
    return WorkspaceHandle(
        workspace_id="ws_w",
        branch="agent/edits-c1",
        backend_kind="local",
        local_path="/tmp/x",
        capabilities=Capabilities.from_mode(WorkspaceMode.EDIT),
    )


def _readonly_handle() -> WorkspaceHandle:
    return WorkspaceHandle(
        workspace_id="ws_r",
        branch="main",
        backend_kind="local",
        local_path="/tmp/x",
        capabilities=Capabilities.from_mode(WorkspaceMode.ANALYSIS),
    )


@pytest.mark.asyncio
async def test_create_pull_request_dispatches_to_provider(tmp_path) -> None:
    provider = FakeGitPlatformProvider()
    client = _build(tmp_path, git_platform=provider)

    pr = await client.create_pull_request(
        _writable_handle(),
        repo="owner/repo",
        title="Fix things",
        body="Some body",
        base_branch="main",
        reviewers=["alice"],
        labels=["agent"],
    )
    assert pr.id == 42
    assert pr.url == "https://example.test/pulls/42"
    assert len(provider.calls) == 1
    sent = provider.calls[0]
    assert sent.repo.repo_name == "owner/repo"
    assert sent.head_branch == "agent/edits-c1"  # from handle
    assert sent.base_branch == "main"
    assert sent.reviewers == ("alice",)
    assert sent.labels == ("agent",)


@pytest.mark.asyncio
async def test_create_pull_request_refuses_readonly_handle(tmp_path) -> None:
    provider = FakeGitPlatformProvider()
    client = _build(tmp_path, git_platform=provider)
    with pytest.raises(SandboxOpError, match="writable"):
        await client.create_pull_request(
            _readonly_handle(),
            repo="owner/repo",
            title="t",
            body="b",
            base_branch="main",
        )
    # No call leaked through to the provider.
    assert provider.calls == []


@pytest.mark.asyncio
async def test_create_pull_request_without_provider_raises(tmp_path) -> None:
    """Service raises a typed error when the platform port isn't wired."""
    client = _build(tmp_path, git_platform=None)
    with pytest.raises(GitPlatformNotConfigured):
        await client.create_pull_request(
            _writable_handle(),
            repo="owner/repo",
            title="t",
            body="b",
            base_branch="main",
        )


@pytest.mark.asyncio
async def test_pull_request_failed_propagates(tmp_path) -> None:
    """PullRequestFailed from the adapter surfaces unchanged."""
    provider = FakeGitPlatformProvider()
    provider.raise_for_next = PullRequestFailed("403 from forge")
    client = _build(tmp_path, git_platform=provider)
    with pytest.raises(PullRequestFailed, match="403"):
        await client.create_pull_request(
            _writable_handle(),
            repo="owner/repo",
            title="t",
            body="b",
            base_branch="main",
        )
