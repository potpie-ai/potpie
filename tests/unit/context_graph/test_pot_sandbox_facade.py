"""PotSandboxFacade — acquire/release lifecycle + ambiguity errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from adapters.outbound.agent_tools.sandbox import (
    PotSandboxConfig,
    PotSandboxFacade,
    RepoAttachment,
    _AmbiguousRepoError,
    _UnknownRepoError,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeHandle:
    workspace_id: str


class _FakeClient:
    """Records ``acquire_session`` calls and hands back deterministic handles."""

    def __init__(self) -> None:
        self.acquire_calls: list[dict[str, Any]] = []
        self.release_calls: list[_FakeHandle] = []
        self.exec_calls: list[tuple[_FakeHandle, list[str]]] = []
        self.exec_responses: list[Any] = []

    async def acquire_session(self, **kwargs: Any) -> _FakeHandle:
        self.acquire_calls.append(kwargs)
        return _FakeHandle(workspace_id=f"ws-{kwargs['repo']}")

    async def release_session(
        self, handle: _FakeHandle, *, destroy_runtime: bool = False
    ) -> None:
        self.release_calls.append(handle)

    async def exec(self, handle, cmd, **_kwargs):
        self.exec_calls.append((handle, list(cmd)))
        if self.exec_responses:
            return self.exec_responses.pop(0)
        return _FakeExec(exit_code=0, stdout=b"", stderr=b"")


@dataclass
class _FakeExec:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""


def _cfg(*repos: tuple[str, str]) -> PotSandboxConfig:
    return PotSandboxConfig(
        user_id="u1",
        pot_id="pot-1",
        provider_host="github.com",
        repos=[
            RepoAttachment(
                owner=owner,
                repo=repo,
                default_branch="main",
                repo_url=f"https://github.com/{owner}/{repo}",
                auth_token=f"tok-{owner}",
            )
            for owner, repo in repos
        ],
    )


class TestResolveRepo:
    def test_single_repo_default(self) -> None:
        facade = PotSandboxFacade(client=_FakeClient(), cfg=_cfg(("a", "x")))
        attachment = facade.resolve_repo(None)
        assert attachment.full_name == "a/x"

    def test_explicit_repo_match(self) -> None:
        facade = PotSandboxFacade(
            client=_FakeClient(), cfg=_cfg(("a", "x"), ("b", "y"))
        )
        assert facade.resolve_repo("b/y").full_name == "b/y"

    def test_ambiguous_when_multi_repo_and_no_arg(self) -> None:
        facade = PotSandboxFacade(
            client=_FakeClient(), cfg=_cfg(("a", "x"), ("b", "y"))
        )
        with pytest.raises(_AmbiguousRepoError):
            facade.resolve_repo(None)

    def test_unknown_repo_raises(self) -> None:
        facade = PotSandboxFacade(
            client=_FakeClient(), cfg=_cfg(("a", "x"), ("b", "y"))
        )
        with pytest.raises(_UnknownRepoError):
            facade.resolve_repo("c/z")

    def test_empty_pot_unknown_repo(self) -> None:
        facade = PotSandboxFacade(client=_FakeClient(), cfg=_cfg())
        with pytest.raises(_UnknownRepoError):
            facade.resolve_repo(None)


@pytest.mark.asyncio
class TestAcquireAndRelease:
    async def test_acquire_caches_handle_per_repo(self) -> None:
        client = _FakeClient()
        facade = PotSandboxFacade(
            client=client, cfg=_cfg(("a", "x"), ("b", "y"))
        )
        a1 = await facade.acquire("a/x")
        a2 = await facade.acquire("a/x")
        # Same handle returned for repeat calls — only one acquire.
        assert a1[1] is a2[1]
        assert len(client.acquire_calls) == 1
        # Different repo gets its own workspace.
        await facade.acquire("b/y")
        assert len(client.acquire_calls) == 2
        repos_acquired = {c["repo"] for c in client.acquire_calls}
        assert repos_acquired == {"a/x", "b/y"}

    async def test_acquire_passes_attachment_metadata(self) -> None:
        client = _FakeClient()
        facade = PotSandboxFacade(client=client, cfg=_cfg(("a", "x")))
        await facade.acquire(None)
        call = client.acquire_calls[0]
        assert call["user_id"] == "u1"
        # project_id slot in SandboxClient gets the pot_id.
        assert call["project_id"] == "pot-1"
        assert call["repo"] == "a/x"
        assert call["branch"] == "main"
        assert call["auth_token"] == "tok-a"
        assert call["repo_url"] == "https://github.com/a/x"

    async def test_release_all_hibernates_every_workspace(self) -> None:
        client = _FakeClient()
        facade = PotSandboxFacade(
            client=client, cfg=_cfg(("a", "x"), ("b", "y"))
        )
        await facade.acquire("a/x")
        await facade.acquire("b/y")
        await facade.release_all()
        assert len(client.release_calls) == 2
        # Calling again is a no-op (state was cleared).
        await facade.release_all()
        assert len(client.release_calls) == 2

    async def test_release_swallows_per_repo_errors(self) -> None:
        client = _FakeClient()

        async def _boom(handle, *, destroy_runtime: bool = False):
            raise RuntimeError("backend dead")

        client.release_session = _boom  # type: ignore[assignment]
        facade = PotSandboxFacade(client=client, cfg=_cfg(("a", "x")))
        await facade.acquire(None)
        # Must not raise.
        await facade.release_all()

    async def test_repo_lock_is_per_repo(self) -> None:
        facade = PotSandboxFacade(
            client=_FakeClient(), cfg=_cfg(("a", "x"), ("b", "y"))
        )
        l1 = facade.repo_lock("a/x")
        l2 = facade.repo_lock("a/x")
        l3 = facade.repo_lock("b/y")
        assert l1 is l2  # same lock object on repeat lookup
        assert l1 is not l3
