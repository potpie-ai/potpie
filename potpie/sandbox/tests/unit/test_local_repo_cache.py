"""Tests for LocalRepoCacheProvider and SandboxService.ensure_repo_cache.

Covers the doc's "Edit Flow step 2" (Ensure repo cache exists) — at the
adapter layer (LocalRepoCacheProvider creates the bare repo on disk and
fetches refs into it) and at the application layer (SandboxService
keys the cache by RepoCacheRequest.key() and persists it in the store).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.local.git_workspace import LocalGitWorkspaceProvider
from sandbox.adapters.outbound.local.repo_cache import LocalRepoCacheProvider
from sandbox.adapters.outbound.local.subprocess_runtime import (
    LocalSubprocessRuntimeProvider,
)
from sandbox.adapters.outbound.memory.locks import InMemoryLockManager
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.application.services.sandbox_service import SandboxService
from sandbox.domain.errors import RepoAuthFailed, RepoCacheUnavailable
from sandbox.domain.models import (
    RepoCacheRequest,
    RepoIdentity,
    WorkspaceMode,
    WorkspaceRequest,
    WorkspaceState,
)


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


def _make_request(source: Path, base_ref: str = "main") -> RepoCacheRequest:
    return RepoCacheRequest(
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref=base_ref,
        user_id="u1",
    )


@pytest.mark.asyncio
async def test_ensure_cache_clones_then_fetches(tmp_path: Path) -> None:
    """First call clones, second call hits the existing bare and fetches."""
    source = _make_repo(tmp_path)
    provider = LocalRepoCacheProvider(tmp_path / ".repos")

    cache = await provider.ensure_cache(_make_request(source))
    assert cache.state is WorkspaceState.READY
    assert cache.location.local_path is not None
    bare = Path(cache.location.local_path)
    assert (bare / "HEAD").exists(), "bare repo should be initialised"
    assert cache.key == "github.com|owner/repo"

    # Second call — bare exists already, just a fetch.
    cache2 = await provider.ensure_cache(_make_request(source))
    assert cache2.location.local_path == cache.location.local_path


@pytest.mark.asyncio
async def test_ensure_cache_fails_loudly_on_unreachable_repo(
    tmp_path: Path,
) -> None:
    """Bad repo URL surfaces as RepoCacheUnavailable, not a bare clone."""
    provider = LocalRepoCacheProvider(tmp_path / ".repos")
    request = RepoCacheRequest(
        repo=RepoIdentity(
            repo_name="owner/repo",
            repo_url=str(tmp_path / "does_not_exist"),
        ),
        base_ref="main",
        user_id="u1",
    )
    with pytest.raises(RepoCacheUnavailable):
        await provider.ensure_cache(request)


@pytest.mark.asyncio
async def test_workspace_provider_links_to_repo_cache(tmp_path: Path) -> None:
    """Workspace.repo_cache_id is populated when the workspace adapter
    composes with a LocalRepoCacheProvider."""
    source = _make_repo(tmp_path)
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    ws_provider = LocalGitWorkspaceProvider(
        tmp_path / ".repos", repo_cache_provider=cache_provider
    )
    workspace = await ws_provider.get_or_create_workspace(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    assert workspace.repo_cache_id is not None
    assert workspace.repo_cache_id.startswith("rc_")


@pytest.mark.asyncio
async def test_service_ensure_repo_cache_is_idempotent_on_key(
    tmp_path: Path,
) -> None:
    """Repeated ensure_repo_cache calls return the SAME persisted cache."""
    source = _make_repo(tmp_path)
    cache_provider = LocalRepoCacheProvider(tmp_path / ".repos")
    ws_provider = LocalGitWorkspaceProvider(
        tmp_path / ".repos", repo_cache_provider=cache_provider
    )
    store = InMemorySandboxStore()
    service = SandboxService(
        workspace_provider=ws_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=store,
        locks=InMemoryLockManager(),
        repo_cache_provider=cache_provider,
    )

    first = await service.ensure_repo_cache(_make_request(source))
    second = await service.ensure_repo_cache(_make_request(source))
    assert first.id == second.id, "service must dedupe by key"
    assert second.last_used_at >= first.last_used_at
    assert len(await store.list_repo_caches()) == 1


@pytest.mark.asyncio
async def test_service_ensure_repo_cache_requires_provider(
    tmp_path: Path,
) -> None:
    """Service without a cache provider raises a clear error."""
    source = _make_repo(tmp_path)
    ws_provider = LocalGitWorkspaceProvider(tmp_path / ".repos")
    service = SandboxService(
        workspace_provider=ws_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
    )
    with pytest.raises(RuntimeError, match="repo_cache_provider"):
        await service.ensure_repo_cache(_make_request(source))


@pytest.mark.asyncio
async def test_delete_cache_removes_bare_dir(tmp_path: Path) -> None:
    """delete_cache should drop the in-memory entry AND remove the bare dir."""
    source = _make_repo(tmp_path)
    provider = LocalRepoCacheProvider(tmp_path / ".repos")
    cache = await provider.ensure_cache(_make_request(source))
    bare = Path(cache.location.local_path)  # type: ignore[arg-type]
    assert bare.exists()

    await provider.delete_cache(cache)

    assert not bare.exists(), "bare repo dir must be removed on delete"
    assert await provider.get_cache(cache.id) is None
    # A fresh ensure call must mint a NEW id (the old one is gone).
    fresh = await provider.ensure_cache(_make_request(source))
    assert fresh.id != cache.id


@pytest.mark.asyncio
async def test_clone_does_not_persist_token_in_cached_origin(
    tmp_path: Path,
) -> None:
    """The bare repo's ``.git/config`` must not contain the auth token.

    Cache rows are keyed by ``(host, repo_name)`` only — multiple users
    share one on-disk bare. If the first user's token were persisted as
    the ``origin`` URL, a second user's fetch would silently piggy-back
    on it. This is the security regression flagged on the PR.
    """
    source = _make_repo(tmp_path)
    provider = LocalRepoCacheProvider(tmp_path / ".repos")
    request = RepoCacheRequest(
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        user_id="u1",
        auth_token="ghp_token_must_not_persist",  # noqa: S106 - test value
    )

    cache = await provider.ensure_cache(request)
    bare = Path(cache.location.local_path)  # type: ignore[arg-type]
    config_text = (bare / "config").read_text()

    assert "ghp_token_must_not_persist" not in config_text, (
        "auth token leaked into the bare repo's .git/config"
    )
    assert "x-access-token" not in config_text, (
        "tokenized origin URL leaked into the bare repo's .git/config"
    )
    # Sanity: the plain URL is what's persisted.
    assert str(source) in config_text


@pytest.mark.asyncio
async def test_fetch_uses_explicit_url_not_persisted_origin(
    tmp_path: Path,
) -> None:
    """A second ensure call (which goes through ``_fetch_ref``) must
    succeed even after the persisted origin has been scrubbed of auth.

    Guards against a regression where the scrub removes the token but
    the subsequent fetch still relies on ``git fetch origin`` and fails
    against a private remote.
    """
    source = _make_repo(tmp_path)
    provider = LocalRepoCacheProvider(tmp_path / ".repos")
    request = RepoCacheRequest(
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        user_id="u1",
        auth_token="ghp_first_user_token",  # noqa: S106 - test value
    )
    first = await provider.ensure_cache(request)

    # Simulate a different user driving the next fetch through the
    # exact same on-disk cache. Their token must be honored on the
    # fetch URL (not silently swapped for the persisted one).
    request_second = RepoCacheRequest(
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        user_id="u2",
        auth_token="ghp_second_user_token",  # noqa: S106 - test value
    )
    second = await provider.ensure_cache(request_second)
    assert second.id == first.id  # same on-disk bare → same cache id

    bare = Path(first.location.local_path)  # type: ignore[arg-type]
    config_text = (bare / "config").read_text()
    assert "ghp_first_user_token" not in config_text
    assert "ghp_second_user_token" not in config_text


def test_raise_git_error_classifies_auth_messages_as_repo_auth_failed() -> None:
    """The shared helper distinguishes auth/permission failures from
    generic infra failures so callers can pick the right typed error.

    We test the helper directly — driving real `git` to emit each
    auth-flavored stderr would require live remotes and credentials.
    """
    from sandbox.adapters.outbound.local._git_ops import raise_git_error

    auth_messages = [
        "fatal: Authentication failed for 'https://github.com/owner/repo.git/'",
        "ERROR: Permission denied (publickey).",
        "fatal: Could not read from remote repository.",
        "remote: Repository not found.",
    ]
    for stderr in auth_messages:
        with pytest.raises(RepoAuthFailed):
            raise_git_error("git fetch failed", stderr)

    # Non-auth failures still raise RepoCacheUnavailable.
    with pytest.raises(RepoCacheUnavailable):
        raise_git_error(
            "git fetch failed",
            "fatal: unable to access 'https://x/': Failed to connect",
        )


@pytest.mark.asyncio
async def test_fetch_path_uses_classification_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fetch path must route through ``raise_git_error`` so an auth
    failure on a refresh fetch surfaces as ``RepoAuthFailed``.

    Pre-fix, the fetch branch raised ``RepoCacheUnavailable`` for every
    failure regardless of root cause, masking expired-token errors as
    infra issues. We let the clone succeed (real local source) then
    monkey-patch the next ``run`` call (the fetch) to inject an
    auth-flavored stderr.
    """
    source = _make_repo(tmp_path)
    provider = LocalRepoCacheProvider(tmp_path / ".repos")
    request = RepoCacheRequest(
        repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
        base_ref="main",
        user_id="u1",
    )
    # First call clones cleanly.
    await provider.ensure_cache(request)

    # Patch `run` inside repo_cache to fail the next fetch with an
    # auth-flavored stderr. Other invocations (set-url, etc.) wouldn't
    # reach this branch — we re-enter via ``ensure_cache`` and the
    # bare already exists, so only ``_fetch_ref`` runs.
    from sandbox.adapters.outbound.local import repo_cache as repo_cache_mod

    real_run = repo_cache_mod.run

    def fail_fetch(cmd: list[str], **kw):
        if "fetch" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=128,
                stdout="",
                stderr="fatal: Authentication failed for 'https://...'\n",
            )
        return real_run(cmd, **kw)

    monkeypatch.setattr(repo_cache_mod, "run", fail_fetch)

    with pytest.raises(RepoAuthFailed):
        await provider.ensure_cache(request)


@pytest.mark.asyncio
async def test_acquire_session_without_cache_provider_skips_cache(
    tmp_path: Path,
) -> None:
    """If no RepoCacheProvider is wired, acquire_session must still create
    the workspace — just without a cache row.

    Daytona today doesn't expose the cache port; the service should
    silently fall back to plain workspace creation rather than 500ing.
    """
    source = _make_repo(tmp_path)
    ws_provider = LocalGitWorkspaceProvider(tmp_path / ".repos")
    service = SandboxService(
        workspace_provider=ws_provider,
        runtime_provider=LocalSubprocessRuntimeProvider(),
        store=InMemorySandboxStore(),
        locks=InMemoryLockManager(),
        # repo_cache_provider intentionally omitted
    )

    workspace = await service.acquire_session(
        WorkspaceRequest(
            user_id="u1",
            project_id="p1",
            repo=RepoIdentity(repo_name="owner/repo", repo_url=str(source)),
            base_ref="main",
            mode=WorkspaceMode.EDIT,
            conversation_id="c1",
        )
    )
    # Workspace exists; the cache id is None because the workspace
    # provider's auto-cache path was used (LocalGitWorkspaceProvider
    # constructs its own LocalRepoCacheProvider when the service
    # doesn't pass one in). The store must have NO RepoCache row,
    # though — that's the wiring the service is responsible for.
    assert workspace.id.startswith("ws_")
    assert len(await service._store.list_repo_caches()) == 0
