"""Store-level tests for RepoCache persistence.

Covers both InMemorySandboxStore and JsonSandboxStore: save/get/find/
delete/list and round-trip persistence across a fresh store instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from sandbox.adapters.outbound.file.json_store import JsonSandboxStore
from sandbox.adapters.outbound.memory.store import InMemorySandboxStore
from sandbox.domain.models import (
    RepoCache,
    RepoCacheRequest,
    RepoIdentity,
    WorkspaceLocation,
    WorkspaceState,
    WorkspaceStorageKind,
    new_id,
)


def _make_cache(*, key: str = "github.com|owner/repo") -> RepoCache:
    return RepoCache(
        id=new_id("rc"),
        key=key,
        repo=RepoIdentity(repo_name="owner/repo"),
        location=WorkspaceLocation(
            kind=WorkspaceStorageKind.LOCAL_PATH,
            local_path="/tmp/.repos/owner/repo/.bare",
        ),
        backend_kind="local",
    )


def test_repo_cache_request_key_is_host_and_repo() -> None:
    request = RepoCacheRequest(
        repo=RepoIdentity(repo_name="owner/repo"),
        base_ref="main",
    )
    assert request.key() == "github.com|owner/repo"


@pytest.mark.asyncio
async def test_in_memory_store_round_trips_repo_cache() -> None:
    store = InMemorySandboxStore()
    cache = _make_cache()

    assert await store.get_repo_cache(cache.id) is None
    assert await store.find_repo_cache_by_key(cache.key) is None

    await store.save_repo_cache(cache)

    fetched = await store.get_repo_cache(cache.id)
    assert fetched is cache
    by_key = await store.find_repo_cache_by_key(cache.key)
    assert by_key is cache
    assert await store.list_repo_caches() == [cache]

    await store.delete_repo_cache(cache.id)
    assert await store.get_repo_cache(cache.id) is None
    assert await store.find_repo_cache_by_key(cache.key) is None
    assert await store.list_repo_caches() == []


@pytest.mark.asyncio
async def test_json_store_persists_repo_cache_across_instances(
    tmp_path: Path,
) -> None:
    """Save in one store, load in a fresh one — survives restart."""
    path = tmp_path / "metadata.json"
    cache = _make_cache()

    saver = JsonSandboxStore(path)
    await saver.save_repo_cache(cache)

    # Fresh instance hits the file load path.
    reloader = JsonSandboxStore(path)
    rehydrated = await reloader.find_repo_cache_by_key(cache.key)
    assert rehydrated is not None
    assert rehydrated.id == cache.id
    assert rehydrated.key == cache.key
    assert rehydrated.location.local_path == cache.location.local_path
    assert rehydrated.backend_kind == "local"
    assert rehydrated.state is WorkspaceState.READY


@pytest.mark.asyncio
async def test_json_store_keys_are_independent_per_repo(
    tmp_path: Path,
) -> None:
    store = JsonSandboxStore(tmp_path / "metadata.json")
    a = _make_cache(key="github.com|alice/repo")
    b = _make_cache(key="github.com|bob/repo")

    await store.save_repo_cache(a)
    await store.save_repo_cache(b)

    assert (await store.find_repo_cache_by_key(a.key)) is not None
    assert (await store.find_repo_cache_by_key(b.key)) is not None
    assert len(await store.list_repo_caches()) == 2

    await store.delete_repo_cache(a.id)
    assert await store.find_repo_cache_by_key(a.key) is None
    assert await store.find_repo_cache_by_key(b.key) is not None
