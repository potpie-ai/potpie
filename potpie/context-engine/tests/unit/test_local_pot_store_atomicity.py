from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.domain.errors import PotNotFound


def test_register_repo_source_is_atomic_and_idempotent(tmp_path) -> None:
    store = LocalPotStore(tmp_path)
    pot = store.create(name="default", use=True)

    first = store.register_repo_source(
        pot_id=pot["pot_id"],
        location="https://github.com/acme/shop.git",
        name="shop",
    )
    second = store.register_repo_source(
        pot_id=pot["pot_id"],
        location="git@github.com:acme/shop.git",
        name="renamed",
    )

    assert first[2:] == (True, True)
    assert second[2:] == (False, True)
    assert second[0]["source_id"] == first[0]["source_id"]
    assert second[0]["name"] == "shop"
    assert len(store.list_sources(pot_id=pot["pot_id"])) == 1
    assert store.repo_default(repo="github.com/acme/shop") == pot["pot_id"]


def test_register_repo_source_without_default_does_not_bind(tmp_path) -> None:
    store = LocalPotStore(tmp_path)
    pot = store.create(name="default")

    result = store.register_repo_source(
        pot_id=pot["pot_id"], location="github.com/acme/shop", make_default=False
    )

    assert result[2:] == (True, False)
    assert store.repo_default(repo="github.com/acme/shop") is None


def test_invalid_registration_leaves_file_unchanged(tmp_path) -> None:
    store = LocalPotStore(tmp_path)
    store.create(name="default")
    before = store._path.read_bytes()

    with pytest.raises(PotNotFound):
        store.register_repo_source(pot_id="missing", location="github.com/acme/shop")

    assert store._path.read_bytes() == before


def test_replace_failure_preserves_original_and_cleans_temp_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalPotStore(tmp_path)
    store.create(name="default")
    before = store._path.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        store.create(name="second")

    assert store._path.read_bytes() == before
    assert list(tmp_path.glob("pots.*.tmp")) == []


def test_concurrent_stores_do_not_duplicate_repo_source_or_lose_updates(
    tmp_path,
) -> None:
    initial = LocalPotStore(tmp_path)
    pot = initial.create(name="default")

    def register(_index: int) -> tuple[dict, str, bool, bool]:
        return LocalPotStore(tmp_path).register_repo_source(
            pot_id=pot["pot_id"], location="github.com/acme/shop"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(register, range(24)))

    state = json.loads(initial._path.read_text())
    assert len(state["sources"][pot["pot_id"]]) == 1
    assert sum(result[2] for result in results) == 1
    assert state["repo_defaults"]["github.com/acme/shop"] == pot["pot_id"]


def test_all_mutations_share_the_locked_update_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalPotStore(tmp_path)
    calls = 0
    original = LocalPotStore._update

    def observed(instance, operation):
        nonlocal calls
        calls += 1
        return original(instance, operation)

    monkeypatch.setattr(LocalPotStore, "_update", observed)
    pot = store.create(name="default")
    store.use(ref=pot["pot_id"])
    store.rename(ref=pot["pot_id"], new_name="renamed")
    source = store.add_source(pot_id=pot["pot_id"], kind="docs", location="README.md")
    store.remove_source(pot_id=pot["pot_id"], source_id=source["source_id"])
    store.set_repo_default(repo="github.com/acme/shop", pot_id=pot["pot_id"])
    store.clear_repo_default(repo="github.com/acme/shop")
    store.archive(ref=pot["pot_id"])

    assert calls == 8
