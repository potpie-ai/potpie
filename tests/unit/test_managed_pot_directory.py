"""``HostConnection.pots``: a managed host's pots for synchronous callers.

The directory runs ``host.*`` operations on its own event loop, so it is usable
from plain threads (an RPC servicer, a script) while engine clients run on the
caller's loop. Driven against the reference host through an ``httpx`` transport.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from potpie_context_engine.core.errors import CapabilityNotImplemented

from potpie.pots.directory import PotDirectory, is_remote_repo_key
from potpie.runtime import remote
from tests.support.reference_host import ALICE_KEY, CAROL_KEY, ReferenceHost

BASE = "https://your-host.example/graph"


@pytest.fixture
def reference(tmp_path: Path):
    host = ReferenceHost.build(tmp_path / "host")
    yield host
    remote._background.run(host.close())


def _connect(reference: ReferenceHost, key: str = ALICE_KEY, seen=None):
    return remote.connect(
        url=BASE, token=key, http_transport=reference.httpx_transport(seen=seen)
    )


def test_the_directory_is_a_pot_directory(reference) -> None:
    assert isinstance(_connect(reference).pots, PotDirectory)


def test_one_listing_answers_pots_the_active_pot_and_the_repo_default(
    reference,
) -> None:
    reference.state.active["alice"] = "pot_shop"
    reference.state.repo_defaults[("alice", "github.com/acme/shop")] = "pot_lab"
    seen: list = []
    connection = _connect(reference, seen=seen)
    pots = remote.ManagedPotDirectory(
        connection.pots.client, repo="github.com/acme/shop"
    )

    names = [pot.name for pot in pots.list_pots()]
    active = pots.active_pot()
    default = pots.repo_default(repo="github.com/acme/shop")

    assert names == ["shop", "lab"]
    assert active is not None and active.pot_id == "pot_shop"
    assert default == "pot_lab"
    assert len(seen) == 1


def test_selection_and_routing_round_trip(reference) -> None:
    pots = _connect(reference).pots

    used = pots.use_pot(ref="shop")
    pots.set_repo_default(repo="github.com/acme/shop", pot_id=used.pot_id)

    assert used.pot_id == "pot_shop" and used.active
    assert pots.active_pot().pot_id == "pot_shop"
    assert pots.repo_default(repo="github.com/acme/shop") == "pot_shop"
    assert pots.clear_repo_default(repo="github.com/acme/shop") is True
    assert pots.repo_default(repo="github.com/acme/shop") is None


def test_create_sources_whoami_and_grants(reference) -> None:
    pots = _connect(reference).pots

    created = pots.create_pot(name="docs", use=True)
    source = pots.add_source(
        pot_id=created.pot_id, kind="repo", location="github.com/acme/docs"
    )

    assert created.created is True
    assert [s.source_id for s in pots.list_sources(pot_id=created.pot_id)] == [
        source.source_id
    ]
    assert [row.pot_id for row in pots.list_repo_sources()] == [created.pot_id]
    assert pots.whoami().subject == "alice"
    granted = pots.grant(pot_id="pot_shop", actor_id="carol", role="reader")
    assert (granted.actor_id, granted.role) == ("carol", "reader")
    assert {g.actor_id for g in pots.list_grants(pot_id="pot_shop")} >= {"carol"}
    assert pots.revoke(pot_id="pot_shop", actor_id="carol") is True


def test_a_local_path_never_reaches_the_host(reference, tmp_path) -> None:
    seen: list = []
    pots = _connect(reference, seen=seen).pots

    assert pots.repo_default(repo=str(tmp_path)) is None
    with pytest.raises(ValueError, match="git remote"):
        pots.set_repo_default(repo=str(tmp_path), pot_id="pot_shop")
    assert seen == []
    assert not is_remote_repo_key(str(tmp_path))
    assert is_remote_repo_key("github.com/acme/shop")
    assert not is_remote_repo_key("C:\\work\\shop")


def test_a_host_refusal_is_raised_with_its_typed_error(reference) -> None:
    pots = _connect(reference, key=CAROL_KEY).pots

    with pytest.raises(remote.RemoteHostError) as raised:
        pots.use_pot(ref="pot_shop")

    assert raised.value.error.code == "pot_not_found"


def test_local_only_lifecycle_is_refused_by_name(reference) -> None:
    pots = _connect(reference).pots

    with pytest.raises(CapabilityNotImplemented, match="pot rename"):
        pots.rename_pot(ref="shop", new_name="store")
    with pytest.raises(CapabilityNotImplemented, match="source remove"):
        pots.remove_source(pot_id="pot_shop", source_id="src_1")


def test_the_directory_serves_callers_on_other_threads(reference) -> None:
    pots = _connect(reference).pots
    answers: list[list[str]] = []

    def _list() -> None:
        answers.append([pot.pot_id for pot in pots.list_pots()])

    threads = [threading.Thread(target=_list) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert answers == [["pot_shop", "pot_lab"]] * 4


def _ids(pots) -> list[str]:
    return [pot.pot_id for pot in pots]


def test_a_listing_is_served_from_the_cache_only_while_it_is_young(reference) -> None:
    now = [100.0]
    seen: list = []
    pots = remote.ManagedPotDirectory(
        _connect(reference, seen=seen).pots.client, clock=lambda: now[0]
    )
    other = _connect(reference).pots

    assert _ids(pots.list_pots()) == ["pot_shop", "pot_lab"]
    other.create_pot(name="docs")
    now[0] += remote.POT_LISTING_MAX_AGE_S - 0.1
    assert "pot_docs" not in _ids(pots.list_pots())
    assert len(seen) == 1

    # Once the listing is older than the window it is asked for again.
    now[0] += 0.1
    assert "pot_docs" in _ids(pots.list_pots())
    assert len(seen) == 2

    # Skipping the cache for one read, or dropping it.
    other.create_pot(name="notes")
    assert "pot_notes" in _ids(pots.listing(fresh=True).pots)
    other.create_pot(name="wiki")
    pots.invalidate()
    assert "pot_wiki" in _ids(pots.list_pots())
    assert len(seen) == 4


def test_a_zero_max_age_asks_the_host_on_every_read(reference) -> None:
    seen: list = []
    pots = remote.ManagedPotDirectory(
        _connect(reference, seen=seen).pots.client, max_age_s=0
    )

    pots.list_pots()
    pots.active_pot()

    assert len(seen) == 2
    with pytest.raises(ValueError, match="negative"):
        remote.ManagedPotDirectory(pots.client, max_age_s=-1)


def test_a_listing_in_flight_when_a_change_lands_is_not_kept(reference) -> None:
    seen: list = []
    connection = _connect(reference, seen=seen)
    holder: dict = {}

    def run_while_another_thread_changes_something(awaitable):
        answer = connection.run(awaitable)
        if len(seen) == 1:
            holder["pots"].invalidate()
        return answer

    pots = holder["pots"] = remote.ManagedPotDirectory(
        connection.pots.client, run=run_while_another_thread_changes_something
    )

    pots.list_pots()
    pots.list_pots()

    assert len(seen) == 2


def test_threads_may_read_and_change_one_directory_at_once(reference) -> None:
    pots = _connect(reference).pots
    errors: list[BaseException] = []
    answers: list[list[str]] = []

    def _reader() -> None:
        try:
            for _ in range(10):
                answers.append(_ids(pots.list_pots()))
                pots.active_pot()
                pots.repo_default(repo="github.com/acme/shop")
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    def _changer() -> None:
        try:
            for _ in range(10):
                pots.use_pot(ref="shop")
                pots.invalidate()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=_reader) for _ in range(6)]
    threads += [threading.Thread(target=_changer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    assert answers == [["pot_shop", "pot_lab"]] * 60
