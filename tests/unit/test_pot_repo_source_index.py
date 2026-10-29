"""The repo→pot index: every live pot's repo sources in one pot-service call.

Callers resolving "which pot owns this working tree" need the repo sources of
every pot. Asking pot by pot costs one read per pot, so the pot service answers
it once, from one load of its state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from potpie.pots.contracts import PotRepoSource
from potpie.pots.local_service import LocalPotManagementService
from potpie.pots.local_store import LocalPotStore
from potpie.runtime.root_services import PotResourceService

pytestmark = pytest.mark.unit


def _service(tmp_path) -> LocalPotManagementService:
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=SimpleNamespace(),
    )


def test_index_joins_every_repo_source_to_its_pot(tmp_path) -> None:
    pots = _service(tmp_path)
    shop = pots.create_pot(name="shop", use=True)
    fork = pots.create_pot(name="shop-fork")
    pots.add_source(pot_id=shop.pot_id, kind="repo", location="github.com/acme/shop")
    pots.add_source(pot_id=fork.pot_id, kind="repo", location="github.com/acme/shop")
    pots.add_source(
        pot_id=fork.pot_id, kind="repo", location="/srv/acme/ops", name="ops"
    )

    assert pots.list_repo_sources() == [
        PotRepoSource(
            pot_id=shop.pot_id,
            pot_name="shop",
            name="github.com/acme/shop",
            location="github.com/acme/shop",
        ),
        PotRepoSource(
            pot_id=fork.pot_id,
            pot_name="shop-fork",
            name="github.com/acme/shop",
            location="github.com/acme/shop",
        ),
        PotRepoSource(
            pot_id=fork.pot_id,
            pot_name="shop-fork",
            name="ops",
            location="/srv/acme/ops",
        ),
    ]


def test_index_carries_only_repo_sources(tmp_path) -> None:
    pots = _service(tmp_path)
    pot = pots.create_pot(name="shop", use=True)
    pots.add_source(pot_id=pot.pot_id, kind="github", location="acme/shop")
    pots.add_source(pot_id=pot.pot_id, kind="linear", location="ACME")

    assert pots.list_repo_sources() == []
    assert len(pots.list_sources(pot_id=pot.pot_id)) == 2


def test_index_agrees_with_the_per_pot_walk_it_replaces(tmp_path) -> None:
    pots = _service(tmp_path)
    for name, location in (("shop", "github.com/acme/shop"), ("ops", "/srv/ops")):
        pot = pots.create_pot(name=name)
        pots.add_source(pot_id=pot.pot_id, kind="repo", location=location)
    pots.create_pot(name="empty")

    walked = [
        (pot.pot_id, pot.name, source.name, source.location)
        for pot in pots.list_pots()
        for source in pots.list_sources(pot_id=pot.pot_id)
        if source.kind == "repo"
    ]

    assert [
        (row.pot_id, row.pot_name, row.name, row.location)
        for row in pots.list_repo_sources()
    ] == walked


def test_index_is_empty_before_any_pot_exists(tmp_path) -> None:
    assert _service(tmp_path).list_repo_sources() == []


def test_the_root_pot_service_serves_the_same_index(tmp_path) -> None:
    pots = _service(tmp_path)
    pot = pots.create_pot(name="shop", use=True)
    pots.add_source(pot_id=pot.pot_id, kind="repo", location="github.com/acme/shop")

    assert PotResourceService(pots).list_repo_sources() == pots.list_repo_sources()
