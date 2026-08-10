"""The repo→pot index: every pot's repo sources in one control-plane call.

Callers resolving "which pot owns this working tree" need the repo sources of
every visible pot. Asking pot by pot is a dict read here and a round trip per
pot against a hosted control plane, so the control plane answers it once.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.domain.ports.services.pot_management import PotRepoSource

pytestmark = pytest.mark.unit


def _service(tmp_path) -> LocalPotManagementService:
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=InMemoryGraphBackend(),
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
