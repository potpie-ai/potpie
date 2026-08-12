"""The control-plane port must describe the implementation that answers it.

``PotManagementService`` is the type a hosted control plane is written against:
nobody types against ``LocalPotManagementService``, they type against the
Protocol and trust it. So a port that lies is not a cosmetic problem — it is the
specification a second implementation will faithfully reproduce.

It lied about exactly the method where the honest answer had just been won.
``remove_source`` was declared ``-> None`` while the service, the store and the
CLI had all moved to a boolean ("did a row actually go away"), which is what
stops ``source remove`` from reporting a registration gone when it is still
alive in the pot the caller did not pass. A hosted implementation reading the
port would have returned ``None`` and put that misreport straight back.

Two things are pinned here, because either alone is weak: the signatures agree
(cheap, total, catches the next drift wherever it lands), and the boolean the
signature promises is actually produced by the real service over the real store
— fakes are exactly what let the previous version of this contract pass.
"""

from __future__ import annotations

import inspect

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.domain.ports.services.pot_management import (
    PotManagementService,
)

pytestmark = pytest.mark.unit


def _port_methods() -> list[str]:
    return [
        name
        for name, _ in inspect.getmembers(PotManagementService, inspect.isfunction)
        if not name.startswith("_")
    ]


def test_the_local_service_implements_every_declared_method() -> None:
    missing = [
        name for name in _port_methods() if not hasattr(LocalPotManagementService, name)
    ]

    assert missing == [], (
        f"{missing} are declared on PotManagementService but the shipped "
        "implementation does not answer them"
    )


@pytest.mark.parametrize("name", _port_methods())
def test_the_port_signature_matches_the_shipped_implementation(name: str) -> None:
    """Including the return annotation, which is the half that drifted.

    ``from __future__ import annotations`` is on in both modules, so these are
    string comparisons — ``'None'`` against ``'bool'`` for the defect this was
    written for.
    """
    declared = inspect.signature(getattr(PotManagementService, name))
    implemented = inspect.signature(getattr(LocalPotManagementService, name))

    assert declared == implemented, (
        f"PotManagementService.{name} declares {declared} but "
        f"LocalPotManagementService.{name} is {implemented}; a hosted "
        "implementation is written against the port, so fix whichever one is "
        "lying"
    )


def _service(tmp_path) -> LocalPotManagementService:
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=InMemoryGraphBackend(),
    )


def test_remove_source_answers_the_boolean_the_port_declares(tmp_path) -> None:
    """The behaviour behind the annotation, over the real flat-file store.

    A miss must be distinguishable from a removal: ``source remove`` turns the
    ``False`` into "no source '<id>' in pot '<pot>'" rather than congratulating
    the user on deleting something that is still registered elsewhere.
    """
    pots = _service(tmp_path)
    pot = pots.create_pot(name="port-contract", use=True)
    source = pots.add_source(
        pot_id=pot.pot_id, kind="repo", location="github.com/acme/x"
    )

    assert pots.remove_source(pot_id=pot.pot_id, source_id=source.source_id) is True
    assert pots.remove_source(pot_id=pot.pot_id, source_id=source.source_id) is False
    assert pots.remove_source(pot_id=pot.pot_id, source_id="never-registered") is False
