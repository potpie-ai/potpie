"""Reusable public conformance checks for third-party graph backends."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

from potpie_context_core.api import (
    ClaimQueryFilter,
    DEFAULT_GRAPH_DEFINITION,
    EdgeUpsert,
    EntityUpsert,
    MutationBatch,
)


def run_graph_backend_conformance(backend_factory: Callable[[], object]) -> None:
    """Exercise canonical writes, idempotency, invalidation, and pot isolation."""

    backend = backend_factory().bind_definition(DEFAULT_GRAPH_DEFINITION)
    caps = backend.capabilities()
    if not caps.mutation or not caps.claim_query:
        raise AssertionError("mutation and claim_query capabilities are mandatory")

    pot_a = "conformance:pot-a"
    pot_b = "conformance:pot-b"
    claim_key = "claim:conformance:one"
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert(entity_key="service:alpha", labels=("Service",)),
            EntityUpsert(entity_key="service:beta", labels=("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="RELATED_TO",
                from_entity_key="service:alpha",
                to_entity_key="service:beta",
                properties={
                    "claim_key": claim_key,
                    "truth": "agent_claim",
                    "subgraph": "admin",
                    "fact": "alpha relates to beta",
                },
            )
        ],
    )
    first = backend.mutation.apply(batch, expected_pot_id=pot_a)
    second = backend.mutation.apply(batch, expected_pot_id=pot_a)
    if not first.ok or not second.ok:
        raise AssertionError("canonical mutation failed")

    rows_a = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_a))
    rows_b = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_b))
    if len([row for row in rows_a if row.claim_key == claim_key]) != 1:
        raise AssertionError("claim-key idempotency is not enforced")
    if rows_b:
        raise AssertionError("one pot can read another pot's claims")

    invalidated = backend.mutation.invalidate(
        pot_id=pot_a, claim_keys=(claim_key,), reason="conformance"
    )
    if invalidated != 1:
        raise AssertionError("claim invalidation did not affect exactly one claim")
    if backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_a)):
        raise AssertionError("invalidated claims are visible by default")
    invalidated_rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=pot_a, include_invalidated=True)
    )
    if len(invalidated_rows) != 1:
        raise AssertionError("invalidated claim history is not queryable")

    # A reset and every optional projection must stay partition-scoped.
    backend.mutation.apply(batch, expected_pot_id=pot_b)
    backend.mutation.reset_pot(pot_a)
    if not backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot_b)):
        raise AssertionError("resetting one pot changed another pot")

    if caps.inspection:
        foreign = backend.inspection.slice(
            pot_id=pot_a, filter_=ClaimQueryFilter(pot_id=pot_a)
        )
        if foreign.nodes or foreign.edges:
            raise AssertionError("inspection leaked reset pot state")
    if caps.analytics:
        counts_a = backend.analytics.counts(pot_a)
        counts_b = backend.analytics.counts(pot_b)
        if counts_a.get("claims", 0) != 0 or counts_b.get("claims", 0) < 1:
            raise AssertionError("analytics counts are not pot-isolated")
    if caps.snapshot:
        with TemporaryDirectory() as directory:
            destination = str(Path(directory) / "pot-b.json")
            manifest = backend.snapshot.export(pot_id=pot_b, destination=destination)
            if manifest.pot_id != pot_b or manifest.claim_count < 1:
                raise AssertionError("snapshot export did not preserve pot metadata")
            backend.snapshot.import_(pot_id="conformance:pot-c", source=destination)
            rows_c = backend.claim_query.find_claims(
                ClaimQueryFilter(pot_id="conformance:pot-c")
            )
            if not rows_c or any(row.pot_id != "conformance:pot-c" for row in rows_c):
                raise AssertionError("snapshot import is not target-pot isolated")


class GraphBackendConformanceMixin:
    """Pytest-compatible mixin for backend adapter test suites."""

    backend_factory: Callable[[], object]

    def test_graph_backend_conformance(self) -> None:
        run_graph_backend_conformance(self.backend_factory)


__all__ = [
    "GraphBackendConformanceMixin",
    "run_graph_backend_conformance",
]
