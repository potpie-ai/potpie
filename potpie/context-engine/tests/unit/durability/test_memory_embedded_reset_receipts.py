"""Reset convenience methods expose durable replayable execution identity."""

import asyncio

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch

pytestmark = pytest.mark.unit
POT = "reset-receipt-pot"


@pytest.fixture(params=["memory", "embedded"])
def backend(request, tmp_path):
    return (
        InMemoryGraphBackend()
        if request.param == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )


def _plan(prefix="before"):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert(f"service:{prefix}-a", ("Service",)),
            EntityUpsert(f"service:{prefix}-b", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert("RELATED_TO", f"service:{prefix}-a", f"service:{prefix}-b", {})
        ],
    )


def _reset(backend, mode, mutation_id=None):
    if mode == "sync":
        return backend.mutation.reset_pot(POT, mutation_id=mutation_id)
    return asyncio.run(backend.mutation.reset_pot_async(POT, mutation_id=mutation_id))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_reset_wrapper_replays_original_receipt_without_changing_current_graph(
    backend, mode
):
    plan = _plan()
    before = backend.mutation.apply(
        plan,
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="write:before"),
    )
    first = _reset(backend, mode, "reset:stable")
    assert first["removed_claims"] == 1
    assert first["mutation_id"] == "reset:stable"
    assert first["execution"]["version_before"] == 1
    assert first["execution"]["version_after"] == 2
    assert first["execution"]["applied_at"]
    lookup = backend.mutation.lookup_execution(
        MutationBatch(reset_pot=True),
        expected_pot_id=POT,
        mutation_id=first["mutation_id"],
    )
    assert lookup.state == "completed"
    assert lookup.result.execution.to_dict() == first["execution"]

    if isinstance(backend, EmbeddedGraphBackend):
        backend = EmbeddedGraphBackend(home=backend.home)
    assert _reset(backend, mode, "reset:stable") == first
    assert backend.mutation.current_version(POT) == 2
    assert (
        backend.mutation.apply(
            plan,
            expected_pot_id=POT,
            provenance_context=ProvenanceContext(mutation_id="write:before"),
        )
        == before
    )
    assert backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT)) == []
    backend.mutation.apply(_plan("after"), expected_pot_id=POT)
    assert backend.mutation.current_version(POT) == 3
    assert _reset(backend, mode, "reset:stable") == first
    rows = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))
    assert [row.subject_key for row in rows] == ["service:after-a"]
    assert backend.mutation.current_version(POT) == 3


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_generated_reset_identity_is_returned_and_can_be_replayed(backend, mode):
    first = _reset(backend, mode)
    second = _reset(backend, mode)
    assert first["removed_claims"] == second["removed_claims"] == 0
    assert first["mutation_id"] and second["mutation_id"]
    assert first["mutation_id"] != second["mutation_id"]
    assert first["execution"]["version_before"] == 0
    assert first["execution"]["version_after"] == 1
    assert second["execution"]["version_before"] == 1
    assert second["execution"]["version_after"] == 2
    assert _reset(backend, mode, first["mutation_id"]) == first
    assert backend.mutation.current_version(POT) == 2


def test_reset_cannot_rebind_identity_used_by_a_different_batch(backend):
    backend.mutation.apply(
        _plan(),
        expected_pot_id=POT,
        provenance_context=ProvenanceContext(mutation_id="already-bound"),
    )
    with pytest.raises(MutationExecutionReuseError):
        backend.mutation.reset_pot(POT, mutation_id="already-bound")
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=POT))) == 1
    assert backend.mutation.current_version(POT) == 1
