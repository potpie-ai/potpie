"""Cross-backend mutation reliability; selected native profiles cannot skip."""

import pytest

from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from tests.graph_reliability import (
    assertion_batch,
    isolated_backend,
    reliability_profiles,
)


@pytest.fixture(params=reliability_profiles())
def reliability_backend(request, tmp_path, monkeypatch):
    with isolated_backend(request.param, tmp_path, monkeypatch) as result:
        yield result


def test_completed_execution_replays_without_revision_change(reliability_backend):
    pot, backend = reliability_backend
    batch = assertion_batch()
    context = ProvenanceContext(mutation_id="reliability:once")
    first = backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=context
    )
    version = backend.mutation.current_version(pot)
    second = backend.mutation.apply(
        batch, expected_pot_id=pot, provenance_context=context
    )
    assert first.ok and first == second
    assert backend.mutation.current_version(pot) == version == 1
    assert len(backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot))) == 1


def test_stale_revision_does_not_apply_new_execution(reliability_backend):
    from potpie_context_engine.core.errors import GraphMutationVersionConflict

    pot, backend = reliability_backend
    batch = assertion_batch()
    backend.mutation.compare_and_apply(
        batch,
        expected_pot_id=pot,
        expected_version=0,
        provenance_context=ProvenanceContext(mutation_id="reliability:winner"),
    )
    with pytest.raises(GraphMutationVersionConflict):
        backend.mutation.compare_and_apply(
            batch,
            expected_pot_id=pot,
            expected_version=0,
            provenance_context=ProvenanceContext(mutation_id="reliability:stale"),
        )
    assert backend.mutation.current_version(pot) == 1
