"""Administrative entry points reject unsupported or malformed execution."""

from types import SimpleNamespace

import pytest

from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_atomic import apply_atomic
from potpie_context_engine.core.definition import DEFAULT_GRAPH_DEFINITION
from potpie_context_engine.core.errors import MutationBatchValidationError
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.reconciliation import MutationBatch


def test_injected_writer_does_not_advertise_unsupported_administrative_writes():
    backend = Neo4jGraphBackend(SimpleNamespace(), writer=SimpleNamespace(enabled=True))
    assert not backend.capabilities().snapshot
    assert not backend.mutation.readiness("test").capability_ready["snapshot"]
    with pytest.raises(NotImplementedError, match="snapshot import requires native"):
        backend.snapshot.import_data(pot_id="test", payload={})
    with pytest.raises(NotImplementedError, match="entity repair requires native"):
        backend.analytics.repair("test", targets=("entity_summaries",))
    with pytest.raises(NotImplementedError, match="reset requires native"):
        backend.mutation.reset_pot("test", mutation_id="reset:unsupported")


async def test_mutated_administrative_batch_fails_before_opening_driver():
    def unexpected_driver():
        pytest.fail("malformed administrative intent must fail before touching Neo4j")

    batch = MutationBatch(reset_pot=True)
    batch.entity_upserts.append(EntityUpsert("service:api", ("Service",)))
    with pytest.raises(MutationBatchValidationError, match="cannot share a batch"):
        await apply_atomic(
            SimpleNamespace(_new_driver=unexpected_driver),
            batch,
            expected_pot_id="test",
            provenance_context=ProvenanceContext(mutation_id="reserved"),
            definition=DEFAULT_GRAPH_DEFINITION,
        )
