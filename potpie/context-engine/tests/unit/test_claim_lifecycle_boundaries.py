"""Supported writes must not bypass preparation or corrupt lifecycle identity."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.core.claim_lifecycle import ClaimLifecycleConflict
from potpie_context_engine.core.errors import MutationBatchValidationError
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.reconciliation import MutationBatch


@pytest.fixture(params=["memory", "embedded"])
def backend(request, tmp_path):
    return (
        InMemoryGraphBackend()
        if request.param == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )


def batch():
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:a", ("Service",)),
            EntityUpsert("service:b", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:a",
                "service:b",
                {
                    "source_system": "inventory",
                    "source_ref": "file:a",
                    "fact": "original",
                    "valid_from": "2025-01-01T00:00:00+00:00",
                },
            )
        ],
    )


def test_raw_write_cannot_supply_an_untrusted_frozen_resolution(backend):
    plan = batch()
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id="forged",
        anchor=datetime.now(timezone.utc),
        query=backend.claim_query,
        definition=backend.definition,
    )
    with pytest.raises(MutationExecutionReuseError, match="version-2 prepared"):
        backend.mutation.apply(plan, expected_pot_id="p")
    assert backend.mutation.current_version("p") == 0


def test_ordinary_assertion_cannot_retire_an_incarnation(backend):
    plan = batch()
    assert backend.mutation.apply(plan, expected_pot_id="p").ok
    plan.edge_upserts[0].properties.update(
        retired=True, valid_from="2025-03-01T00:00:00+00:00"
    )
    with pytest.raises(ClaimLifecycleConflict, match="cannot retire"):
        backend.mutation.apply(plan, expected_pot_id="p")
    assert backend.mutation.current_version("p") == 1


@pytest.mark.parametrize(
    "name", ["valid_at", "valid_from", "invalid_at", "valid_until", "observed_at"]
)
def test_malformed_assertion_time_never_becomes_an_open_interval(backend, name):
    plan = batch()
    plan.edge_upserts[0].properties[name] = "not-a-time"
    with pytest.raises(
        (ClaimLifecycleConflict, MutationBatchValidationError), match="timestamp"
    ):
        backend.mutation.apply(plan, expected_pot_id="p")
    assert backend.mutation.current_version("p") == 0


def test_schema1_without_receipt_requires_fresh_preparation(backend):
    plan = batch()
    context = ProvenanceContext(
        mutation_id="legacy",
        prepared=PreparedMutation(
            "p",
            "legacy",
            "request",
            batch_fingerprint(plan),
            0,
            datetime.now(timezone.utc),
            schema_version=1,
        ),
    )
    with pytest.raises(MutationExecutionReuseError, match="migration_required"):
        backend.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    assert backend.mutation.current_version("p") == 0


def test_schema1_receipt_lookup_precedes_migration_rejection(backend):
    plan = MutationBatch(entity_upserts=[EntityUpsert("service:a", ("Service",))])
    prepared = PreparedMutation(
        "p",
        "recorded",
        "request",
        batch_fingerprint(plan),
        0,
        datetime.now(timezone.utc),
    )
    context = ProvenanceContext(mutation_id="recorded", prepared=prepared)
    original = backend.mutation.apply(
        plan, expected_pot_id="p", provenance_context=context
    )
    replay = backend.mutation.apply(
        plan,
        expected_pot_id="p",
        provenance_context=replace(
            context, prepared=replace(prepared, schema_version=1)
        ),
    )
    assert replay == original
    assert backend.mutation.current_version("p") == 1
