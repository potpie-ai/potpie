"""Writer reads retain touched history while excluding unrelated pot history."""

import asyncio
from copy import deepcopy
import os

import pytest

from potpie_context_engine.core.graph_mutations import InvalidationOp, ProvenanceContext
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch
from tests.graph_reliability import assertion_batch, isolated_backend


@pytest.mark.integration
@pytest.mark.parametrize(
    "profile",
    [
        "falkordb_lite",
        pytest.param(
            "falkordb",
            marks=pytest.mark.skipif(
                not os.getenv("PROTOCOL_TEST_FALKORDB_URL"),
                reason="disposable FalkorDB server required",
            ),
        ),
        pytest.param(
            "neo4j",
            marks=pytest.mark.skipif(
                not os.getenv("PROTOCOL_TEST_NEO4J_URI"),
                reason="disposable Neo4j server required",
            ),
        ),
    ],
)
def test_scoped_writer_reads_preserve_closed_incarnations(
    profile, tmp_path, monkeypatch
):
    with isolated_backend(profile, tmp_path, monkeypatch) as (pot, backend):
        batch = assertion_batch()
        backend.mutation.apply(batch, expected_pot_id=pot)
        first = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot))[0]
        backend.mutation.apply(
            MutationBatch(
                invalidations=[
                    InvalidationOp(
                        None,
                        None,
                        "closed for reassertion",
                        valid_to="2026-02-01T00:00:00+00:00",
                        target_claim_record_ids=(first.record_id,),
                    )
                ]
            ),
            expected_pot_id=pot,
        )
        # The normal write path must see the closed incarnation before it can
        # prepare a distinct March incarnation.
        backend.mutation.apply(
            assertion_batch(valid_at="2026-03-01T00:00:00+00:00"), expected_pot_id=pot
        )
        unrelated = deepcopy(batch)
        unrelated.entity_upserts[0].entity_key = "service:unrelated"
        unrelated.edge_upserts[0].from_entity_key = "service:unrelated"
        unrelated.edge_upserts[0].properties["claim_key"] = "claim:unrelated"
        backend.mutation.apply(unrelated, expected_pot_id=pot)
        writer = (
            backend.mutation._get_writer() if profile == "neo4j" else backend.writer
        )

        async def check():
            assert await writer.claim_rows(pot) == []
            rows = await writer.claim_rows(
                pot, history_scopes=(("service:api", "DEPENDS_ON"),)
            )
            assert len(rows) == 2
            assert {row.subject_key for row in rows} == {"service:api"}
            assert {row.invalid_at is not None for row in rows} == {False, True}
            selected = await writer.claim_rows(pot, record_ids=(first.record_id,))
            assert [row.record_id for row in selected] == [first.record_id]
            assert selected[0].invalid_at is not None
            assert (
                await writer.claim_rows("other-pot", record_ids=(first.record_id,))
                == []
            )
            assert (
                await writer.claim_rows(
                    pot,
                    corrections=(
                        InvalidationOp(
                            "service:unrelated",
                            None,
                            "wrong scope",
                            target_claim_record_ids=(first.record_id,),
                        ),
                    ),
                )
                == []
            )
            return rows

        rows = asyncio.run(check())
        # Native application still resolves exact corrections over the full
        # selected subject/predicate history, with unrelated history untouched.
        current = next(row for row in rows if row.invalid_at is None)
        backend.mutation.apply(
            MutationBatch(
                invalidations=[
                    InvalidationOp(
                        None,
                        None,
                        "close current",
                        valid_to="2026-04-01T00:00:00+00:00",
                        target_claim_record_ids=(current.record_id,),
                    )
                ]
            ),
            expected_pot_id=pot,
            provenance_context=ProvenanceContext(mutation_id="close-current"),
        )
        remaining = backend.claim_query.find_claims(ClaimQueryFilter(pot_id=pot))
        assert [row.subject_key for row in remaining] == ["service:unrelated"]
