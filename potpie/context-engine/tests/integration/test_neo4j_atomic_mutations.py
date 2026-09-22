"""Run with POTPIE_TEST_NEO4J_URI pointed at a disposable local test database."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
import threading
import uuid

import pytest

from potpie_context_core.errors import (
    GraphMutationVersionConflict,
    ReconciliationApplyError,
)
from potpie_context_core.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceContext
from potpie_context_core.ports.claim_query import ClaimQueryFilter
from potpie_context_core.reconciliation import MutationBatch
from potpie_context_core.workbench_service import GraphWorkbenchService
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("POTPIE_TEST_NEO4J_URI"),
        reason="requires disposable Neo4j URI",
    ),
]


class Settings:
    def is_enabled(self):
        return True

    def neo4j_uri(self):
        return os.environ["POTPIE_TEST_NEO4J_URI"]

    def neo4j_user(self):
        return "neo4j"

    def neo4j_password(self):
        return "unused-in-disposable-auth-disabled-database"


def _batch(value):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert(
                "service:api",
                ("Service",),
                {"summary": value},
            )
        ]
    )


def test_two_backend_instances_compare_and_apply_atomically():
    pot = "atomic-" + uuid.uuid4().hex
    backends = [Neo4jGraphBackend(Settings()), Neo4jGraphBackend(Settings())]
    initial = backends[0].mutation.current_version(pot)
    ready = threading.Barrier(2)

    def apply(index):
        ready.wait(timeout=10)
        try:
            return (
                backends[index]
                .mutation.compare_and_apply(
                    _batch(str(index)),
                    expected_pot_id=pot,
                    expected_version=initial,
                    provenance_context=ProvenanceContext(mutation_id=f"write-{index}"),
                )
                .ok
            )
        except GraphMutationVersionConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(apply, [0, 1]))
    assert sorted(results) == [False, True]
    assert backends[0].mutation.current_version(pot) == initial + 1


def test_receipt_survives_new_backend_and_rejects_different_body():
    from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
        MutationExecutionReuseError,
    )

    pot = "retry-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())
    context = ProvenanceContext(mutation_id="stable-retry")
    first = backend.mutation.compare_and_apply(
        _batch("one"),
        expected_pot_id=pot,
        expected_version=0,
        provenance_context=context,
    )
    restarted = Neo4jGraphBackend(Settings())
    replay = restarted.mutation.compare_and_apply(
        _batch("one"),
        expected_pot_id=pot,
        expected_version=0,
        provenance_context=context,
    )
    assert replay == first
    assert (
        restarted.mutation.lookup_execution(
            _batch("one"), expected_pot_id=pot, mutation_id="stable-retry"
        ).result
        == first
    )
    assert restarted.mutation.current_version(pot) == 1
    with pytest.raises(MutationExecutionReuseError):
        restarted.mutation.apply(
            _batch("different"), expected_pot_id=pot, provenance_context=context
        )


def test_reset_advances_revision_and_keeps_historical_receipt():
    pot = "reset-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())
    context = ProvenanceContext(mutation_id="before-reset")
    first = backend.mutation.apply(
        _batch("one"), expected_pot_id=pot, provenance_context=context
    )
    version = backend.mutation.current_version(pot)
    assert backend.mutation.reset_pot(pot)["ok"]
    assert backend.mutation.current_version(pot) == version + 1
    with pytest.raises(GraphMutationVersionConflict):
        backend.mutation.compare_and_apply(
            _batch("stale"),
            expected_pot_id=pot,
            expected_version=version,
            provenance_context=ProvenanceContext(mutation_id="stale"),
        )
    assert (
        backend.mutation.lookup_execution(
            _batch("one"), expected_pot_id=pot, mutation_id="before-reset"
        ).result
        == first
    )


def test_batch_error_rolls_back_entities_revision_and_receipt(monkeypatch):
    pot = "rollback-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())

    async def fail_edges(self, *args):
        raise RuntimeError("injected after entity write")

    monkeypatch.setattr(Neo4jGraphWriter, "upsert_edges", fail_edges)
    context = ProvenanceContext(mutation_id="failed-batch")
    with pytest.raises(ReconciliationApplyError, match="injected"):
        backend.mutation.compare_and_apply(
            _batch("must roll back"),
            expected_pot_id=pot,
            expected_version=0,
            provenance_context=context,
        )
    assert backend.mutation.current_version(pot) == 0
    assert (
        backend.claim_query.entity_properties(pot_id=pot, entity_key="service:api")
        == {}
    )
    assert (
        backend.mutation.lookup_execution(
            _batch("must roll back"), expected_pot_id=pot, mutation_id="failed-batch"
        ).state
        == "absent"
    )


def test_entity_only_stale_workbench_plan_is_rejected(tmp_path):
    pot = "workbench-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())
    workbench = GraphWorkbenchService(
        backend=backend, plan_store=LocalJsonGraphPlanStore(home=tmp_path)
    )

    def payload(value):
        return {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {
                        "key": "service:api",
                        "type": "Service",
                        "properties": {"summary": value},
                    },
                }
            ]
        }

    stale = workbench.propose(payload("old"), pot_id=pot)
    newer = workbench.propose(payload("new"), pot_id=pot)
    assert workbench.commit(newer.plan_id, pot_id=pot).ok
    receipt = workbench.commit(stale.plan_id, pot_id=pot)
    assert not receipt.ok
    assert receipt.status == "conflict"
    assert (
        backend.claim_query.entity_properties(pot_id=pot, entity_key="service:api")[
            "summary"
        ]
        == "new"
    )


@pytest.mark.parametrize(
    "operation", ["end_relation_validity", "retract_claim", "supersede_claim"]
)
def test_correction_preserves_staging_and_verifies_exact_targets(tmp_path, operation):
    pot = "correction-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())
    workbench = GraphWorkbenchService(
        backend=backend, plan_store=LocalJsonGraphPlanStore(home=tmp_path)
    )
    base = {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "subject": {"key": "service:api", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:ledger", "type": "Service"},
        "truth": "source_observation",
        "evidence": [{"source_ref": "fixture:dependency"}],
        "description": "API calls ledger in this environment",
    }
    seed = workbench.propose(
        {"operations": [{**base, "environment": env} for env in ("prod", "staging")]},
        pot_id=pot,
    )
    seeded = workbench.commit(seed.plan_id, pot_id=pot, verify=True)
    assert seeded.verification.ok, seeded.verification.content_readback
    query = ClaimQueryFilter(
        pot_id=pot, predicate_in=("DEPENDS_ON",), include_invalidated=True
    )
    before = backend.claim_query.find_claims(query)
    prod = next(row for row in before if row.environment == "prod")
    staging = next(row for row in before if row.environment == "staging")
    correction = {
        **base,
        "op": operation,
        "environment": "prod",
        "reason": "Production target changed",
    }
    if operation == "supersede_claim":
        correction["superseded_by"] = {"key": "service:ledger-v2", "type": "Service"}
    preview = workbench.propose(
        {"operations": [correction]}, pot_id=pot, approved_by="test:reviewer"
    )
    assert preview.diff.retracted_claim_keys == (prod.claim_key,)
    receipt = workbench.commit(preview.plan_id, pot_id=pot, verify=True)
    assert receipt.ok
    assert not receipt.verification.content_readback["mismatches"]
    after = {row.claim_key: row for row in backend.claim_query.find_claims(query)}
    assert after[staging.claim_key] == staging
    assert after[prod.claim_key].invalid_at is not None


def test_raw_claim_invalidation_advances_revision(tmp_path):
    from potpie_context_core.graph_mutations import EdgeUpsert

    pot = "invalidate-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())
    plan = MutationBatch(entity_upserts=[EntityUpsert("service:api", ("Service",)), EntityUpsert("service:ledger", ("Service",))], edge_upserts=[EdgeUpsert("DEPENDS_ON", "service:api", "service:ledger", {
        "claim_key": "claim:raw-invalidate", "fact": "API calls ledger",
    })])
    backend.mutation.apply(plan, expected_pot_id=pot)
    version = backend.mutation.current_version(pot)
    assert backend.mutation.invalidate(pot_id=pot, claim_keys=["claim:raw-invalidate"]) == 1
    assert backend.mutation.current_version(pot) == version + 1
    with pytest.raises(GraphMutationVersionConflict):
        backend.mutation.compare_and_apply(_batch("stale"), expected_pot_id=pot, expected_version=version)


@pytest.mark.parametrize("expiry_days", [7, -1], ids=["future", "expired"])
def test_evidence_marker_refresh_preserves_invalidation_and_claim(expiry_days):
    pot = "marker-" + uuid.uuid4().hex
    backend = Neo4jGraphBackend(Settings())
    invalid_at = (
        datetime.now(timezone.utc) + timedelta(days=expiry_days)
    ).isoformat()

    def batch(properties):
        return MutationBatch(
            entity_upserts=[
                EntityUpsert("document:guide", ("Entity", "Document")),
                EntityUpsert("service:api", ("Entity", "Service")),
            ],
            edge_upserts=[
                EdgeUpsert(
                    "DOCUMENTS",
                    "document:guide",
                    "service:api",
                    {
                        "claim_key": "claim:guide-api",
                        "source_ref": "resource:guide",
                        "fact": "Guide documents API behavior",
                        **properties,
                    },
                )
            ],
        )

    backend.mutation.apply(
        batch(
            {
                "invalid_at": invalid_at,
                "invalidation_reason": "scheduled source retirement",
                "invalidated_by": "refresh:scheduled",
            }
        ),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id="marker-seed"),
    )
    before = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot,
            claim_key_in=("claim:guide-api",),
            include_invalidated=True,
        )
    )[0]
    backend.mutation.apply(
        batch(
            {
                "evidence_review_required": True,
                "evidence_review_reason": "source version changed",
            }
        ),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id="marker-refresh"),
    )

    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot,
            claim_key_in=("claim:guide-api",),
            include_invalidated=True,
        )
    )
    assert len(rows) == 1
    assert rows[0].invalid_at.isoformat() == invalid_at
    assert rows[0].fact == "Guide documents API behavior"
    assert rows[0].properties["invalidation_reason"] == "scheduled source retirement"
    assert rows[0].properties["evidence_review_required"] is True
    assert rows[0].properties.get("revived_at") == before.properties.get("revived_at")

    backend.mutation.apply(
        batch({}),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id="ordinary-reassert"),
    )
    revived = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot,
            claim_key_in=("claim:guide-api",),
            include_invalidated=True,
        )
    )
    assert len(revived) == 1
    assert revived[0].invalid_at is None
