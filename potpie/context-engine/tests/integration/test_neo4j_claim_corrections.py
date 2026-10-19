"""Frozen correction intent and derived effects on a disposable Neo4j server."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from types import SimpleNamespace
import uuid

import pytest

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionReuseError,
)
from potpie_context_engine.adapters.outbound.graph.backends.neo4j_backend import (
    Neo4jGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.neo4j_writer import Neo4jGraphWriter
from potpie_context_engine.core.claim_corrections import (
    ClaimClosureResolution,
    ClaimInterval,
    ClaimTarget,
)
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.errors import (
    GraphMutationVersionConflict,
    ReconciliationApplyError,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch
from potpie_context_engine.core.workbench_service import GraphWorkbenchService
from potpie_context_engine.testing import InMemoryGraphPlanStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("PROTOCOL_TEST_NEO4J_URI"),
        reason="PROTOCOL_TEST_NEO4J_URI must name a disposable test server",
    ),
]

JAN = datetime(2020, 1, 1, tzinfo=timezone.utc)
FEB = datetime(2020, 2, 1, tzinfo=timezone.utc)
MAR = datetime(2020, 3, 1, tzinfo=timezone.utc)
APR = datetime(2020, 4, 1, tzinfo=timezone.utc)


@pytest.fixture
def graph():
    from neo4j import GraphDatabase

    settings = SimpleNamespace(
        is_enabled=lambda: True,
        neo4j_uri=lambda: os.environ["PROTOCOL_TEST_NEO4J_URI"],
        neo4j_user=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_USERNAME", "neo4j"),
        neo4j_password=lambda: os.environ.get("PROTOCOL_TEST_NEO4J_PASSWORD", ""),
    )
    backend = Neo4jGraphBackend(settings)
    pot = "correction-" + uuid.uuid4().hex
    with GraphDatabase.driver(
        settings.neo4j_uri(), auth=(settings.neo4j_user(), settings.neo4j_password())
    ) as driver:
        try:
            yield pot, backend, driver
        finally:
            backend.claim_query.close()
            _run(
                driver,
                "MATCH (n) WHERE n.group_id=$pot OR n.pot_id=$pot DETACH DELETE n",
                pot=pot,
            )


def _run(driver, query, **params):
    with driver.session() as session:
        return [dict(row) for row in session.run(query, **params)]


def _edge(point=JAN, *, target="team:red", **properties):
    return EdgeUpsert(
        "OWNED_BY",
        "service:api",
        target,
        {
            "claim_key": "claim:owner:" + target,
            "subgraph": "working",
            "environment": "prod",
            "source_system": "git",
            "source_ref": "file:owners",
            "truth": "source_observation",
            "fact": "API is owned by " + target,
            "valid_from": point.isoformat(),
            **properties,
        },
    )


def _seed(graph, *, closed=False):
    props = (
        {"invalid_at": FEB.isoformat(), "valid_until": FEB.isoformat()}
        if closed
        else {}
    )
    _apply(
        graph,
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:api", ("Service",)),
                EntityUpsert("team:red", ("Team",)),
                EntityUpsert("team:blue", ("Team",)),
            ],
            edge_upserts=[_edge(**props)],
        ),
        "seed",
    )
    return _rows(graph)[0]


def _rows(graph, **filters):
    pot, backend, _ = graph
    return backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=pot, include_invalidated=True, **filters)
    )


def _apply(graph, batch, mutation_id, *, context=None):
    pot, backend, _ = graph
    return backend.mutation.apply(
        batch,
        expected_pot_id=pot,
        provenance_context=context or ProvenanceContext(mutation_id=mutation_id),
    )


def _prepare(graph, batch, mutation_id, *, anchor=APR):
    pot, backend, _ = graph
    prepare_claim_assertions(
        batch,
        pot_id=pot,
        mutation_id=mutation_id,
        anchor=anchor,
        query=backend.claim_query,
        definition=backend.definition,
    )
    return ProvenanceContext(
        mutation_id=mutation_id,
        prepared=PreparedMutation(
            pot_id=pot,
            mutation_id=mutation_id,
            request_fingerprint="request:" + mutation_id,
            batch_fingerprint=batch_fingerprint(batch),
            expected_version=backend.mutation.current_version(pot),
            prepared_at=anchor,
            target_claim_record_ids=tuple(
                sorted(
                    {
                        closure.target.record_id
                        for op in (*batch.edge_deletes, *batch.invalidations)
                        for closure in op.closures
                    }
                )
            ),
        ),
    )


def _correction(row, *, endpoint=FEB, **fields):
    return InvalidationOp(
        None,
        None,
        "ownership corrected",
        valid_to=endpoint.isoformat(),
        target_claim_keys=(row.claim_key,),
        target_claim_record_ids=(row.record_id,),
        **fields,
    )


def _snapshot(graph):
    pot, backend, _ = graph
    return backend.snapshot.export_data(pot_id=pot)


def _receipts(graph):
    pot, _, driver = graph
    return _run(
        driver,
        "MATCH (r:PotpieMutationReceipt {pot_id:$pot}) RETURN r.mutation_id AS id,r.payload AS payload",
        pot=pot,
    )


@pytest.mark.parametrize("prepared", [False, True], ids=["raw", "prepared"])
def test_closure_rejects_cross_kind_identity_collision_atomically(graph, prepared):
    pot, backend, driver = graph
    original = _seed(graph)
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert(
                "service:api", (), {"summary": "must roll back"}, preserve_labels=True
            )
        ],
        invalidations=[_correction(original)],
    )
    context = _prepare(graph, batch, "colliding-closure") if prepared else None
    _run(
        driver,
        "CREATE (:Entity:Service {group_id:$pot,entity_key:'service:unrelated',"
        "record_id:$record_id,summary:'isolated identity corruption'})",
        pot=pot,
        record_id=original.record_id,
    )

    def physical_graph():
        return (
            _run(
                driver,
                "MATCH (e:Entity {group_id:$pot}) RETURN properties(e) AS props,labels(e) AS labels "
                "ORDER BY e.entity_key",
                pot=pot,
            ),
            _run(
                driver,
                "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() RETURN properties(r) AS props "
                "ORDER BY coalesce(r.record_id,r.uuid)",
                pot=pot,
            ),
        )

    before = physical_graph()
    receipts = _receipts(graph)
    version = backend.mutation.current_version(pot)
    with pytest.raises(
        ReconciliationApplyError, match="collides with an entity identity"
    ):
        _apply(graph, batch, "colliding-closure", context=context)
    assert physical_graph() == before
    assert backend.mutation.current_version(pot) == version
    assert _receipts(graph) == receipts
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="colliding-closure"
        ).state
        == "absent"
    )


@pytest.mark.parametrize(
    "same_batch", [False, True], ids=["existing-history", "one-batch"]
)
def test_plain_raw_assertions_reject_overlapping_stated_singleton_history(
    graph, same_batch
):
    pot, backend, _ = graph
    if not same_batch:
        _seed(graph)
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",), {"summary": "must roll back"}),
            EntityUpsert("team:red", ("Team",)),
            EntityUpsert("team:blue", ("Team",)),
        ],
        edge_upserts=[
            *([_edge(JAN)] if same_batch else []),
            _edge(MAR, target="team:blue", truth="agent_claim"),
        ],
    )
    before = _snapshot(graph)
    receipts = _receipts(graph)
    version = backend.mutation.current_version(pot)
    assert all(edge.resolution is None for edge in batch.edge_upserts)
    with pytest.raises(ReconciliationApplyError, match="same-source singleton history"):
        _apply(graph, batch, "overlapping-raw-assertions")
    assert _snapshot(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert _receipts(graph) == receipts
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="overlapping-raw-assertions"
        ).state
        == "absent"
    )


@pytest.mark.asyncio
async def test_writer_claim_rows_reads_uncommitted_retired_history_from_borrowed_transaction(
    graph, monkeypatch
):
    pot, backend, _ = graph
    await backend.mutation.apply_async(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:api", ("Service",)),
                EntityUpsert("team:red", ("Team",)),
            ]
        ),
        expected_pot_id=pot,
    )
    driver = backend.graph_writer._new_driver()
    observed = []

    async def read(transaction):
        result = await transaction.run(
            "MATCH (a:Entity {group_id:$pot,entity_key:'service:api'}),"
            "(b:Entity {group_id:$pot,entity_key:'team:red'}) "
            "CREATE (a)-[:RELATES_TO {group_id:$pot,name:'OWNED_BY',"
            "subject_key:'service:api',object_key:'team:red',uuid:'private-legacy',"
            "retired:true,valid_at:$start,invalid_at:$end}]->(b)",
            pot=pot,
            start=JAN.isoformat(),
            end=FEB.isoformat(),
        )
        await result.consume()
        writer = backend.graph_writer.in_transaction(transaction)

        def no_second_driver():
            pytest.fail("borrowed correction read must use its owning transaction")

        monkeypatch.setattr(writer, "_new_driver", no_second_driver)
        rows = await writer.claim_rows(pot)
        observed.extend(
            (row.record_id, row.retired, row.valid_at, row.invalid_at) for row in rows
        )
        assert _rows(graph, include_retired=True) == []
        raise RuntimeError("rollback read probe")

    try:
        async with driver.session() as session:
            with pytest.raises(RuntimeError, match="rollback read probe"):
                await session.execute_write(read)
    finally:
        await driver.close()
    assert observed == [("private-legacy", True, JAN, FEB)]
    assert _rows(graph, include_retired=True) == []


def test_frozen_empty_entity_selection_has_no_projection_or_supersedes_effect(graph):
    _seed(graph)
    before = _snapshot(graph)
    batch = MutationBatch(
        invalidations=[
            InvalidationOp(
                "service:api",
                None,
                "empty selection",
                superseded_by_key="team:blue",
                target_claim_keys=(),
                target_claim_record_ids=(),
            )
        ]
    )
    context = _prepare(graph, batch, "empty")
    assert batch.invalidations[0].closures == ()
    assert batch.entity_upserts == batch.edge_upserts == []
    first = _apply(graph, batch, "empty", context=context)
    assert first.execution.affected_claim_record_ids == ()
    assert first.mutation_summary.invalidations_applied == 0
    assert _snapshot(graph) == before
    assert _apply(graph, batch, "empty", context=context) == first


def test_explicit_physical_correction_extends_only_selected_historical_interval(graph):
    original = _seed(graph, closed=True)
    _apply(graph, MutationBatch(edge_upserts=[_edge(MAR)]), "march")
    newer = next(row for row in _rows(graph) if row.record_id != original.record_id)
    before = _snapshot(graph)
    endpoint = FEB + timedelta(days=14)
    batch = MutationBatch(invalidations=[_correction(original, endpoint=endpoint)])
    context = _prepare(graph, batch, "extend-history")
    closure = batch.invalidations[0].closures[0]
    assert closure.target == ClaimTarget(original.claim_key, original.record_id)
    assert closure.before == ClaimInterval(JAN, FEB, FEB)
    assert closure.after == ClaimInterval(JAN, endpoint, endpoint)
    result = _apply(graph, batch, "extend-history", context=context)
    rows = {row.record_id: row for row in _rows(graph)}
    assert (
        rows[original.record_id].valid_at,
        rows[original.record_id].invalid_at,
        rows[original.record_id].valid_until,
    ) == (JAN, endpoint, endpoint)
    assert rows[newer.record_id] == newer
    old_payload = next(
        row for row in before["claims"] if row["record_id"] == newer.record_id
    )
    assert (
        next(
            row
            for row in _snapshot(graph)["claims"]
            if row["record_id"] == newer.record_id
        )
        == old_payload
    )
    assert result.execution.affected_claim_record_ids == (original.record_id,)
    pot, backend, _ = graph
    visible = backend.claim_query.find_claims(
        ClaimQueryFilter(
            pot_id=pot, predicate_in=("OWNED_BY",), as_of=FEB + timedelta(days=7)
        )
    )
    assert [row.record_id for row in visible] == [original.record_id]


def test_delayed_execution_uses_frozen_closure_past_original_endpoint(graph):
    original = _seed(graph, closed=True)
    anchor = JAN + timedelta(days=14)
    batch = MutationBatch(
        invalidations=[
            InvalidationOp(
                None,
                None,
                "prepared while still applicable",
                target_claim_keys=(original.claim_key,),
            )
        ]
    )
    context = _prepare(graph, batch, "delayed", anchor=anchor)
    assert batch.invalidations[0].closures[0].after.invalid_at == anchor
    assert datetime.now(timezone.utc) > FEB
    result = _apply(graph, batch, "delayed", context=context)
    row = _rows(graph)[0]
    assert (row.valid_at, row.invalid_at, row.valid_until) == (JAN, anchor, FEB)
    assert result.execution.affected_claim_record_ids == (original.record_id,)


def _replacement_batch(original, *, entity_projection=False):
    op = _correction(original, endpoint=MAR, superseded_by_key="team:blue")
    if entity_projection:
        op = InvalidationOp(
            "service:api",
            None,
            "raw entity correction",
            valid_to=MAR.isoformat(),
            superseded_by_key="team:blue",
        )
    return MutationBatch(
        edge_upserts=[_edge(MAR, target="team:blue")], invalidations=[op]
    )


def test_prepared_entity_projection_preserves_base_registered_and_custom_labels(graph):
    pot, _, driver = graph
    original = _seed(graph)
    labels = set(
        _run(
            driver,
            "MATCH (e:Entity {group_id:$pot,entity_key:'service:api'}) "
            "SET e:Repository:RetainedCustom RETURN labels(e) AS labels",
            pot=pot,
        )[0]["labels"]
    )
    assert {"Entity", "Service", "Repository", "RetainedCustom"} <= labels
    batch = _replacement_batch(original, entity_projection=True)
    context = _prepare(graph, batch, "projection")
    projection = next(
        item for item in batch.entity_upserts if item.entity_key == "service:api"
    )
    assert projection.preserve_labels
    assert projection.labels == ()
    result = _apply(graph, batch, "projection", context=context)
    assert result.mutation_summary.entity_upserts_applied == 1
    rows = _run(
        driver,
        "MATCH (e:Entity {group_id:$pot,entity_key:'service:api'}) "
        "RETURN labels(e) AS labels,e.invalidated AS invalidated,e.valid_to AS valid_to",
        pot=pot,
    )
    assert set(rows[0]["labels"]) == labels
    assert rows[0]["invalidated"] is True
    assert rows[0]["valid_to"] == MAR.isoformat()


def test_missing_projection_target_rolls_back_earlier_property_patch(graph):
    pot, backend, _ = graph
    _seed(graph)
    before = _snapshot(graph)
    receipts = _receipts(graph)
    version = backend.mutation.current_version(pot)
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert(
                "service:api", (), {"summary": "must roll back"}, preserve_labels=True
            ),
            EntityUpsert(
                "service:missing", (), {"invalidated": True}, preserve_labels=True
            ),
        ]
    )
    context = _prepare(graph, batch, "missing-projection")
    with pytest.raises(ReconciliationApplyError, match="projection target is missing"):
        _apply(graph, batch, "missing-projection", context=context)
    assert _snapshot(graph) == before
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == version
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="missing-projection"
        ).state
        == "absent"
    )


@pytest.mark.parametrize("prepared", [False, True], ids=["raw", "prepared"])
def test_supersede_has_one_cutover_explicit_effects_and_original_receipt_identity(
    graph, prepared
):
    pot, backend, _ = graph
    original = _seed(graph)
    batch = _replacement_batch(original)
    context = _prepare(graph, batch, "replace") if prepared else None
    before = deepcopy(batch)
    fingerprint = batch_fingerprint(batch)
    result = _apply(graph, batch, "replace", context=context)
    assert batch == before
    assert result.execution.batch_fingerprint == fingerprint
    rows = _rows(graph)
    old = next(row for row in rows if row.record_id == original.record_id)
    replacement = next(
        row
        for row in rows
        if row.predicate == "OWNED_BY" and row.object_key == "team:blue"
    )
    effect = next(row for row in rows if row.predicate == "SUPERSEDES")
    assert (old.valid_at, old.invalid_at) == (JAN, MAR)
    assert (replacement.valid_at, replacement.invalid_at, replacement.valid_until) == (
        MAR,
        None,
        None,
    )
    assert effect.valid_at == MAR
    assert effect.properties["supersedes_record_id"] == old.record_id
    assert effect.properties["replacement_record_id"] == replacement.record_id
    assert (
        effect.subgraph,
        effect.environment,
        effect.source_system,
        effect.source_ref,
    ) == (
        original.subgraph,
        original.environment,
        original.source_system,
        original.source_ref,
    )
    assert set(result.execution.affected_claim_record_ids) == {
        old.record_id,
        replacement.record_id,
        effect.record_id,
    }
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="replace"
        ).result
        == result
    )
    revision = backend.mutation.current_version(pot)
    assert _apply(graph, batch, "replace", context=context) == result
    assert backend.mutation.current_version(pot) == revision


@pytest.mark.parametrize("duplicate", ["operations", "selector"])
def test_duplicate_physical_targets_fail_without_graph_revision_or_receipt_change(
    graph, duplicate
):
    pot, backend, _ = graph
    original = _seed(graph)
    op = _correction(original)
    operations = [op, deepcopy(op)]
    if duplicate == "selector":
        op.target_claim_record_ids = (original.record_id, original.record_id)
        operations = [op]
    batch = MutationBatch(
        invalidations=operations,
        entity_upserts=[
            EntityUpsert("service:api", ("Service",), {"summary": "must roll back"})
        ],
    )
    before = _snapshot(graph)
    version = backend.mutation.current_version(pot)
    receipts = _receipts(graph)
    with pytest.raises(ReconciliationApplyError, match="duplicate"):
        _apply(graph, batch, "duplicate")
    assert _snapshot(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert _receipts(graph) == receipts


def test_failure_after_projection_closure_and_system_effect_rolls_back_together(
    graph, monkeypatch
):
    pot, backend, _ = graph
    original = _seed(graph)
    before = _snapshot(graph)
    version = backend.mutation.current_version(pot)
    receipts = _receipts(graph)
    batch = _replacement_batch(original, entity_projection=True)
    original_write = Neo4jGraphWriter.upsert_edges
    observed = []

    async def fail_after_effects(self, selected_pot, edges, provenance):
        await original_write(self, selected_pot, edges, provenance)
        result = await self._transaction.run(
            "MATCH (e:Entity {group_id:$pot,entity_key:'service:api'}) "
            "OPTIONAL MATCH ()-[r:RELATES_TO {group_id:$pot,name:'SUPERSEDES'}]->() "
            "RETURN e.invalidated AS projected,count(r) AS effects",
            pot=pot,
        )
        row = await result.single()
        await result.consume()
        observed.append((row["projected"], row["effects"]))
        raise RuntimeError("injected after explicit derived effects")

    monkeypatch.setattr(Neo4jGraphWriter, "upsert_edges", fail_after_effects)
    with pytest.raises(ReconciliationApplyError, match="injected after explicit"):
        _apply(graph, batch, "failed-effects")
    assert observed == [(True, 1)]
    assert _snapshot(graph) == before
    assert backend.mutation.current_version(pot) == version
    assert _receipts(graph) == receipts
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="failed-effects"
        ).state
        == "absent"
    )


def test_later_incarnation_after_preparation_conflicts_before_touching_new_record(
    graph,
):
    pot, backend, _ = graph
    original = _seed(graph)
    batch = MutationBatch(invalidations=[_correction(original)])
    context = _prepare(graph, batch, "previewed")
    _apply(
        graph, MutationBatch(invalidations=[_correction(original)]), "concurrent-close"
    )
    _apply(graph, MutationBatch(edge_upserts=[_edge(MAR)]), "later-incarnation")
    before = _snapshot(graph)
    with pytest.raises(GraphMutationVersionConflict):
        _apply(graph, batch, "previewed", context=context)
    assert _snapshot(graph) == before
    assert (
        backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="previewed"
        ).state
        == "absent"
    )


@pytest.mark.parametrize(
    "closures",
    [
        (),
        (
            ClaimClosureResolution(
                ClaimTarget("claim:never", "record:never"),
                ClaimInterval(JAN, None, None),
                ClaimInterval(JAN, FEB, None),
            ),
        ),
    ],
    ids=["empty", "populated"],
)
def test_raw_frozen_closure_injection_is_rejected(graph, closures):
    pot, backend, _ = graph
    batch = MutationBatch(
        edge_deletes=[EdgeDelete("OWNED_BY", "service:api", "team:red", closures)]
    )
    with pytest.raises(MutationExecutionReuseError, match="version-3"):
        _apply(graph, batch, "injected")
    assert backend.mutation.current_version(pot) == 0
    assert _receipts(graph) == []


@pytest.mark.parametrize("changed", ["interval", "logical-key"])
def test_frozen_before_state_detects_native_drift_without_revision_change(
    graph, changed
):
    pot, backend, driver = graph
    original = _seed(graph)
    batch = MutationBatch(invalidations=[_correction(original)])
    context = _prepare(graph, batch, "reviewed")
    assignment = (
        "r.invalid_at=$value" if changed == "interval" else "r.claim_key=$value"
    )
    _run(
        driver,
        "MATCH ()-[r:RELATES_TO {group_id:$pot}]->() "
        f"WHERE r.record_id=$record_id SET {assignment}",
        pot=pot,
        record_id=original.record_id,
        value=MAR.isoformat() if changed == "interval" else "claim:changed",
    )
    before = _snapshot(graph)
    receipts = _receipts(graph)
    expected = (
        "interval changed" if changed == "interval" else "outside its selected scope"
    )
    with pytest.raises(ReconciliationApplyError, match=expected):
        _apply(graph, batch, "reviewed", context=context)
    assert _snapshot(graph) == before
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == context.prepared.expected_version


def test_semantic_supersede_rejects_distinct_start_and_end_cutovers(graph):
    pot, backend, _ = graph
    workbench = GraphWorkbenchService(
        backend=backend, plan_store=InMemoryGraphPlanStore()
    )
    base = {
        "op": "link_entities",
        "subgraph": "infra_topology",
        "environment": "prod",
        "subject": {"key": "service:api", "type": "Service"},
        "predicate": "DEPENDS_ON",
        "object": {"key": "service:ledger", "type": "Service"},
        "truth": "source_observation",
        "evidence": [{"source_ref": "fixture:dependency"}],
        "description": "API calls ledger",
        "valid_from": JAN.isoformat(),
    }
    proposed = workbench.propose({"operations": [base]}, pot_id=pot)
    assert workbench.commit(proposed.plan_id, pot_id=pot).ok
    before = _snapshot(graph)
    receipts = _receipts(graph)
    version = backend.mutation.current_version(pot)
    rejected = workbench.propose(
        {
            "operations": [
                {
                    **base,
                    "op": "supersede_claim",
                    "valid_until": MAR.isoformat(),
                    "superseded_by": {
                        "key": "service:ledger-v2",
                        "type": "Service",
                    },
                    "reason": "dependency changed",
                }
            ]
        },
        pot_id=pot,
        approved_by="test:reviewer",
    )
    assert not rejected.ok
    assert rejected.status == "invalid"
    assert any("same cutover" in issue["message"] for issue in rejected.issues)
    assert _snapshot(graph) == before
    assert _receipts(graph) == receipts
    assert backend.mutation.current_version(pot) == version


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_preparation_recovers_completed_receipt_but_cannot_start_new_execution(
    graph, schema
):
    pot, backend, driver = graph
    batch = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",)),
            EntityUpsert("team:red", ("Team",)),
        ],
        edge_upserts=[_edge()],
    )
    if schema == 2:
        current = _prepare(graph, batch, "legacy")
        first = _apply(graph, batch, "legacy", context=current)
        context = replace(current, prepared=replace(current.prepared, schema_version=2))
    else:
        first = _apply(graph, batch, "legacy")
        payload = json.loads(
            next(row["payload"] for row in _receipts(graph) if row["id"] == "legacy")
        )
        payload[0]["result"].pop("execution")
        _run(
            driver,
            "MATCH (r:PotpieMutationReceipt {pot_id:$pot,mutation_id:'legacy'}) SET r.payload=$payload",
            pot=pot,
            payload=json.dumps(payload, sort_keys=True),
        )
        first = backend.mutation.lookup_execution(
            batch, expected_pot_id=pot, mutation_id="legacy"
        ).result
        context = ProvenanceContext(
            mutation_id="legacy",
            prepared=PreparedMutation(
                pot_id=pot,
                mutation_id="legacy",
                request_fingerprint="legacy-request",
                batch_fingerprint=batch_fingerprint(batch),
                expected_version=0,
                prepared_at=JAN,
                schema_version=1,
            ),
        )
    assert _apply(graph, batch, "legacy", context=context) == first
    before = _snapshot(graph)
    receipts = _receipts(graph)
    new = replace(
        context,
        mutation_id="unexecuted-legacy",
        prepared=replace(context.prepared, mutation_id="unexecuted-legacy"),
    )
    with pytest.raises(MutationExecutionReuseError, match="migration_required"):
        _apply(graph, batch, "unexecuted-legacy", context=new)
    assert backend.mutation.current_version(pot) == 1
    assert _snapshot(graph) == before
    assert _receipts(graph) == receipts
