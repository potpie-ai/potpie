# ruff: noqa: F811 - imported pytest fixtures are injected as test parameters.
"""Real server/Lite temporal publication and vector post-filter contracts."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from potpie_context_engine.adapters.outbound.graph.falkordb_reader import (
    FalkorDBClaimQueryStore,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
)
from potpie_context_engine.core.errors import MutationBatchValidationError
from potpie_context_engine.core.graph_snapshot_mutation import prepare_snapshot_import
from potpie_context_engine.core.claim_preparation import prepare_claim_assertions
from potpie_context_engine.core.mutation_execution import (
    PreparedMutation,
    batch_fingerprint,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
)

from tests.integration.backends.test_falkordb_reliability import (
    Settings,
    backend,
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration
POINT = datetime(2026, 3, 1, tzinfo=timezone.utc)


class Embedder:
    name = "native-temporal-test"
    dimensions = 3

    def embed(self, _text):
        return (1.0, 0.0, 0.0)


def reader(graph, *, vector, require_vector=True):
    if vector:
        graph.query(
            "CREATE VECTOR INDEX FOR ()-[r:RELATES_TO]->() ON (r.fact_embedding) "
            "OPTIONS {dimension:3,similarityFunction:'cosine'}"
        )
    result = FalkorDBClaimQueryStore(
        Settings(), graph=graph, embedder=Embedder() if vector else None
    )
    if vector and require_vector:
        # A lexical fallback cannot prove native ANN filtering behavior.
        result._find_claims_lexical = lambda _: pytest.fail(
            "native vector query unexpectedly fell back to lexical"
        )
    return result


def legacy_claim(
    graph,
    index,
    *,
    subject_label="Service",
    object_label="Team",
    subject_retired=False,
    object_retired=False,
    **properties,
):
    props = {
        "group_id": "p",
        "name": "USES",
        "subject_key": f"service:s{index}",
        "object_key": f"team:o{index}",
        "record_id": f"record:{index}",
        "uuid": f"uuid:{index}",
        "claim_key": f"claim:{index}",
        "fact": "matching fact",
        "valid_from": "2026-02-28T17:30:00-05:30",
        **properties,
    }
    graph.query(
        f"CREATE (a:Entity:{subject_label} {{group_id:'p',entity_key:$source,retired:$sr}}),"
        f"(b:Entity:{object_label} {{group_id:'p',entity_key:$target,retired:$tr}}) "
        "CREATE (a)-[r:RELATES_TO]->(b) SET r=$props "
        "SET r.fact_embedding=vecf32($embedding)",
        params={
            "source": props["subject_key"],
            "target": props["object_key"],
            "sr": subject_retired,
            "tr": object_retired,
            "props": props,
            "embedding": [1.0, index / 10, 0.0],
        },
    )


@pytest.mark.parametrize("vector", (False, True), ids=("lexical", "vector"))
@pytest.mark.parametrize(
    "axis",
    (
        "time",
        "environment",
        "truth",
        "exact-text",
        "pattern",
        "endpoint-label",
        "subject-label",
        "object-label",
        "retired-subject",
        "retired-object",
    ),
)
def test_native_post_filters_skip_first_candidate_before_limit(graph, vector, axis):
    query = reader(graph, vector=vector)
    first, second, filters = {}, {}, {}
    if axis == "time":
        first["invalid_at"] = "2026-03-01T04:00:00+05:30"
        second["invalid_at"] = "2026-02-28T20:00:00-05:30"
        filters.update(
            valid_at_after=datetime(2026, 2, 28, 22, tzinfo=timezone.utc),
            valid_at_before=datetime(2026, 2, 28, 23, 30, tzinfo=timezone.utc),
        )
    elif axis == "environment":
        first["environment"], second["environment"] = "stage", "PROD"
        filters["environment_in"] = ("prod",)
    elif axis == "truth":
        first["truth"], second["truth"] = "source_observation", "decision"
        filters["truth_in"] = ("DECISION",)
    elif axis == "exact-text":
        first["description"], second["description"] = "unrelated", "PR-12"
        filters["exact_text_in"] = ("pr-12",)
    elif axis == "pattern":
        first["description"], second["description"] = "PR-123", "PR-12"
        filters["exact_text_pattern"] = r".*\bpr-12\b.*"
    elif axis in {"endpoint-label", "subject-label"}:
        second["subject_label"] = "Person"
        filters[axis.replace("-", "_")] = "Person"
    elif axis == "object-label":
        second["object_label"] = "Person"
        filters["object_label"] = "Person"
    else:
        first[axis.replace("retired-", "") + "_retired"] = True
    legacy_claim(graph, 0, **first)
    legacy_claim(graph, 1, **second)
    rows = query.find_claims(
        ClaimQueryFilter(
            pot_id="p",
            fact_query="matching fact" if vector else None,
            as_of=POINT,
            limit=1,
            **filters,
        )
    )
    assert [row.record_id for row in rows] == ["record:1"]


@pytest.mark.parametrize("vector", (False, True), ids=("lexical", "vector"))
@pytest.mark.parametrize("label_filter", ("subject_label", "endpoint_label"))
def test_label_filters_honor_explicit_retired_endpoint_inclusion(
    graph, vector, label_filter
):
    query = reader(graph, vector=vector)
    legacy_claim(graph, 0, subject_label="Person", subject_retired=True)
    rows = query.find_claims(
        ClaimQueryFilter(
            pot_id="p",
            fact_query="matching fact" if vector else None,
            as_of=POINT,
            include_retired=True,
            limit=1,
            **{label_filter: "Person"},
        )
    )
    assert [row.record_id for row in rows] == ["record:0"]


@pytest.mark.parametrize("vector", (False, True), ids=("lexical", "vector"))
def test_limit_zero_does_not_query_or_embed(graph, vector):
    query = reader(graph, vector=vector)
    query._get_graph = lambda: pytest.fail("limit zero queried the graph")
    assert (
        query.find_claims(
            ClaimQueryFilter(
                pot_id="p", fact_query="match" if vector else None, limit=0
            )
        )
        == []
    )


@pytest.mark.parametrize("vector", (False, True), ids=("lexical", "vector"))
def test_half_open_boundaries_compare_offset_instants_and_microseconds(graph, vector):
    query = reader(graph, vector=vector, require_vector=False)
    legacy_claim(
        graph,
        0,
        valid_from="2026-03-01T05:30:00.000001+05:30",
        valid_until="2026-02-28T19:00:00.000003-05:00",
    )
    for offset, expected in ((0, []), (1, ["record:0"]), (2, ["record:0"]), (3, [])):
        rows = query.find_claims(
            ClaimQueryFilter(
                pot_id="p",
                fact_query="match" if vector else None,
                as_of=POINT + timedelta(microseconds=offset),
                limit=1,
            )
        )
        assert [row.record_id for row in rows] == expected


def import_payload(timestamp):
    return {
        "format_version": "2",
        "pot_id": "source",
        "entities": [
            {"key": "service:a", "labels": ["Entity", "Service"], "properties": {}},
            {"key": "team:b", "labels": ["Entity", "Team"], "properties": {}},
        ],
        "claims": [
            {
                "predicate": "USES",
                "subject_key": "service:a",
                "object_key": "team:b",
                "claim_key": "imported",
                "valid_at": timestamp,
                "observed_at": "2026-03-01T00:30:00Z",
                "valid_until": "2026-03-02T05:30:00+05:30",
                "properties": {},
            }
        ],
    }


def test_snapshot_times_canonicalize_after_identity_without_mutating_payload(graph):
    store = backend(graph)
    payload = import_payload("2026-03-01T05:30:00.000001+05:30")
    original = deepcopy(payload)
    batch, context = prepare_snapshot_import(
        pot_id="p", payload=payload, expected_version=0
    )
    imported = store.snapshot.import_data(pot_id="p", payload=payload)
    assert payload == original
    assert imported.metadata["mutation_id"] == context.mutation_id
    row = store.snapshot.export_data(pot_id="p")["claims"][0]
    assert row["valid_at"] == "2026-03-01T00:00:00.000001+00:00"
    assert row["observed_at"] == "2026-03-01T00:30:00+00:00"
    assert row["valid_until"] == "2026-03-02T00:00:00+00:00"
    assert row["record_id"] == batch.snapshot_import["claims"][0]["record_id"]
    assert imported.metadata["execution"]["batch_fingerprint"] == batch_fingerprint(
        batch
    )
    before = snapshot(graph)
    assert store.snapshot.import_data(pot_id="p", payload=payload) == imported
    assert snapshot(graph) == before


@pytest.mark.parametrize("bad", ("2026-03-01", "2026-03-01T00:00:00", "not-a-time"))
def test_invalid_snapshot_time_has_no_data_receipt_or_revision_advance(graph, bad):
    store = backend(graph)
    graph.query("CREATE (:TemporalSentinel {value:'outside-pot'})")
    with pytest.raises(ValueError):
        store.snapshot.import_data(pot_id="p", payload=import_payload(bad))
    assert snapshot(graph)[:2] == ([], [])
    assert store.mutation.current_version("p") == 0
    assert snapshot(graph)[3] == []


def test_raw_occurrence_and_receipt_observation_defaults_are_distinct_and_replay(graph):
    store = backend(graph)
    occurred = datetime.fromisoformat("2026-03-01T05:30:00+05:30")
    received = datetime.fromisoformat("2026-03-01T02:00:00-04:00")
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:a", ("Service",)),
            EntityUpsert("service:b", ("Service",)),
        ],
        edge_upserts=[EdgeUpsert("DEPENDS_ON", "service:a", "service:b", {})],
    )
    context = ProvenanceContext(
        mutation_id="distinct-times",
        event_occurred_at=occurred,
        event_received_at=received,
    )
    result = store.mutation.apply(plan, expected_pot_id="p", provenance_context=context)
    row = store.claim_query.find_claims(ClaimQueryFilter(pot_id="p", as_of=POINT))[0]
    assert row.valid_at == occurred.astimezone(timezone.utc)
    assert row.observed_at == received.astimezone(timezone.utc)
    before = snapshot(graph)
    assert (
        store.mutation.apply(
            plan,
            expected_pot_id="p",
            provenance_context=replace(
                context, event_received_at=POINT + timedelta(days=3)
            ),
        )
        == result
    )
    assert snapshot(graph) == before


@pytest.mark.parametrize(
    "properties",
    (
        {"valid_at": "2026-03-01"},
        {"valid_at": "2026-03-01T00:00:00"},
        {"valid_at": "not-a-time"},
        {"valid_at": "2026-03-01T00:00:00Z", "valid_from": "2026-03-01T00:00:00+01:00"},
        {"valid_at": "2026-03-01T00:00:00Z", "valid_until": "2026-03-01T00:00:00Z"},
        {"occurred_at": "2026-03-01T00:00:00"},
    ),
)
def test_invalid_raw_claim_times_reject_without_receipt_or_revision_change(
    graph, properties
):
    store = backend(graph)
    entities = [
        EntityUpsert("service:a", ("Service",)),
        EntityUpsert("service:b", ("Service",)),
    ]
    store.mutation.apply(
        MutationBatch(entity_upserts=entities),
        expected_pot_id="p",
        provenance_context=ProvenanceContext(mutation_id="seed"),
    )
    before = snapshot(graph)
    with pytest.raises((ValueError, MutationBatchValidationError)):
        store.mutation.apply(
            MutationBatch(
                edge_upserts=[
                    EdgeUpsert("DEPENDS_ON", "service:a", "service:b", properties)
                ]
            ),
            expected_pot_id="p",
            provenance_context=ProvenanceContext(mutation_id="invalid-time"),
        )
    assert snapshot(graph) == before


def test_prepared_generated_claim_times_remain_frozen_until_execution_and_replay(graph):
    store = backend(graph)
    plan = MutationBatch(
        entity_upserts=[
            EntityUpsert("service:a", ("Service",)),
            EntityUpsert("service:b", ("Service",)),
        ],
        edge_upserts=[EdgeUpsert("DEPENDS_ON", "service:a", "service:b", {})],
    )
    anchor = POINT + timedelta(microseconds=123456)
    prepare_claim_assertions(
        plan,
        pot_id="p",
        mutation_id="frozen-time",
        anchor=anchor,
        query=store.claim_query,
        definition=store.definition,
    )
    context = ProvenanceContext(
        mutation_id="frozen-time",
        prepared=PreparedMutation(
            "p", "frozen-time", "frozen-request", batch_fingerprint(plan), 0, anchor
        ),
    )
    original = deepcopy(plan)
    result = store.mutation.compare_and_apply(
        plan,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=context,
    )
    props = snapshot(graph)[1][0][0]
    assert props["valid_at"] == props["valid_from"] == anchor.isoformat()
    assert props["observed_at"] == anchor.isoformat()
    assert plan == original
    before = snapshot(graph)
    assert (
        store.mutation.compare_and_apply(
            plan,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context,
        )
        == result
    )
    assert snapshot(graph) == before
