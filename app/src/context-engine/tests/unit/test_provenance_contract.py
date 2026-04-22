"""The 13-field provenance contract: construction → apply → surface."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from application.use_cases.apply_reconciliation_plan import apply_reconciliation_plan
from application.use_cases.query_context import search_pot_context
from domain.context_events import EventRef
from domain.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    ProvenanceContext,
    ProvenanceRef,
)
from domain.reconciliation import ReconciliationPlan


def test_provenance_ref_carries_full_13_field_contract() -> None:
    occurred = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    received = datetime(2026, 4, 1, 12, 1, tzinfo=timezone.utc)
    written = datetime(2026, 4, 1, 12, 2, tzinfo=timezone.utc)
    ref = ProvenanceRef(
        pot_id="pot-1",
        source_event_id="evt-1",
        episode_uuid="ep-1",
        source_system="github",
        source_kind="pull_request",
        source_ref="github/pr/42",
        event_occurred_at=occurred,
        event_received_at=received,
        graph_updated_at=written,
        valid_from=occurred,
        valid_to=None,
        confidence=0.92,
        created_by_agent="ingestion-agent-v3",
        reconciliation_run_id="run-xyz",
    )
    props = ref.to_properties()
    # Every non-None field surfaces as a prov_* property.
    assert props["prov_pot_id"] == "pot-1"
    assert props["prov_source_event_id"] == "evt-1"
    assert props["prov_episode_uuid"] == "ep-1"
    assert props["prov_source_system"] == "github"
    assert props["prov_source_kind"] == "pull_request"
    assert props["prov_source_ref"] == "github/pr/42"
    assert props["prov_event_occurred_at"] == occurred.isoformat()
    assert props["prov_event_received_at"] == received.isoformat()
    assert props["prov_graph_updated_at"] == written.isoformat()
    assert props["prov_valid_from"] == occurred.isoformat()
    assert props["prov_confidence"] == 0.92
    assert props["prov_created_by_agent"] == "ingestion-agent-v3"
    assert props["prov_reconciliation_run_id"] == "run-xyz"
    # valid_to is None, so it must be omitted (not emitted as null).
    assert "prov_valid_to" not in props


def test_provenance_ref_omits_none_fields() -> None:
    ref = ProvenanceRef(pot_id="p", source_event_id="e")
    props = ref.to_properties()
    assert props == {"prov_pot_id": "p", "prov_source_event_id": "e"}


def test_apply_reconciliation_plan_threads_full_provenance_to_applier() -> None:
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="evt-1", source_system="github", pot_id="pot-1"),
        summary="s",
        episodes=[],
        entity_upserts=[
            EntityUpsert(
                entity_key="e1",
                labels=("Entity", "SourceReference"),
                properties={
                    "external_id": "pr-42",
                    "ref_type": "pull_request",
                    "source_system": "github",
                },
            ),
            EntityUpsert(
                entity_key="e2",
                labels=("Entity", "SourceSystem"),
                properties={"name": "github", "source_type": "git_provider"},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="FROM_SOURCE",
                from_entity_key="e1",
                to_entity_key="e2",
                properties={},
            )
        ],
        confidence=0.77,
    )

    episodic = MagicMock()
    episodic.write_episode_drafts.return_value = ["ep-1"]
    structural = MagicMock()
    applier = MagicMock()
    applier.apply_entity_upserts.return_value = 1
    applier.apply_edge_upserts.return_value = 1
    applier.apply_edge_deletes.return_value = 0
    applier.apply_invalidations.return_value = 0

    ctx = ProvenanceContext(
        source_kind="pull_request",
        source_ref="github/pr/42",
        event_occurred_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        event_received_at=datetime(2026, 4, 1, 0, 1, tzinfo=timezone.utc),
        created_by_agent="ingestion-agent",
        reconciliation_run_id="run-1",
    )

    apply_reconciliation_plan(
        episodic,
        structural,
        plan,
        expected_pot_id="pot-1",
        mutation_applier=applier,
        provenance_context=ctx,
    )

    # Entity upsert received a fully populated ProvenanceRef.
    call = applier.apply_entity_upserts.call_args
    prov: ProvenanceRef = call.args[2]
    assert prov.pot_id == "pot-1"
    assert prov.source_event_id == "evt-1"
    assert prov.source_system == "github"
    assert prov.source_kind == "pull_request"
    assert prov.source_ref == "github/pr/42"
    assert prov.event_occurred_at == ctx.event_occurred_at
    assert prov.event_received_at == ctx.event_received_at
    assert prov.graph_updated_at is not None
    assert prov.valid_from == ctx.event_occurred_at
    assert prov.confidence == 0.77
    assert prov.created_by_agent == "ingestion-agent"
    assert prov.reconciliation_run_id == "run-1"
    assert prov.episode_uuid == "ep-1"


def test_apply_reconciliation_plan_works_without_provenance_context() -> None:
    """Callers that don't yet thread context still get a valid ProvenanceRef."""
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="evt-2", source_system="github", pot_id="pot-2"),
        summary="s",
        episodes=[],
        entity_upserts=[
            EntityUpsert(
                entity_key="e1",
                labels=("Entity", "SourceReference"),
                properties={
                    "external_id": "pr-42",
                    "ref_type": "pull_request",
                    "source_system": "github",
                },
            )
        ],
    )
    episodic = MagicMock()
    episodic.write_episode_drafts.return_value = []
    applier = MagicMock()
    applier.apply_entity_upserts.return_value = 1
    applier.apply_edge_upserts.return_value = 0
    applier.apply_edge_deletes.return_value = 0
    applier.apply_invalidations.return_value = 0

    apply_reconciliation_plan(
        episodic,
        MagicMock(),
        plan,
        expected_pot_id="pot-2",
        mutation_applier=applier,
    )
    prov = applier.apply_entity_upserts.call_args.args[2]
    assert prov.pot_id == "pot-2"
    assert prov.source_event_id == "evt-2"
    assert prov.source_system == "github"
    assert prov.graph_updated_at is not None
    # Optional fields remain None and must be absent from properties.
    assert "prov_source_ref" not in prov.to_properties()
    assert "prov_reconciliation_run_id" not in prov.to_properties()


def test_search_row_exposes_provenance_from_edge_attributes() -> None:
    edge = MagicMock()
    edge.uuid = "edge-uuid-1"
    edge.name = "Decision"
    edge.summary = "Use Postgres"
    edge.fact = "Use Postgres"
    edge.source_node_uuid = "a"
    edge.target_node_uuid = "b"
    edge.attributes = {
        "source_refs": ["adr-0042"],
        "episode_uuid": "ep-uuid",
        "_context_similarity_score": 0.9,
        "prov_pot_id": "pot-1",
        "prov_source_event_id": "evt-1",
        "prov_source_system": "github",
        "prov_source_kind": "pull_request",
        "prov_source_ref": "github/pr/42",
        "prov_event_occurred_at": "2026-04-01T12:00:00+00:00",
        "prov_graph_updated_at": "2026-04-01T12:05:00+00:00",
        "prov_confidence": 0.9,
        "prov_created_by_agent": "ingestion-agent",
        "prov_reconciliation_run_id": "run-1",
    }
    edge.created_at = None
    edge.valid_at = None
    edge.invalid_at = None

    episodic = MagicMock()
    episodic.enabled = True
    episodic.search.return_value = [edge]

    rows = search_pot_context(episodic, "pot-1", "q", limit=3)
    assert len(rows) == 1
    prov = rows[0]["provenance"]
    assert prov["pot_id"] == "pot-1"
    assert prov["source_event_id"] == "evt-1"
    assert prov["source_system"] == "github"
    assert prov["source_kind"] == "pull_request"
    assert prov["source_ref"] == "github/pr/42"
    assert prov["event_occurred_at"] == "2026-04-01T12:00:00+00:00"
    assert prov["graph_updated_at"] == "2026-04-01T12:05:00+00:00"
    assert prov["confidence"] == 0.9
    assert prov["created_by_agent"] == "ingestion-agent"
    assert prov["reconciliation_run_id"] == "run-1"


def test_search_row_has_no_provenance_key_when_attributes_have_no_prov_fields() -> None:
    edge = MagicMock()
    edge.uuid = "u"
    edge.name = "x"
    edge.summary = None
    edge.fact = None
    edge.source_node_uuid = None
    edge.target_node_uuid = None
    edge.attributes = {"source_refs": ["a"]}
    edge.created_at = None
    edge.valid_at = None
    edge.invalid_at = None
    episodic = MagicMock()
    episodic.enabled = True
    episodic.search.return_value = [edge]
    rows = search_pot_context(episodic, "pot-1", "q", limit=3)
    assert "provenance" not in rows[0]


def test_freshness_report_distinguishes_source_event_from_graph_update() -> None:
    from domain.source_references import FreshnessReport

    r = FreshnessReport(
        status="known",
        last_graph_update="2026-04-22T10:00:00+00:00",
        last_source_event_at="2026-04-22T09:30:00+00:00",
        last_source_verification="2026-04-22T09:45:00+00:00",
    )
    assert r.last_graph_update != r.last_source_event_at
    assert r.last_source_event_at != r.last_source_verification


def test_assess_freshness_picks_max_source_event_from_evidence_rows() -> None:
    from domain.source_references import assess_freshness

    rows = [
        {"provenance": {"event_occurred_at": "2026-04-20T10:00:00+00:00"}},
        {"provenance": {"event_occurred_at": "2026-04-22T09:30:00+00:00"}},
        {"provenance": {"event_occurred_at": "2026-04-21T12:00:00+00:00"}},
        {},  # no provenance → ignored
        {"provenance": {}},  # empty provenance → ignored
    ]
    r = assess_freshness([], evidence_rows=rows)
    # ``_latest_timestamp`` normalizes offsets, so we check prefix only.
    assert r.last_source_event_at is not None
    assert r.last_source_event_at.startswith("2026-04-22T09:30:00")


def test_assess_freshness_without_evidence_has_no_source_event() -> None:
    from domain.source_references import assess_freshness

    r = assess_freshness([])
    assert r.last_source_event_at is None


def test_context_graph_result_model_dump_keeps_provenance_nested() -> None:
    """HTTP route returns ``result.model_dump()``; ensure nested provenance
    survives the pydantic round-trip (not flattened into top-level keys).
    """
    from unittest.mock import patch

    from adapters.outbound.graphiti.context_graph import GraphitiContextGraphAdapter
    from domain.graph_query import ContextGraphQuery, ContextGraphStrategy

    rows_with_provenance = [
        {
            "uuid": "edge-1",
            "name": "Decision",
            "summary": "Use Postgres",
            "fact": None,
            "provenance": {
                "pot_id": "pot-1",
                "source_event_id": "evt-1",
                "source_system": "github",
                "source_kind": "pull_request",
                "event_occurred_at": "2026-04-22T09:30:00+00:00",
                "graph_updated_at": "2026-04-22T09:31:00+00:00",
                "confidence": 0.9,
                "created_by_agent": "ingestion-agent",
            },
        }
    ]

    episodic = MagicMock()
    episodic.enabled = True
    structural = MagicMock()
    adapter = GraphitiContextGraphAdapter(episodic=episodic, structural=structural)

    with patch(
        "adapters.outbound.graphiti.context_graph.search_pot_context",
        return_value=rows_with_provenance,
    ):
        result = adapter.query(
            ContextGraphQuery(
                pot_id="pot-1",
                query="auth",
                strategy=ContextGraphStrategy.SEMANTIC,
                include=["semantic_search"],
            )
        )

    serialized = result.model_dump()
    assert serialized["kind"] == "semantic_search"
    row = serialized["result"][0]
    assert isinstance(row["provenance"], dict)
    assert row["provenance"]["source_system"] == "github"
    assert row["provenance"]["event_occurred_at"] == "2026-04-22T09:30:00+00:00"
    # No flattened prov_* keys leaked into top-level row.
    assert not any(k.startswith("prov_") for k in row.keys())
