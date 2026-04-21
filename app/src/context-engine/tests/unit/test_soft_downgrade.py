"""Ontology soft-fail (CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL) downgrade behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from domain.context_events import EventRef
from domain.errors import ReconciliationPlanValidationError
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ontology import CANONICAL_EDGE_TYPES, CANONICAL_LABELS, EDGE_TYPES
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

pytestmark = pytest.mark.unit


def _ref() -> EventRef:
    return EventRef(event_id="evt-1", source_system="test", pot_id="p1")


def _episode() -> EpisodeDraft:
    return EpisodeDraft(
        name="ep",
        episode_body="body",
        source_description="test",
        reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_related_to_is_canonical() -> None:
    assert "RELATED_TO" in EDGE_TYPES
    assert "RELATED_TO" in CANONICAL_EDGE_TYPES


def test_canonical_labels_export() -> None:
    assert "Decision" in CANONICAL_LABELS


def test_soft_fail_coerces_adr_like_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")

    plan = ReconciliationPlan(
        event_ref=_ref(),
        summary="adr",
        episodes=[_episode()],
        entity_upserts=[
            EntityUpsert(
                entity_key="person:alice",
                labels=("Entity", "Person"),
                properties={"name": "Alice"},
            ),
            EntityUpsert(
                entity_key="adr:0042",
                labels=("Entity", "ADR", "Database", "Technology"),
                properties={
                    "title": "Migrate ledger",
                    "summary": "Move to new store",
                    "status": "recorded",
                },
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DECIDED_BY",
                from_entity_key="person:alice",
                to_entity_key="adr:0042",
                properties={},
            ),
        ],
    )
    validate_reconciliation_plan(plan, "p1")

    assert plan.ontology_downgrades
    kinds = {d["kind"] for d in plan.ontology_downgrades}
    assert "unknown_labels" in kinds
    assert "edge_type" in kinds

    edge = plan.edge_upserts[0]
    assert edge.edge_type == "RELATED_TO"
    assert edge.properties.get("original_edge_type") == "DECIDED_BY"
    assert edge.properties.get("confidence") == 0.3

    adr = next(e for e in plan.entity_upserts if e.entity_key == "adr:0042")
    assert "Document" in adr.labels
    assert adr.properties.get("title") == "Migrate ledger"


def test_soft_fail_coerces_decision_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", "1")
    plan = ReconciliationPlan(
        event_ref=_ref(),
        summary="d",
        episodes=[_episode()],
        entity_upserts=[
            EntityUpsert(
                entity_key="dec:1",
                labels=("Entity", "Decision"),
                properties={
                    "title": "t",
                    "summary": "s",
                    "status": "recorded",
                },
            ),
        ],
        edge_upserts=[],
    )
    validate_reconciliation_plan(plan, "p1")
    ent = plan.entity_upserts[0]
    assert ent.properties.get("status") == "unknown"
    assert any(d["kind"] == "lifecycle_status" for d in plan.ontology_downgrades)


def test_soft_fail_off_still_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")

    plan = ReconciliationPlan(
        event_ref=_ref(),
        summary="x",
        episodes=[_episode()],
        entity_upserts=[
            EntityUpsert(
                entity_key="e1",
                labels=("Entity", "ADR"),
                properties={"title": "t", "summary": "s", "status": "recorded"},
            ),
        ],
        edge_upserts=[],
    )
    with pytest.raises(ReconciliationPlanValidationError):
        validate_reconciliation_plan(plan, "p1")


def test_strict_overrides_soft_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_ONTOLOGY_STRICT", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")

    plan = ReconciliationPlan(
        event_ref=_ref(),
        summary="x",
        episodes=[_episode()],
        entity_upserts=[
            EntityUpsert(
                entity_key="e1",
                labels=("Entity", "ADR"),
                properties={"title": "t", "summary": "s", "status": "accepted"},
            ),
        ],
        edge_upserts=[],
    )
    with pytest.raises(ReconciliationPlanValidationError):
        validate_reconciliation_plan(plan, "p1")


def test_duplicate_entity_key_hard_error() -> None:
    plan = ReconciliationPlan(
        event_ref=_ref(),
        summary="dup",
        episodes=[_episode()],
        entity_upserts=[
            EntityUpsert(
                entity_key="same",
                labels=("Entity", "Person"),
                properties={"name": "a"},
            ),
            EntityUpsert(
                entity_key="same",
                labels=("Entity", "Person"),
                properties={"name": "b"},
            ),
        ],
    )
    with pytest.raises(ReconciliationPlanValidationError, match="duplicate entity_key"):
        validate_reconciliation_plan(plan, "p1")


def test_invalid_iso_temporal_hard_error() -> None:
    plan = ReconciliationPlan(
        event_ref=_ref(),
        summary="t",
        episodes=[_episode()],
        entity_upserts=[
            EntityUpsert(
                entity_key="p:1",
                labels=("Entity", "Person"),
                properties={"name": "x", "valid_at": "not-a-date"},
            ),
        ],
    )
    with pytest.raises(ReconciliationPlanValidationError, match="ISO 8601"):
        validate_reconciliation_plan(plan, "p1")
