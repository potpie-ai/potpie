"""Canonical node label inference (edge patterns + reconciliation enrichment)."""

from __future__ import annotations

import pytest

from potpie_context_engine.application.services.reconciliation_validation import validate_reconciliation_plan
from potpie_context_engine.domain.canonical_label_inference import enrich_reconciliation_plan_entity_labels
from potpie_context_engine.domain.context_events import EventRef
from potpie_context_engine.domain.graph_mutations import EdgeUpsert, EntityUpsert
from potpie_context_engine.domain.ontology import (
    EDGE_ENDPOINT_INFERRED_LABELS,
    inferred_labels_for_episodic_edge_endpoint,
    is_canonical_entity_label,
)
from potpie_context_engine.domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.unit


def test_edge_endpoint_rules_cover_deployed_to() -> None:
    assert ("DEPLOYED_TO", "target") in EDGE_ENDPOINT_INFERRED_LABELS
    assert inferred_labels_for_episodic_edge_endpoint("deployed_to", "target") == (
        "Environment",
    )
    assert inferred_labels_for_episodic_edge_endpoint("DEPLOYED_TO", "source") == (
        "Service",
    )


def test_ambiguous_roles_return_empty() -> None:
    assert inferred_labels_for_episodic_edge_endpoint("OWNS", "target") == ()
    assert inferred_labels_for_episodic_edge_endpoint("FIXES", "source") == ()


def test_enrich_plan_adds_inferred_endpoint_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="test", pot_id="p1"),
        summary="t",
        entity_upserts=[
            EntityUpsert(
                entity_key="service:ledger",
                labels=("Service",),
                properties={"name": "ledger"},
            ),
            EntityUpsert(
                entity_key="environment:prod",
                labels=("Entity",),
                properties={},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPLOYED_TO",
                from_entity_key="service:ledger",
                to_entity_key="environment:prod",
                properties={},
            )
        ],
    )
    enrich_reconciliation_plan_entity_labels(plan)
    by_key = {e.entity_key: e for e in plan.entity_upserts}
    # Rule: DEPLOYED_TO → Service on source, Environment on target.
    assert "Service" in by_key["service:ledger"].labels
    assert "Environment" in by_key["environment:prod"].labels


def test_validate_reconciliation_runs_enrich_when_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="test", pot_id="p1"),
        summary="t",
        entity_upserts=[
            EntityUpsert(
                entity_key="service:ledger",
                labels=("Service",),
                properties={"name": "ledger"},
            ),
            EntityUpsert(
                entity_key="environment:prod",
                labels=("Entity",),
                properties={},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DEPLOYED_TO",
                from_entity_key="service:ledger",
                to_entity_key="environment:prod",
                properties={},
            )
        ],
    )
    validate_reconciliation_plan(plan, "p1")
    by_key = {e.entity_key: e for e in plan.entity_upserts}
    assert "Environment" in by_key["environment:prod"].labels
    assert "Service" in by_key["service:ledger"].labels


def test_safe_label_guard() -> None:
    assert is_canonical_entity_label("Service") is True
    assert is_canonical_entity_label("NotAType") is False


def test_infer_labels_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")
    from potpie_context_engine.domain.reconciliation_flags import infer_canonical_labels_enabled

    assert infer_canonical_labels_enabled() is False


def test_infer_labels_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_INFER_LABELS", raising=False)
    from potpie_context_engine.domain.reconciliation_flags import infer_canonical_labels_enabled

    assert infer_canonical_labels_enabled() is True
