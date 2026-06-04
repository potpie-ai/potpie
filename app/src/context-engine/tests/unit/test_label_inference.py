"""Canonical node label inference (edge patterns + reconciliation enrichment)."""

from __future__ import annotations

import pytest

from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from domain.canonical_label_inference import enrich_reconciliation_plan_entity_labels
from domain.context_events import EventRef
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ontology import (
    EDGE_ENDPOINT_INFERRED_LABELS,
    inferred_labels_for_episodic_edge_endpoint,
    is_canonical_entity_label,
)
from domain.reconciliation import ReconciliationPlan

pytestmark = pytest.mark.unit


def test_edge_endpoint_rules_cover_decides_for_target() -> None:
    assert ("DECIDES_FOR", "target") in EDGE_ENDPOINT_INFERRED_LABELS
    assert inferred_labels_for_episodic_edge_endpoint("decides_for", "target") == ("Decision",)
    assert inferred_labels_for_episodic_edge_endpoint("DECIDES_FOR", "source") == ()


def test_ambiguous_roles_return_empty() -> None:
    assert inferred_labels_for_episodic_edge_endpoint("OWNS", "target") == ()
    assert inferred_labels_for_episodic_edge_endpoint("FIXES", "source") == ()


def test_enrich_plan_adds_decision_label_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="test", pot_id="p1"),
        summary="t",
        episodes=[],
        entity_upserts=[
            EntityUpsert(
                entity_key="k-scope",
                labels=("Service",),
                properties={"name": "ledger", "criticality": "high", "lifecycle_state": "active"},
            ),
            EntityUpsert(
                entity_key="k-adr",
                labels=("Document",),
                properties={"title": "ADR 1", "source_uri": "https://example.com/a"},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DECIDES_FOR",
                from_entity_key="k-adr",
                to_entity_key="k-scope",
                properties={},
            )
        ],
    )
    enrich_reconciliation_plan_entity_labels(plan)
    by_key = {e.entity_key: e for e in plan.entity_upserts}
    # Rule: DECIDES_FOR → label on target endpoint (see 03-canonical-node-labels.md).
    assert "Document" in by_key["k-adr"].labels
    assert "Decision" in by_key["k-scope"].labels
    assert "Service" in by_key["k-scope"].labels
    assert by_key["k-scope"].properties.get("title") == "k-scope"
    assert by_key["k-scope"].properties.get("summary") == ""
    assert by_key["k-scope"].properties.get("status") == "unknown"


def test_validate_reconciliation_runs_enrich_when_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="test", pot_id="p1"),
        summary="t",
        episodes=[],
        entity_upserts=[
            EntityUpsert(
                entity_key="k-scope",
                labels=("Service",),
                properties={"name": "ledger", "criticality": "high", "lifecycle_state": "active"},
            ),
            EntityUpsert(
                entity_key="k-adr",
                labels=("Document",),
                properties={"title": "ADR 1", "source_uri": "https://example.com/a"},
            ),
        ],
        edge_upserts=[
            EdgeUpsert(
                edge_type="DECIDES_FOR",
                from_entity_key="k-adr",
                to_entity_key="k-scope",
                properties={},
            )
        ],
    )
    validate_reconciliation_plan(plan, "p1")
    by_key = {e.entity_key: e for e in plan.entity_upserts}
    assert "Decision" in by_key["k-scope"].labels
    assert "Service" in by_key["k-scope"].labels


def test_safe_label_guard() -> None:
    assert is_canonical_entity_label("Decision") is True
    assert is_canonical_entity_label("NotAType") is False


def test_infer_labels_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")
    from domain.reconciliation_flags import infer_canonical_labels_enabled

    assert infer_canonical_labels_enabled() is False


def test_infer_labels_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_ENGINE_INFER_LABELS", raising=False)
    from domain.reconciliation_flags import infer_canonical_labels_enabled

    assert infer_canonical_labels_enabled() is True
