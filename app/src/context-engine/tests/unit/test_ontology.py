"""Unit tests for the canonical context graph ontology."""

from datetime import datetime, timezone

import pytest

from application.use_cases.reconciliation_validation import validate_reconciliation_plan
from domain.context_events import EventRef
from domain.errors import ReconciliationPlanValidationError
from domain.graph_mutations import EdgeUpsert, EntityUpsert
from domain.ontology import (
    EDGE_TYPES,
    ENTITY_TYPES,
    ONTOLOGY_VERSION,
    allowed_edge_types_between,
    validate_structural_mutations,
)
from domain.reconciliation import EpisodeDraft, ReconciliationPlan

pytestmark = pytest.mark.unit


def test_phase_one_catalog_contains_project_context_domains() -> None:
    assert ONTOLOGY_VERSION == "2026-04-phase-7"
    for label in (
        "Pot",
        "Repository",
        "Service",
        "Feature",
        "Functionality",
        "Deployment",
        "Incident",
        "BugPattern",
        "Fix",
        "SourceReference",
        "AgentInstruction",
        "LocalWorkflow",
        "Agent",
        "QualityIssue",
        "MaintenanceJob",
        "MaterializedAccessPath",
    ):
        assert label in ENTITY_TYPES

    for edge_type in (
        "SCOPES",
        "BACKED_BY",
        "IMPLEMENTS",
        "EVIDENCED_BY",
        "MATCHES_PATTERN",
        "RESOLVED",
        "HAS_SIGNAL",
        "INFORMS",
        "FLAGS",
        "REPAIRS",
        "MATERIALIZES",
    ):
        assert edge_type in EDGE_TYPES


def test_phase_five_debugging_memory_edges_are_valid() -> None:
    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="fix:timeout",
                labels=("Entity", "Fix"),
                properties={
                    "summary": "Increase ingestion timeout",
                    "fix_type": "configuration",
                },
            ),
            EntityUpsert(
                entity_key="pattern:timeout",
                labels=("Entity", "BugPattern"),
                properties={"summary": "Repository ingestion timeout"},
            ),
            EntityUpsert(
                entity_key="signal:timeout",
                labels=("Entity", "DiagnosticSignal"),
                properties={
                    "signal_type": "error_signature",
                    "summary": "Timeout while fetching repository metadata",
                },
            ),
        ],
        [
            EdgeUpsert("RESOLVED", "fix:timeout", "pattern:timeout", {}),
            EdgeUpsert("HAS_SIGNAL", "pattern:timeout", "signal:timeout", {}),
        ],
        [],
    )
    assert errors == []


def test_validates_canonical_entities_and_edges() -> None:
    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="service:api",
                labels=("Entity", "Service"),
                properties={
                    "name": "api",
                    "criticality": "high",
                    "lifecycle_state": "active",
                },
            ),
            EntityUpsert(
                entity_key="repo:potpie",
                labels=("Entity", "Repository"),
                properties={"name": "potpie", "provider": "github"},
            ),
        ],
        [EdgeUpsert("BACKED_BY", "service:api", "repo:potpie", {})],
        [],
    )
    assert errors == []


def test_rejects_unknown_labels_and_missing_required_properties() -> None:
    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="svc:bad", labels=("Entity", "Service"), properties={}
            )
        ],
        [],
        [],
    )
    assert any("missing required properties" in error for error in errors)

    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="x:1", labels=("Entity", "GitHubOnlyThing"), properties={}
            )
        ],
        [],
        [],
    )
    assert any("unknown canonical labels" in error for error in errors)
    assert any("at least one public canonical label" in error for error in errors)


def test_validates_source_reference_freshness_metadata() -> None:
    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="source-ref:bad",
                labels=("Entity", "SourceReference"),
                properties={
                    "source_system": "github",
                    "external_id": "pr:1",
                    "ref_type": "pull_request",
                    "last_seen_at": "not-a-date",
                    "access": "maybe",
                    "sync_status": "lost",
                    "freshness_ttl_hours": "never",
                    "verification_state": "guessing",
                },
            )
        ],
        [],
        [],
    )
    assert any("last_seen_at" in error for error in errors)
    assert any("access" in error for error in errors)
    assert any("sync_status" in error for error in errors)
    assert any("freshness_ttl_hours" in error for error in errors)
    assert any("verification_state" in error for error in errors)


def test_phase_seven_quality_drift_edges_are_valid() -> None:
    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="quality:stale-doc",
                labels=("Entity", "QualityIssue"),
                properties={
                    "code": "stale_facts",
                    "severity": "warning",
                    "status": "active",
                },
            ),
            EntityUpsert(
                entity_key="job:verify-docs",
                labels=("Entity", "MaintenanceJob"),
                properties={"job_type": "verify_entity", "status": "active"},
            ),
            EntityUpsert(
                entity_key="view:service-runbooks",
                labels=("Entity", "MaterializedAccessPath"),
                properties={
                    "name": "service to runbooks",
                    "pattern_type": "service_runbook_lookup",
                },
            ),
            EntityUpsert(
                entity_key="doc:runbook",
                labels=("Entity", "Document"),
                properties={"title": "Runbook", "source_uri": "repo://runbook.md"},
            ),
        ],
        [
            EdgeUpsert("FLAGS", "quality:stale-doc", "doc:runbook", {}),
            EdgeUpsert("REPAIRS", "job:verify-docs", "doc:runbook", {}),
            EdgeUpsert("MATERIALIZES", "view:service-runbooks", "doc:runbook", {}),
        ],
        [],
    )
    assert errors == []


def test_rejects_invalid_edge_endpoint_labels_when_known_in_batch() -> None:
    errors = validate_structural_mutations(
        [
            EntityUpsert(
                entity_key="feature:ctx",
                labels=("Entity", "Feature"),
                properties={"name": "Context"},
            ),
            EntityUpsert(
                entity_key="incident:1",
                labels=("Entity", "Incident"),
                properties={"title": "Outage", "severity": "sev2", "status": "open"},
            ),
        ],
        [EdgeUpsert("IMPLEMENTS", "incident:1", "feature:ctx", {})],
        [],
    )
    assert any("invalid endpoint labels" in error for error in errors)


def test_code_graph_labels_match_code_asset_bridge_edges() -> None:
    assert "TOUCHES_CODE" in allowed_edge_types_between(("Feature",), ("FILE",))


def test_reconciliation_plan_validation_uses_ontology_boundary() -> None:
    plan = ReconciliationPlan(
        event_ref=EventRef(event_id="e1", source_system="test", pot_id="p1"),
        summary="bad structural mutation",
        episodes=[
            EpisodeDraft(
                name="n",
                episode_body="b",
                source_description="test",
                reference_time=datetime(2026, 4, 17, tzinfo=timezone.utc),
            )
        ],
        entity_upserts=[
            EntityUpsert(entity_key="x:1", labels=("Entity",), properties={})
        ],
    )

    with pytest.raises(
        ReconciliationPlanValidationError, match="ontology validation failed"
    ):
        validate_reconciliation_plan(plan, "p1")
