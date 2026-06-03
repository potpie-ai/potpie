"""Tests for graph quality, drift, and maintenance policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.graph_quality import (
    EpisodicEdgeConflictInput,
    assess_graph_quality,
    detect_family_conflicts,
    freshness_ttl_hours_for_source_type,
    source_of_truth_for_source_type,
)
from domain.intelligence_models import CoverageReport
from domain.source_references import SourceReferenceRecord

pytestmark = pytest.mark.unit


def test_graph_quality_flags_stale_and_unverified_refs() -> None:
    report = assess_graph_quality(
        refs=[
            SourceReferenceRecord(
                ref="doc:old-runbook",
                source_type="Document",
                external_id="old-runbook",
                last_verified_at="2026-01-01T00:00:00Z",
                verification_state="unverified",
                freshness_ttl_hours=24,
            )
        ],
        coverage=CoverageReport(status="complete", available=["docs"]),
        fallbacks=[],
        now=datetime(2026, 4, 17, tzinfo=timezone.utc),
    )

    assert report.status == "watch"
    assert report.metrics["stale_ref_count"] == 1
    assert report.metrics["needs_verification_ref_count"] == 1
    assert {item["job"] for item in report.recommended_maintenance} == {
        "expire_stale_facts",
        "verify_entity",
    }


def test_graph_quality_degrades_on_source_access_gap() -> None:
    report = assess_graph_quality(
        refs=[
            SourceReferenceRecord(
                ref="alert:prod",
                source_type="Alert",
                external_id="prod",
                access="source_unreachable",
                sync_status="source_unreachable",
            )
        ],
        coverage=CoverageReport(status="partial", missing=["alerts"]),
        fallbacks=[],
        now=datetime(2026, 4, 17, tzinfo=timezone.utc),
    )

    assert report.status == "degraded"
    assert report.metrics["source_access_gap_count"] == 1
    assert any(issue.code == "source_access_gap" for issue in report.issues)
    assert any(
        item["job"] == "resync_source_scope" for item in report.recommended_maintenance
    )


def test_fact_family_policies_are_source_type_aware() -> None:
    assert freshness_ttl_hours_for_source_type(
        "Alert"
    ) < freshness_ttl_hours_for_source_type("Decision")
    assert source_of_truth_for_source_type("Code") == "authoritative_code_truth"
    assert source_of_truth_for_source_type("Preference") == "soft_inference"


def test_detect_family_conflicts_same_valid_at_is_contradiction() -> None:
    t = datetime(2025, 1, 15, tzinfo=timezone.utc)
    edges = [
        EpisodicEdgeConflictInput(
            uuid="e1",
            name="USES_DATA_STORE",
            source_uuid="svc",
            target_uuid="mongo",
            valid_at=t,
        ),
        EpisodicEdgeConflictInput(
            uuid="e2",
            name="STORED_IN",
            source_uuid="svc",
            target_uuid="pg",
            valid_at=t,
        ),
    ]
    out = detect_family_conflicts(edges)
    assert len(out) == 1
    assert out[0]["conflict_type"] == "contradiction"
    assert out[0]["auto_resolvable"] is False
    assert {out[0]["edge_a_uuid"], out[0]["edge_b_uuid"]} == {"e1", "e2"}


def test_detect_family_conflicts_chose_vs_migrated_with_datastore_hint() -> None:
    """``CHOSE`` joins datastore family when the target has ``DataStore`` (fix 02)."""
    t_old = datetime(2025, 1, 15, tzinfo=timezone.utc)
    t_new = datetime(2025, 8, 12, tzinfo=timezone.utc)
    edges = [
        EpisodicEdgeConflictInput(
            uuid="e-chose",
            name="CHOSE",
            source_uuid="ledger",
            target_uuid="mongo",
            valid_at=t_old,
            target_labels=("Entity", "DataStore"),
        ),
        EpisodicEdgeConflictInput(
            uuid="e-mig",
            name="MIGRATED_TO",
            source_uuid="ledger",
            target_uuid="pg",
            valid_at=t_new,
        ),
    ]
    out = detect_family_conflicts(edges)
    assert len(out) == 1
    assert out[0]["conflict_type"] == "supersession_pending"
    assert {out[0]["edge_a_uuid"], out[0]["edge_b_uuid"]} == {"e-chose", "e-mig"}


def test_detect_family_conflicts_distinct_valid_at_supersession_pending() -> None:
    t_old = datetime(2025, 1, 15, tzinfo=timezone.utc)
    t_new = datetime(2025, 8, 15, tzinfo=timezone.utc)
    edges = [
        EpisodicEdgeConflictInput(
            uuid="e1",
            name="USES_DATA_STORE",
            source_uuid="svc",
            target_uuid="mongo",
            valid_at=t_old,
        ),
        EpisodicEdgeConflictInput(
            uuid="e2",
            name="USES_DATA_STORE",
            source_uuid="svc",
            target_uuid="pg",
            valid_at=t_new,
        ),
    ]
    out = detect_family_conflicts(edges)
    assert len(out) == 1
    assert out[0]["conflict_type"] == "supersession_pending"
    assert out[0]["auto_resolvable"] is True
    assert out[0]["suggested_action"] == "supersede_older"
