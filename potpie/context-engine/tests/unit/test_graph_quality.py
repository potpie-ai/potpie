"""Tests for graph quality, drift, and maintenance policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from context_engine.domain.graph_quality import (
    EpisodicEdgeConflictInput,
    assess_graph_quality,
    detect_family_conflicts,
    freshness_ttl_hours_for_source_type,
    source_of_truth_for_source_type,
)
from context_engine.domain.graph_quality import CoverageReport
from context_engine.domain.source_references import SourceReferenceRecord

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
    # Topology facts (code-derived) are fresher-by-default than generic unknowns.
    assert freshness_ttl_hours_for_source_type(
        "Service"
    ) < freshness_ttl_hours_for_source_type("Mystery")
    # Code-derived topology / ownership families are authoritative-code truth.
    assert source_of_truth_for_source_type("Service") == "authoritative_code_truth"
    assert source_of_truth_for_source_type("Team") == "authoritative_code_truth"


def test_detect_family_conflicts_same_valid_at_is_contradiction() -> None:
    # Two OWNED_BY edges from one service to different teams, same valid_at
    # → owner_binding contradiction. owner_binding is the only mutually
    # exclusive family (a service has exactly one owner); multi-binding
    # families like USES accumulate and are not contradictions
    # (see test_detect_family_conflicts_ignores_non_exclusive_families).
    t = datetime(2025, 1, 15, tzinfo=timezone.utc)
    edges = [
        EpisodicEdgeConflictInput(
            uuid="e1",
            name="OWNED_BY",
            source_uuid="svc",
            target_uuid="team_a",
            valid_at=t,
        ),
        EpisodicEdgeConflictInput(
            uuid="e2",
            name="OWNED_BY",
            source_uuid="svc",
            target_uuid="team_b",
            valid_at=t,
        ),
    ]
    out = detect_family_conflicts(edges)
    assert len(out) == 1
    assert out[0]["conflict_type"] == "contradiction"
    assert out[0]["auto_resolvable"] is False
    assert {out[0]["edge_a_uuid"], out[0]["edge_b_uuid"]} == {"e1", "e2"}


def test_detect_family_conflicts_distinct_valid_at_supersession_pending() -> None:
    t_old = datetime(2025, 1, 15, tzinfo=timezone.utc)
    t_new = datetime(2025, 8, 15, tzinfo=timezone.utc)
    edges = [
        EpisodicEdgeConflictInput(
            uuid="e1",
            name="OWNED_BY",
            source_uuid="svc",
            target_uuid="team_a",
            valid_at=t_old,
        ),
        EpisodicEdgeConflictInput(
            uuid="e2",
            name="OWNED_BY",
            source_uuid="svc",
            target_uuid="team_b",
            valid_at=t_new,
        ),
    ]
    out = detect_family_conflicts(edges)
    assert len(out) == 1
    assert out[0]["conflict_type"] == "supersession_pending"
    assert out[0]["auto_resolvable"] is True
    assert out[0]["suggested_action"] == "supersede_older"


def test_detect_family_conflicts_ignores_non_exclusive_families() -> None:
    # USES (datastore_binding) is a multi-binding family — a service may use
    # several datastores at once, so two concurrent USES edges are NOT a
    # contradiction. Only mutually exclusive families (owner_binding) conflict.
    t = datetime(2025, 1, 15, tzinfo=timezone.utc)
    edges = [
        EpisodicEdgeConflictInput(
            uuid="e1",
            name="USES",
            source_uuid="svc",
            target_uuid="mongo",
            valid_at=t,
        ),
        EpisodicEdgeConflictInput(
            uuid="e2",
            name="USES",
            source_uuid="svc",
            target_uuid="pg",
            valid_at=t,
        ),
    ]
    assert detect_family_conflicts(edges) == []
