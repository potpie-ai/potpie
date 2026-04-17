"""Tests for graph quality, drift, and maintenance policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.graph_quality import (
    assess_graph_quality,
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
