"""Graph quality scoring, freshness policy, and conflict detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from domain.graph_quality import (
    FACT_FAMILY_FRESHNESS_TTL_HOURS,
    MAINTENANCE_JOB_FAMILIES,
    SOURCE_OF_TRUTH_POLICIES,
    EpisodicEdgeConflictInput,
    GraphQualityIssue,
    assess_graph_quality,
    classify_predicate_family_pair,
    detect_family_conflicts,
    fact_family_for_source_type,
    freshness_ttl_hours_for_source_type,
    is_reference_stale,
    make_source_ref,
    source_of_truth_for_source_type,
    temporal_supersession_resolved_issue,
)
from domain.intelligence_models import CoverageReport
from domain.source_references import SourceReferenceRecord

pytestmark = pytest.mark.unit


# --- Family + policy lookup ------------------------------------------------


class TestFactFamilyForSourceType:
    @pytest.mark.parametrize(
        "raw,family",
        [
            ("PullRequest", "change"),
            ("PR", "change"),
            ("commit", "change"),
            ("issue", "change"),
            ("Ticket", "change"),
            ("DiagnosticSignal", "diagnosticsignal"),
            ("BugPattern", "bugpattern"),
            ("AgentInstruction", "agentinstruction"),
            ("ownership", "ownership"),  # direct mapping
            ("incident", "incident"),
            ("alert", "alert"),
            ("not-a-real-thing", "unknown"),
        ],
    )
    def test_classification(self, raw: str, family: str) -> None:
        assert fact_family_for_source_type(raw) == family

    def test_none_returns_unknown(self) -> None:
        assert fact_family_for_source_type(None) == "unknown"


class TestFreshnessTtlAndSourceOfTruth:
    def test_pr_uses_change_ttl(self) -> None:
        assert freshness_ttl_hours_for_source_type("PR") == FACT_FAMILY_FRESHNESS_TTL_HOURS["change"]

    def test_alert_has_short_ttl(self) -> None:
        assert freshness_ttl_hours_for_source_type("alert") == 2

    def test_unknown_falls_back(self) -> None:
        assert freshness_ttl_hours_for_source_type("foo") == FACT_FAMILY_FRESHNESS_TTL_HOURS["unknown"]

    def test_source_of_truth_for_pr(self) -> None:
        assert (
            source_of_truth_for_source_type("PR")
            == SOURCE_OF_TRUTH_POLICIES["change"]
        )

    def test_source_of_truth_for_decision(self) -> None:
        assert source_of_truth_for_source_type("decision") == "canonicalized_memory"


# --- Factory ---------------------------------------------------------------


class TestMakeSourceRef:
    def test_default_ttl_from_family(self) -> None:
        ref = make_source_ref("pr", ref="github:42", external_id="42")
        assert ref.freshness_ttl_hours == FACT_FAMILY_FRESHNESS_TTL_HOURS["change"]
        assert ref.ref == "github:42"

    def test_explicit_ttl_overrides_default(self) -> None:
        ref = make_source_ref("pr", ref="github:42", freshness_ttl_hours=999)
        assert ref.freshness_ttl_hours == 999


# --- is_reference_stale ----------------------------------------------------


class TestIsReferenceStale:
    def test_marked_stale(self) -> None:
        ref = SourceReferenceRecord(ref="a", source_type="pr", freshness="stale")
        assert is_reference_stale(ref) is True

    def test_sync_status_stale(self) -> None:
        ref = SourceReferenceRecord(ref="a", source_type="pr", sync_status="stale")
        assert is_reference_stale(ref) is True

    def test_no_timestamp_not_stale(self) -> None:
        ref = SourceReferenceRecord(ref="a", source_type="pr")
        assert is_reference_stale(ref) is False

    def test_recent_timestamp_not_stale(self) -> None:
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            last_seen_at=recent,
            freshness_ttl_hours=24,
        )
        assert is_reference_stale(ref, now=now) is False

    def test_old_timestamp_is_stale(self) -> None:
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=48)).isoformat().replace("+00:00", "Z")
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            last_seen_at=old,
            freshness_ttl_hours=24,
        )
        assert is_reference_stale(ref, now=now) is True

    def test_invalid_timestamp_not_stale(self) -> None:
        ref = SourceReferenceRecord(
            ref="a", source_type="pr", last_seen_at="not-a-date"
        )
        assert is_reference_stale(ref) is False

    def test_uses_family_ttl_when_no_explicit(self) -> None:
        # Alert family has 2-hour TTL.
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
        ref = SourceReferenceRecord(
            ref="a", source_type="alert", last_seen_at=old
        )
        assert is_reference_stale(ref, now=now) is True


# --- assess_graph_quality --------------------------------------------------


class TestAssessGraphQuality:
    def test_empty_returns_unknown_status(self) -> None:
        cov = CoverageReport(status="empty")
        report = assess_graph_quality(refs=[], coverage=cov, fallbacks=[])
        assert report.status == "unknown"
        assert report.metrics["source_ref_count"] == 0

    def test_good_status_with_verified_refs(self) -> None:
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            access="allowed",
            verification_state="verified",
            freshness="fresh",
        )
        cov = CoverageReport(status="complete", available=["x"])
        report = assess_graph_quality(refs=[ref], coverage=cov, fallbacks=[])
        assert report.status == "good"

    def test_degraded_when_source_access_gap(self) -> None:
        ref = SourceReferenceRecord(
            ref="a", source_type="pr", access="permission_denied"
        )
        cov = CoverageReport(status="complete")
        report = assess_graph_quality(refs=[ref], coverage=cov, fallbacks=[])
        assert report.status == "degraded"
        assert any(i.code == "source_access_gap" for i in report.issues)

    def test_degraded_when_verification_failed(self) -> None:
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            access="allowed",
            verification_state="verification_failed",
        )
        cov = CoverageReport(status="complete")
        report = assess_graph_quality(refs=[ref], coverage=cov, fallbacks=[])
        assert report.status == "degraded"

    def test_watch_when_stale(self) -> None:
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            access="allowed",
            freshness="stale",
            verification_state="verified",
        )
        cov = CoverageReport(status="complete")
        report = assess_graph_quality(refs=[ref], coverage=cov, fallbacks=[])
        assert report.status == "watch"
        assert any(i.code == "stale_facts" for i in report.issues)

    def test_watch_when_coverage_missing(self) -> None:
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            access="allowed",
            freshness="fresh",
            verification_state="verified",
        )
        cov = CoverageReport(status="partial", missing=["owners"])
        report = assess_graph_quality(refs=[ref], coverage=cov, fallbacks=[])
        assert report.status == "watch"
        assert any(i.code == "incomplete_coverage" for i in report.issues)

    def test_recommended_jobs_deduped(self) -> None:
        ref = SourceReferenceRecord(
            ref="a",
            source_type="pr",
            access="allowed",
            freshness="stale",
            verification_state="verified",
        )
        cov = CoverageReport(status="partial", missing=["owners"])
        report = assess_graph_quality(refs=[ref, ref], coverage=cov, fallbacks=[])
        # Even with two stale refs, the recommendation appears once.
        assert sum(
            1 for r in report.recommended_maintenance if r["job"] == "expire_stale_facts"
        ) == 1

    def test_metrics_are_populated(self) -> None:
        ref_stale = SourceReferenceRecord(
            ref="s",
            source_type="pr",
            freshness="stale",
            verification_state="verified",
        )
        ref_unverified = SourceReferenceRecord(
            ref="u",
            source_type="pr",
            freshness="fresh",
            verification_state="unverified",
        )
        cov = CoverageReport(status="partial", missing=["m"])
        report = assess_graph_quality(
            refs=[ref_stale, ref_unverified], coverage=cov, fallbacks=[]
        )
        assert report.metrics["stale_ref_count"] == 1
        assert report.metrics["needs_verification_ref_count"] == 1
        assert report.metrics["missing_coverage_count"] == 1


# --- Conflict classification ----------------------------------------------


class TestClassifyPredicateFamilyPair:
    def _input(self, uuid: str, ts: datetime | None) -> EpisodicEdgeConflictInput:
        return EpisodicEdgeConflictInput(
            uuid=uuid, name="DEPLOYED_TO", source_uuid="src", target_uuid=f"t-{uuid}", valid_at=ts
        )

    def test_same_time_yields_blocking_contradiction(self) -> None:
        ts = datetime(2026, 4, 27, tzinfo=timezone.utc)
        out = classify_predicate_family_pair(
            self._input("a", ts), self._input("b", ts), family="deployment_target", subject_uuid="s"
        )
        assert out is not None
        assert out["conflict_type"] == "contradiction"
        assert out["severity"] == "blocking"
        assert out["auto_resolvable"] is False

    def test_newer_after_older_is_supersession_pending(self) -> None:
        older = self._input("a", datetime(2026, 4, 27, tzinfo=timezone.utc))
        newer = self._input("b", datetime(2026, 4, 28, tzinfo=timezone.utc))
        out = classify_predicate_family_pair(
            older, newer, family="deployment_target", subject_uuid="s"
        )
        assert out is not None
        assert out["conflict_type"] == "supersession_pending"
        assert out["auto_resolvable"] is True

    def test_older_after_newer_returns_none(self) -> None:
        # If "older" is actually newer, it's not a conflict to record.
        older = self._input("a", datetime(2026, 4, 28, tzinfo=timezone.utc))
        newer = self._input("b", datetime(2026, 4, 27, tzinfo=timezone.utc))
        out = classify_predicate_family_pair(
            older, newer, family="deployment_target", subject_uuid="s"
        )
        assert out is None

    def test_both_missing_timestamps_yields_overlap(self) -> None:
        out = classify_predicate_family_pair(
            self._input("a", None), self._input("b", None),
            family="deployment_target", subject_uuid="s",
        )
        assert out is not None
        assert out["conflict_type"] == "overlap"

    def test_one_missing_timestamp_yields_overlap(self) -> None:
        out = classify_predicate_family_pair(
            self._input("a", None),
            self._input("b", datetime(2026, 4, 27, tzinfo=timezone.utc)),
            family="deployment_target",
            subject_uuid="s",
        )
        assert out is not None
        assert out["conflict_type"] == "overlap"


class TestDetectFamilyConflicts:
    def test_empty_input(self) -> None:
        assert detect_family_conflicts([]) == []

    def test_single_edge_no_conflict(self) -> None:
        edges = [
            EpisodicEdgeConflictInput(
                uuid="e1", name="DEPLOYED_TO", source_uuid="s", target_uuid="t1"
            )
        ]
        assert detect_family_conflicts(edges) == []

    def test_two_edges_same_subject_different_objects_yield_conflict(self) -> None:
        edges = [
            EpisodicEdgeConflictInput(
                uuid="e1", name="DEPLOYED_TO", source_uuid="s", target_uuid="t1",
                valid_at=datetime(2026, 4, 27, tzinfo=timezone.utc),
            ),
            EpisodicEdgeConflictInput(
                uuid="e2", name="DEPLOYED_TO", source_uuid="s", target_uuid="t2",
                valid_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
            ),
        ]
        out = detect_family_conflicts(edges)
        assert len(out) == 1
        assert out[0]["family"] == "deployment_target"
        assert out[0]["subject_uuid"] == "s"

    def test_unknown_family_skipped(self) -> None:
        edges = [
            EpisodicEdgeConflictInput(
                uuid="e1", name="WHATEVER_EDGE", source_uuid="s", target_uuid="t1"
            )
        ]
        assert detect_family_conflicts(edges) == []


# --- Misc helpers ----------------------------------------------------------


def test_temporal_supersession_resolved_issue_shape() -> None:
    issue = temporal_supersession_resolved_issue(
        group_id="pot1",
        superseded_edge_uuid="old",
        superseding_edge_uuid="new",
        predicate_family="deployment_target",
    )
    assert isinstance(issue, GraphQualityIssue)
    assert issue.code == "temporal_supersession_resolved"
    assert "deployment_target" in issue.message
    assert "old" in issue.refs
    assert "new" in issue.refs
    assert "pot1" in issue.refs


def test_maintenance_job_families_constant() -> None:
    assert "verify_entity" in MAINTENANCE_JOB_FAMILIES
    assert "expire_stale_facts" in MAINTENANCE_JOB_FAMILIES
    # The constants are exported as a tuple so they're immutable.
    assert isinstance(MAINTENANCE_JOB_FAMILIES, tuple)
