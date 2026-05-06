"""Source reference normalization, freshness assessment, and policy fallbacks."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.source_references import (
    FRESHNESS_STATES,
    RESOLVE_MODES,
    SOURCE_ACCESS_STATES,
    SOURCE_POLICIES,
    SOURCE_SYNC_STATES,
    VERIFICATION_STATES,
    FreshnessReport,
    SourceFallback,
    SourceReferenceRecord,
    assess_freshness,
    dedupe_source_references,
    normalize_resolve_mode,
    normalize_source_policy,
    source_policy_fallbacks,
    source_ref_key,
    source_reference_from_mapping,
    validate_source_reference_properties,
)

pytestmark = pytest.mark.unit


# --- Normalization ---------------------------------------------------------


class TestNormalizeResolveMode:
    @pytest.mark.parametrize("mode", list(RESOLVE_MODES))
    def test_known_modes_pass_through(self, mode: str) -> None:
        assert normalize_resolve_mode(mode) == mode

    def test_unknown_falls_back_to_fast(self) -> None:
        assert normalize_resolve_mode("weird") == "fast"

    def test_none_falls_back_to_fast(self) -> None:
        assert normalize_resolve_mode(None) == "fast"

    def test_strips_and_lowercases(self) -> None:
        assert normalize_resolve_mode("  DEEP  ") == "deep"


class TestNormalizeSourcePolicy:
    @pytest.mark.parametrize("policy", list(SOURCE_POLICIES))
    def test_known_policies_pass_through(self, policy: str) -> None:
        assert normalize_source_policy(policy) == policy

    def test_unknown_falls_back_to_references_only(self) -> None:
        assert normalize_source_policy("weird") == "references_only"

    def test_none_falls_back_to_references_only(self) -> None:
        assert normalize_source_policy(None) == "references_only"


# --- validate_source_reference_properties ----------------------------------


class TestValidateSourceReferenceProperties:
    def test_valid_iso_dates_no_errors(self) -> None:
        errors = validate_source_reference_properties(
            {"last_seen_at": "2026-04-27T00:00:00Z", "last_verified_at": "2026-04-27T00:00:00+00:00"}
        )
        assert errors == []

    def test_invalid_iso_date_errors(self) -> None:
        errors = validate_source_reference_properties({"last_seen_at": "not-a-date"})
        assert any("must be an ISO 8601 timestamp" in e for e in errors)

    def test_datetime_object_accepted(self) -> None:
        # ``_is_iso_datetime`` accepts datetime instances directly.
        assert validate_source_reference_properties(
            {"last_seen_at": datetime(2026, 4, 27, tzinfo=timezone.utc)}
        ) == []

    def test_invalid_access_state(self) -> None:
        errors = validate_source_reference_properties({"access": "weird"})
        assert any("access" in e for e in errors)

    def test_valid_access_state_passes(self) -> None:
        assert validate_source_reference_properties({"access": "allowed"}) == []

    def test_invalid_freshness(self) -> None:
        errors = validate_source_reference_properties({"freshness": "wrong"})
        assert any("freshness" in e for e in errors)

    def test_invalid_verification_state(self) -> None:
        errors = validate_source_reference_properties({"verification_state": "wrong"})
        assert any("verification_state" in e for e in errors)

    def test_invalid_sync_status(self) -> None:
        errors = validate_source_reference_properties({"sync_status": "wrong"})
        assert any("sync_status" in e for e in errors)

    def test_freshness_ttl_must_be_int(self) -> None:
        errors = validate_source_reference_properties({"freshness_ttl_hours": "abc"})
        assert any("must be an integer" in e for e in errors)

    def test_freshness_ttl_must_be_positive(self) -> None:
        errors = validate_source_reference_properties({"freshness_ttl_hours": 0})
        assert any("must be positive" in e for e in errors)

    def test_freshness_ttl_negative_errors(self) -> None:
        errors = validate_source_reference_properties({"freshness_ttl_hours": -1})
        assert any("must be positive" in e for e in errors)

    def test_freshness_ttl_string_int_accepted(self) -> None:
        assert validate_source_reference_properties({"freshness_ttl_hours": "24"}) == []

    def test_state_constants_are_disjoint_lists(self) -> None:
        # Sanity: the constants defined here must include canonical values.
        assert "fresh" in FRESHNESS_STATES
        assert "verified" in VERIFICATION_STATES
        assert "synced" in SOURCE_SYNC_STATES
        assert "allowed" in SOURCE_ACCESS_STATES


# --- source_ref_key --------------------------------------------------------


class TestSourceRefKey:
    def test_uses_source_system_first(self) -> None:
        assert source_ref_key("pr", "42", source_system="github") == "github:42"

    def test_falls_back_to_source_type(self) -> None:
        assert source_ref_key("pr", "42") == "pr:42"

    def test_uses_uri_when_no_external_id(self) -> None:
        assert source_ref_key("doc", None, uri="https://docs/x") == "doc:https://docs/x"

    def test_unknown_pair_returns_unknown_unknown(self) -> None:
        assert source_ref_key(None, None) == "unknown:unknown"

    def test_strips_whitespace(self) -> None:
        assert source_ref_key("  pr  ", " 42 ") == "pr:42"


# --- source_reference_from_mapping -----------------------------------------


class TestSourceReferenceFromMapping:
    def test_unknown_unknown_returns_none(self) -> None:
        # No fields → can't form a key → returns None.
        assert source_reference_from_mapping({}) is None

    def test_minimal_mapping(self) -> None:
        rec = source_reference_from_mapping(
            {"source_type": "pr", "external_id": "42"}
        )
        assert rec is not None
        assert rec.source_type == "pr"
        assert rec.external_id == "42"
        assert rec.ref == "pr:42"
        assert rec.fetchable is False

    def test_alias_keys_are_picked_up(self) -> None:
        rec = source_reference_from_mapping({"kind": "issue", "uuid": "12"})
        assert rec is not None
        assert rec.source_type == "issue"

    def test_uri_makes_fetchable(self) -> None:
        rec = source_reference_from_mapping(
            {"source_type": "doc", "external_id": "x", "url": "https://docs/x"}
        )
        assert rec is not None
        assert rec.fetchable is True
        assert rec.uri == "https://docs/x"
        assert rec.retrieval_uri == "https://docs/x"

    def test_invalid_state_strings_normalize_to_safe_defaults(self) -> None:
        rec = source_reference_from_mapping(
            {
                "source_type": "pr",
                "external_id": "1",
                "access": "weird",
                "freshness": "weird",
                "verification_state": "weird",
                "sync_status": "weird",
            }
        )
        assert rec is not None
        assert rec.access == "unknown"
        assert rec.freshness == "unknown"
        assert rec.verification_state == "unverified"
        assert rec.sync_status == "unknown"

    def test_freshness_ttl_int_extracted(self) -> None:
        rec = source_reference_from_mapping(
            {"source_type": "pr", "external_id": "1", "freshness_ttl_hours": "48"}
        )
        assert rec is not None
        assert rec.freshness_ttl_hours == 48

    def test_resolver_hint_only_includes_present_fields(self) -> None:
        rec = source_reference_from_mapping(
            {"source_type": "pr", "external_id": "1"}
        )
        assert rec is not None
        # ``source_system`` not provided → not in resolver_hint.
        assert "source_system" not in rec.resolver_hint
        assert rec.resolver_hint["source_type"] == "pr"


# --- dedupe_source_references ----------------------------------------------


class TestDedupeSourceReferences:
    def test_empty(self) -> None:
        assert dedupe_source_references([]) == []

    def test_dedupes_by_ref(self) -> None:
        a = SourceReferenceRecord(ref="a", source_type="pr")
        b = SourceReferenceRecord(ref="b", source_type="pr")
        a_dup = SourceReferenceRecord(ref="a", source_type="pr", title="dup")
        out = dedupe_source_references([a, b, a_dup])
        assert [r.ref for r in out] == ["a", "b"]
        # Preserves first occurrence.
        assert out[0].title is None


# --- assess_freshness ------------------------------------------------------


class TestAssessFreshness:
    def test_no_refs_returns_unknown(self) -> None:
        report = assess_freshness([])
        assert report.status == "unknown"

    def test_unreachable_dominates(self) -> None:
        refs = [
            SourceReferenceRecord(ref="a", source_type="pr", freshness="source_unreachable"),
            SourceReferenceRecord(ref="b", source_type="pr", freshness="stale"),
        ]
        report = assess_freshness(refs)
        assert report.status == "source_unreachable"

    def test_stale_outranks_needs_verification(self) -> None:
        refs = [
            SourceReferenceRecord(ref="a", source_type="pr", freshness="stale"),
            SourceReferenceRecord(ref="b", source_type="pr", freshness="needs_verification"),
        ]
        report = assess_freshness(refs)
        assert report.status == "stale"
        assert "a" in report.stale_refs

    def test_needs_verification_when_unverified(self) -> None:
        refs = [
            SourceReferenceRecord(
                ref="a",
                source_type="pr",
                freshness="fresh",
                verification_state="unverified",
            )
        ]
        report = assess_freshness(refs)
        assert report.status == "needs_verification"

    def test_fresh_when_verified_and_no_issues(self) -> None:
        refs = [
            SourceReferenceRecord(
                ref="a",
                source_type="pr",
                freshness="fresh",
                verification_state="verified",
                sync_status="synced",
                last_verified_at="2026-04-27T00:00:00Z",
            )
        ]
        report = assess_freshness(refs)
        assert report.status == "fresh"

    def test_evidence_rows_extract_latest_event_at(self) -> None:
        refs: list[SourceReferenceRecord] = []
        evidence = [
            {"provenance": {"event_occurred_at": "2026-04-26T00:00:00Z"}},
            {"provenance": {"event_occurred_at": "2026-04-27T00:00:00Z"}},
            {"no_provenance": True},
        ]
        report = assess_freshness(refs, evidence_rows=evidence)
        assert report.last_source_event_at is not None
        # Should pick the more recent.
        assert "2026-04-27" in report.last_source_event_at

    def test_evidence_rows_handle_datetime_values(self) -> None:
        evidence = [{"provenance": {"event_occurred_at": datetime(2026, 4, 27, tzinfo=timezone.utc)}}]
        report = assess_freshness([], evidence_rows=evidence)
        assert "2026-04-27" in (report.last_source_event_at or "")


# --- source_policy_fallbacks ----------------------------------------------


class TestSourcePolicyFallbacks:
    def test_references_only_returns_no_fallbacks(self) -> None:
        refs = [SourceReferenceRecord(ref="a", source_type="pr", fetchable=True)]
        assert source_policy_fallbacks(source_policy="references_only", refs=refs) == []

    def test_no_refs_under_summary_policy(self) -> None:
        out = source_policy_fallbacks(source_policy="summary", refs=[])
        assert any(f.code == "no_source_references" for f in out)

    def test_no_fetchable_under_summary_policy(self) -> None:
        refs = [SourceReferenceRecord(ref="a", source_type="pr", fetchable=False)]
        out = source_policy_fallbacks(source_policy="summary", refs=refs)
        assert any(f.code == "source_resolver_unavailable" for f in out)

    def test_verify_policy_flags_unverified_each_ref(self) -> None:
        refs = [
            SourceReferenceRecord(
                ref="a",
                source_type="pr",
                fetchable=True,
                verification_state="unverified",
            ),
            SourceReferenceRecord(
                ref="b",
                source_type="pr",
                fetchable=True,
                verification_state="verified",
            ),
        ]
        out = source_policy_fallbacks(source_policy="verify", refs=refs)
        assert any(f.code == "source_unverified" and f.ref == "a" for f in out)
        # Verified ones don't generate fallbacks.
        assert not any(f.ref == "b" for f in out)

    def test_unknown_policy_falls_back_to_references_only(self) -> None:
        # Bad policy normalizes to ``references_only`` → no fallbacks.
        refs = [SourceReferenceRecord(ref="a", source_type="pr")]
        assert source_policy_fallbacks(source_policy="bogus", refs=refs) == []


def test_freshness_report_default_state() -> None:
    report = FreshnessReport()
    assert report.status == "unknown"
    assert report.stale_refs == []
    assert report.needs_verification_refs == []


def test_source_fallback_default_state() -> None:
    fb = SourceFallback(code="x", message="m")
    assert fb.impact is None
    assert fb.ref is None
