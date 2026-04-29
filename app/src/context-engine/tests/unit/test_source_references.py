"""Comprehensive tests for domain/source_references.py."""

from __future__ import annotations

import pytest

from domain.source_references import (
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


# ---------------------------------------------------------------------------
# normalize_resolve_mode
# ---------------------------------------------------------------------------


def test_normalize_resolve_mode_known_values() -> None:
    for mode in ("fast", "balanced", "deep", "verify"):
        assert normalize_resolve_mode(mode) == mode


def test_normalize_resolve_mode_none_defaults_to_fast() -> None:
    assert normalize_resolve_mode(None) == "fast"


def test_normalize_resolve_mode_empty_defaults_to_fast() -> None:
    assert normalize_resolve_mode("") == "fast"


def test_normalize_resolve_mode_unknown_defaults_to_fast() -> None:
    assert normalize_resolve_mode("turbo") == "fast"


def test_normalize_resolve_mode_strips_whitespace() -> None:
    assert normalize_resolve_mode("  deep  ") == "deep"


def test_normalize_resolve_mode_case_insensitive() -> None:
    assert normalize_resolve_mode("FAST") == "fast"
    assert normalize_resolve_mode("Deep") == "deep"


# ---------------------------------------------------------------------------
# normalize_source_policy
# ---------------------------------------------------------------------------


def test_normalize_source_policy_known_values() -> None:
    for policy in (
        "references_only",
        "summary",
        "verify",
        "snippets",
        "full_if_needed",
    ):
        assert normalize_source_policy(policy) == policy


def test_normalize_source_policy_none_defaults_to_references_only() -> None:
    assert normalize_source_policy(None) == "references_only"


def test_normalize_source_policy_unknown_defaults_to_references_only() -> None:
    assert normalize_source_policy("aggressive") == "references_only"


def test_normalize_source_policy_strips_and_lowercases() -> None:
    assert normalize_source_policy("  SUMMARY  ") == "summary"


# ---------------------------------------------------------------------------
# validate_source_reference_properties
# ---------------------------------------------------------------------------


def test_validate_source_reference_properties_empty_dict() -> None:
    assert validate_source_reference_properties({}) == []


def test_validate_source_reference_properties_valid_timestamps() -> None:
    errors = validate_source_reference_properties(
        {
            "last_seen_at": "2024-01-01T00:00:00Z",
            "last_verified_at": "2024-06-01T12:00:00+00:00",
        }
    )
    assert errors == []


def test_validate_source_reference_properties_invalid_timestamp() -> None:
    errors = validate_source_reference_properties({"last_seen_at": "not-a-date"})
    assert any("last_seen_at" in e for e in errors)


def test_validate_source_reference_properties_invalid_last_verified() -> None:
    errors = validate_source_reference_properties({"last_verified_at": "yesterday"})
    assert any("last_verified_at" in e for e in errors)


def test_validate_source_reference_properties_valid_access() -> None:
    for access in (
        "allowed",
        "unknown",
        "permission_denied",
        "source_unreachable",
        "missing",
    ):
        errors = validate_source_reference_properties({"access": access})
        assert errors == [], f"Expected no error for access={access}"


def test_validate_source_reference_properties_invalid_access() -> None:
    errors = validate_source_reference_properties({"access": "denied_by_policy"})
    assert any("access" in e for e in errors)


def test_validate_source_reference_properties_invalid_freshness() -> None:
    errors = validate_source_reference_properties({"freshness": "moldy"})
    assert any("freshness" in e for e in errors)


def test_validate_source_reference_properties_invalid_verification_state() -> None:
    errors = validate_source_reference_properties({"verification_state": "maybe"})
    assert any("verification_state" in e for e in errors)


def test_validate_source_reference_properties_invalid_sync_status() -> None:
    errors = validate_source_reference_properties({"sync_status": "in_flight"})
    assert any("sync_status" in e for e in errors)


def test_validate_source_reference_properties_ttl_must_be_positive() -> None:
    errors = validate_source_reference_properties({"freshness_ttl_hours": 0})
    assert any("freshness_ttl_hours" in e for e in errors)


def test_validate_source_reference_properties_ttl_negative() -> None:
    errors = validate_source_reference_properties({"freshness_ttl_hours": -5})
    assert any("freshness_ttl_hours" in e for e in errors)


def test_validate_source_reference_properties_ttl_non_integer() -> None:
    errors = validate_source_reference_properties({"freshness_ttl_hours": "three"})
    assert any("freshness_ttl_hours" in e for e in errors)


def test_validate_source_reference_properties_ttl_positive_is_valid() -> None:
    errors = validate_source_reference_properties({"freshness_ttl_hours": 24})
    assert errors == []


def test_validate_source_reference_properties_multiple_errors() -> None:
    errors = validate_source_reference_properties(
        {
            "access": "bad",
            "freshness": "bad",
            "verification_state": "bad",
        }
    )
    assert len(errors) == 3


# ---------------------------------------------------------------------------
# source_ref_key
# ---------------------------------------------------------------------------


def test_source_ref_key_with_all_fields() -> None:
    key = source_ref_key("github", "pr:42", source_system="github.com")
    assert key == "github.com:pr:42"


def test_source_ref_key_falls_back_to_source_type() -> None:
    key = source_ref_key("jira", "JIRA-99")
    assert key == "jira:JIRA-99"


def test_source_ref_key_uses_uri_when_no_external_id() -> None:
    key = source_ref_key("docs", None, uri="https://docs.example.com/page")
    assert key == "docs:https://docs.example.com/page"


def test_source_ref_key_all_none_returns_unknown_unknown() -> None:
    key = source_ref_key(None, None)
    assert key == "unknown:unknown"


def test_source_ref_key_strips_whitespace() -> None:
    key = source_ref_key("  github  ", "  pr:1  ")
    assert key == "github:pr:1"


# ---------------------------------------------------------------------------
# source_reference_from_mapping
# ---------------------------------------------------------------------------


def test_source_reference_from_mapping_minimal() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "github", "external_id": "pr:1"}
    )
    assert ref is not None
    assert ref.source_type == "github"
    assert ref.external_id == "pr:1"
    assert ref.ref == "github:pr:1"


def test_source_reference_from_mapping_uses_ref_field() -> None:
    ref = source_reference_from_mapping(
        {"ref": "explicit-ref-key", "source_type": "github", "external_id": "x"}
    )
    assert ref is not None
    assert ref.ref == "explicit-ref-key"


def test_source_reference_from_mapping_falls_back_to_ref_type() -> None:
    ref = source_reference_from_mapping({"ref_type": "linear", "external_id": "L-1"})
    assert ref is not None
    assert ref.source_type == "linear"


def test_source_reference_from_mapping_falls_back_to_kind() -> None:
    ref = source_reference_from_mapping({"kind": "doc", "external_id": "d-1"})
    assert ref is not None
    assert ref.source_type == "doc"


def test_source_reference_from_mapping_unknown_unknown_returns_none() -> None:
    result = source_reference_from_mapping({})
    assert result is None


def test_source_reference_from_mapping_invalid_access_falls_back_to_unknown() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "github", "external_id": "x", "access": "forbidden"}
    )
    assert ref is not None
    assert ref.access == "unknown"


def test_source_reference_from_mapping_invalid_freshness_falls_back() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "github", "external_id": "x", "freshness": "rotten"}
    )
    assert ref is not None
    assert ref.freshness == "unknown"


def test_source_reference_from_mapping_invalid_verification_falls_back() -> None:
    ref = source_reference_from_mapping(
        {
            "source_type": "github",
            "external_id": "x",
            "verification_state": "maybe",
        }
    )
    assert ref is not None
    assert ref.verification_state == "unverified"


def test_source_reference_from_mapping_fetchable_when_uri_present() -> None:
    ref = source_reference_from_mapping(
        {
            "source_type": "github",
            "external_id": "pr:1",
            "retrieval_uri": "https://github.com/org/repo/pull/1",
        }
    )
    assert ref is not None
    assert ref.fetchable is True


def test_source_reference_from_mapping_not_fetchable_without_uri() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "github", "external_id": "pr:1"}
    )
    assert ref is not None
    assert ref.fetchable is False


def test_source_reference_from_mapping_title_fallback_name_headline() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "doc", "external_id": "d1", "name": "Architecture ADR"}
    )
    assert ref is not None
    assert ref.title == "Architecture ADR"


def test_source_reference_from_mapping_resolver_hint_excludes_nulls() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "github", "external_id": "pr:1", "source_system": None}
    )
    assert ref is not None
    assert "source_system" not in ref.resolver_hint


def test_source_reference_from_mapping_provider_maps_to_source_system() -> None:
    ref = source_reference_from_mapping(
        {"source_type": "linear", "external_id": "L-5", "provider": "linear.app"}
    )
    assert ref is not None
    assert ref.source_system == "linear.app"


# ---------------------------------------------------------------------------
# dedupe_source_references
# ---------------------------------------------------------------------------


def _make_ref(ref: str) -> SourceReferenceRecord:
    return SourceReferenceRecord(ref=ref, source_type="github")


def test_dedupe_source_references_removes_duplicates() -> None:
    refs = [_make_ref("a"), _make_ref("b"), _make_ref("a")]
    out = dedupe_source_references(refs)
    assert [r.ref for r in out] == ["a", "b"]


def test_dedupe_source_references_empty() -> None:
    assert dedupe_source_references([]) == []


def test_dedupe_source_references_all_unique() -> None:
    refs = [_make_ref("x"), _make_ref("y"), _make_ref("z")]
    assert len(dedupe_source_references(refs)) == 3


def test_dedupe_source_references_preserves_first_occurrence() -> None:
    r1 = SourceReferenceRecord(ref="same", source_type="github", title="First")
    r2 = SourceReferenceRecord(ref="same", source_type="github", title="Second")
    out = dedupe_source_references([r1, r2])
    assert len(out) == 1
    assert out[0].title == "First"


# ---------------------------------------------------------------------------
# assess_freshness
# ---------------------------------------------------------------------------


def test_assess_freshness_empty_refs() -> None:
    report = assess_freshness([])
    assert report.status == "unknown"
    assert report.last_graph_update is None


def test_assess_freshness_all_fresh() -> None:
    refs = [
        SourceReferenceRecord(
            ref="a",
            source_type="github",
            freshness="fresh",
            verification_state="verified",
            sync_status="synced",
            last_verified_at="2024-06-01T00:00:00Z",
        )
    ]
    report = assess_freshness(refs)
    assert report.status == "fresh"
    assert report.stale_refs == []


def test_assess_freshness_stale_ref() -> None:
    refs = [SourceReferenceRecord(ref="a", source_type="github", freshness="stale")]
    report = assess_freshness(refs)
    assert report.status == "stale"
    assert "a" in report.stale_refs


def test_assess_freshness_source_unreachable_takes_priority() -> None:
    refs = [
        SourceReferenceRecord(ref="a", source_type="github", freshness="stale"),
        SourceReferenceRecord(
            ref="b", source_type="github", freshness="source_unreachable"
        ),
    ]
    report = assess_freshness(refs)
    assert report.status == "source_unreachable"


def test_assess_freshness_unreachable_via_sync_status() -> None:
    refs = [
        SourceReferenceRecord(
            ref="r", source_type="docs", sync_status="source_unreachable"
        )
    ]
    report = assess_freshness(refs)
    assert report.status == "source_unreachable"


def test_assess_freshness_unreachable_via_access() -> None:
    refs = [
        SourceReferenceRecord(ref="r", source_type="docs", access="source_unreachable")
    ]
    report = assess_freshness(refs)
    assert report.status == "source_unreachable"


def test_assess_freshness_needs_verification_when_unverified() -> None:
    refs = [
        SourceReferenceRecord(
            ref="r", source_type="github", verification_state="unverified"
        )
    ]
    report = assess_freshness(refs)
    assert report.status == "needs_verification"
    assert "r" in report.needs_verification_refs


def test_assess_freshness_latest_graph_update_is_most_recent() -> None:
    refs = [
        SourceReferenceRecord(
            ref="a",
            source_type="github",
            freshness="fresh",
            verification_state="verified",
            last_seen_at="2024-01-01T00:00:00Z",
            last_verified_at="2024-06-01T00:00:00Z",
        ),
        SourceReferenceRecord(
            ref="b",
            source_type="github",
            freshness="fresh",
            verification_state="verified",
            last_seen_at="2024-03-01T00:00:00Z",
            last_verified_at="2024-02-01T00:00:00Z",
        ),
    ]
    report = assess_freshness(refs)
    assert report.last_graph_update is not None
    assert (
        "2024-06" in report.last_graph_update or "2024-03" in report.last_graph_update
    )


def test_assess_freshness_mixed_fresh_and_needs_verification() -> None:
    refs = [
        SourceReferenceRecord(
            ref="a",
            source_type="github",
            freshness="fresh",
            verification_state="verified",
            last_verified_at="2024-06-01T00:00:00Z",
        ),
        SourceReferenceRecord(
            ref="b", source_type="docs", verification_state="needs_verification"
        ),
    ]
    report = assess_freshness(refs)
    assert report.status == "needs_verification"


# ---------------------------------------------------------------------------
# source_policy_fallbacks
# ---------------------------------------------------------------------------


def test_source_policy_fallbacks_references_only_no_fallbacks() -> None:
    refs = [_make_ref("a")]
    fallbacks = source_policy_fallbacks(source_policy="references_only", refs=refs)
    assert fallbacks == []


def test_source_policy_fallbacks_no_refs_with_summary_policy() -> None:
    fallbacks = source_policy_fallbacks(source_policy="summary", refs=[])
    assert len(fallbacks) == 1
    assert fallbacks[0].code == "no_source_references"


def test_source_policy_fallbacks_no_fetchable_refs() -> None:
    refs = [SourceReferenceRecord(ref="a", source_type="github", fetchable=False)]
    fallbacks = source_policy_fallbacks(source_policy="summary", refs=refs)
    assert any(f.code == "source_resolver_unavailable" for f in fallbacks)


def test_source_policy_fallbacks_verify_reports_unverified() -> None:
    refs = [
        SourceReferenceRecord(
            ref="a",
            source_type="github",
            verification_state="unverified",
            fetchable=True,
        )
    ]
    fallbacks = source_policy_fallbacks(source_policy="verify", refs=refs)
    assert any(f.code == "source_unverified" and f.ref == "a" for f in fallbacks)


def test_source_policy_fallbacks_verify_skips_verified_refs() -> None:
    refs = [
        SourceReferenceRecord(
            ref="a",
            source_type="github",
            verification_state="verified",
            fetchable=True,
        )
    ]
    fallbacks = source_policy_fallbacks(source_policy="verify", refs=refs)
    assert not any(f.code == "source_unverified" for f in fallbacks)


def test_source_policy_fallbacks_unknown_policy_falls_back_to_references_only() -> None:
    refs = [_make_ref("a")]
    fallbacks = source_policy_fallbacks(source_policy="nonexistent", refs=refs)
    assert fallbacks == []
