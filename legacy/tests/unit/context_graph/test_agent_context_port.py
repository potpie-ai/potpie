"""Pure helpers in ``domain.agent_context_port``: include/intent normalization."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from domain.agent_context_port import (
    CONTEXT_INCLUDE_VALUES,
    CONTEXT_INTENTS,
    DEFAULT_INTENT_INCLUDES,
    build_context_record_source_id,
    bundle_to_agent_envelope,
    context_recipe_for_intent,
    context_port_manifest,
    includes_for_request,
    normalize_context_intent,
    normalize_context_values,
    normalize_record_type,
    unsupported_include_values,
)
from domain.intelligence_models import (
    ArtifactContext,
    ChangeRecord,
    ContextResolutionRequest,
    CoverageReport,
    DecisionRecord,
    IntelligenceBundle,
)
from domain.source_references import SourceReferenceRecord

pytestmark = pytest.mark.unit


class TestNormalizeContextValues:
    def test_none_returns_empty(self) -> None:
        assert normalize_context_values(None) == []

    def test_empty_returns_empty(self) -> None:
        assert normalize_context_values([]) == []

    def test_lowercases_and_strips(self) -> None:
        assert normalize_context_values(["  Foo  ", "BAR"]) == ["foo", "bar"]

    def test_dedupes_preserving_order(self) -> None:
        assert normalize_context_values(["a", "A", "b", "a"]) == ["a", "b"]

    def test_whitespace_only_dropped(self) -> None:
        assert normalize_context_values(["   ", "x"]) == ["x"]

    def test_accepts_tuple(self) -> None:
        assert normalize_context_values(("a", "b")) == ["a", "b"]


class TestNormalizeContextIntent:
    def test_known_intent_passes(self) -> None:
        # Pick any intent known to exist.
        assert normalize_context_intent("debugging") == "debugging"

    def test_lowercases_and_strips(self) -> None:
        assert normalize_context_intent("  Debugging  ") == "debugging"

    def test_unknown_falls_back_to_unknown(self) -> None:
        assert normalize_context_intent("not-a-real-intent") == "unknown"

    def test_none_falls_back_to_unknown(self) -> None:
        assert normalize_context_intent(None) == "unknown"

    def test_all_intent_keys_present(self) -> None:
        # Every intent in ``CONTEXT_INTENTS`` must have a default include list.
        for intent in CONTEXT_INTENTS:
            assert intent in DEFAULT_INTENT_INCLUDES or intent == "unknown"


class TestIncludesForRequest:
    def test_explicit_include_wins(self) -> None:
        out = includes_for_request("debugging", ["docs"], [])
        assert out == ["docs"]

    def test_default_used_when_no_include(self) -> None:
        out = includes_for_request("debugging", [], [])
        # Default debugging include set has these key items.
        assert "incidents" in out
        assert "alerts" in out

    def test_excludes_filter_includes(self) -> None:
        out = includes_for_request("debugging", ["docs", "alerts"], ["alerts"])
        assert "docs" in out
        assert "alerts" not in out

    def test_unknown_intent_uses_unknown_default(self) -> None:
        out = includes_for_request("not-real", [], [])
        assert out == list(DEFAULT_INTENT_INCLUDES["unknown"])

    def test_normalizes_case(self) -> None:
        out = includes_for_request("debugging", ["DOCS"], ["DOCS"])
        # Normalized "docs" exists and is also normalized in excludes → filtered.
        assert "docs" not in out


class TestUnsupportedIncludeValues:
    def test_returns_unknown_includes(self) -> None:
        out = unsupported_include_values(["docs", "not-real", "purpose"])
        assert "not-real" in out
        assert "docs" not in out
        assert "purpose" not in out

    def test_empty_returns_empty(self) -> None:
        assert unsupported_include_values([]) == []

    def test_all_canonical_returns_empty(self) -> None:
        # Pick a canonical subset.
        canonical = list(sorted(CONTEXT_INCLUDE_VALUES))[:3]
        assert unsupported_include_values(canonical) == []


class TestContextRecipeForIntent:
    def test_known_intent_returns_recipe_with_includes(self) -> None:
        recipe = context_recipe_for_intent("debugging")
        assert recipe["intent"] == "debugging"
        assert isinstance(recipe["include"], list)
        # Must be a fresh list, not a reference to internal state.
        recipe["include"].append("mutation")
        assert "mutation" not in DEFAULT_INTENT_INCLUDES.get("debugging", ())

    def test_unknown_intent_falls_back_to_unknown(self) -> None:
        recipe = context_recipe_for_intent("not-real")
        assert recipe["intent"] == "unknown"

    def test_recipe_has_required_keys(self) -> None:
        recipe = context_recipe_for_intent("feature")
        for key in ("intent", "include", "mode", "source_policy"):
            assert key in recipe


class TestContextPortManifest:
    def test_manifest_has_tools_section(self) -> None:
        manifest = context_port_manifest()
        assert "tools" in manifest
        assert "context_resolve" in manifest["tools"]


class TestNormalizeRecordType:
    def test_known_passes(self) -> None:
        # Pick a record type that's known. Try a likely candidate; fall back if missing.
        from domain.agent_context_port import CONTEXT_RECORD_TYPES
        any_known = next(iter(CONTEXT_RECORD_TYPES))
        assert normalize_record_type(any_known.upper()) == any_known

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError) as ei:
            normalize_record_type("not-a-real-record-type")
        assert "Unsupported context record type" in str(ei.value)


class TestBuildContextRecordSourceId:
    def test_idempotency_key_wins_when_set(self) -> None:
        out = build_context_record_source_id(
            record_type="decision",
            summary="anything",
            scope={},
            source_refs=[],
            idempotency_key="custom-key",
        )
        assert out == "custom-key"

    def test_strips_idempotency_key_whitespace(self) -> None:
        out = build_context_record_source_id(
            record_type="decision",
            summary="anything",
            scope={},
            source_refs=[],
            idempotency_key="  spaced  ",
        )
        assert out == "spaced"

    def test_deterministic_hash_when_no_idempotency_key(self) -> None:
        a = build_context_record_source_id(
            record_type="decision",
            summary="x",
            scope={"k": 1},
            source_refs=["a"],
            idempotency_key=None,
        )
        b = build_context_record_source_id(
            record_type="decision",
            summary="x",
            scope={"k": 1},
            source_refs=["a"],
            idempotency_key=None,
        )
        assert a == b
        assert a.startswith("context_record:decision:")

    def test_different_inputs_yield_different_ids(self) -> None:
        a = build_context_record_source_id(
            record_type="decision",
            summary="x",
            scope={},
            source_refs=[],
            idempotency_key=None,
        )
        b = build_context_record_source_id(
            record_type="decision",
            summary="y",  # changed
            scope={},
            source_refs=[],
            idempotency_key=None,
        )
        assert a != b

    def test_empty_idempotency_key_falls_back_to_hash(self) -> None:
        out = build_context_record_source_id(
            record_type="decision",
            summary="x",
            scope={},
            source_refs=[],
            idempotency_key="   ",
        )
        # Whitespace-only is treated as missing.
        assert out.startswith("context_record:decision:")


def _bundle(**kwargs) -> IntelligenceBundle:
    request = ContextResolutionRequest(pot_id="p", query="q")
    return IntelligenceBundle(request=request, **kwargs)


class TestBundleToAgentEnvelope:
    def test_empty_bundle_uses_fallback_summary(self) -> None:
        envelope = bundle_to_agent_envelope(_bundle())
        assert envelope["ok"] is True
        assert "No matching project context" in envelope["answer"]["summary"]
        # Confidence falls in the empty bucket.
        assert envelope["confidence"] == 0.2

    def test_complete_coverage_yields_high_confidence(self) -> None:
        bundle = _bundle(coverage=CoverageReport(status="complete"))
        envelope = bundle_to_agent_envelope(bundle)
        assert envelope["confidence"] == 0.82

    def test_partial_coverage_middle_confidence(self) -> None:
        bundle = _bundle(coverage=CoverageReport(status="partial"))
        envelope = bundle_to_agent_envelope(bundle)
        assert envelope["confidence"] == 0.55

    def test_explicit_summary_overrides_fallback(self) -> None:
        envelope = bundle_to_agent_envelope(_bundle(), answer_summary="hand-written")
        assert envelope["answer"]["summary"] == "hand-written"

    def test_summary_aggregates_counts(self) -> None:
        bundle = _bundle(
            artifacts=[ArtifactContext(kind="pr", identifier="1")],
            changes=[ChangeRecord()],
            decisions=[DecisionRecord(decision="d")],
        )
        envelope = bundle_to_agent_envelope(bundle)
        # Each non-empty list contributes a phrase.
        summary = envelope["answer"]["summary"]
        assert "1 artifact" in summary
        assert "1 recent change" in summary
        assert "1 decision" in summary

    def test_verification_state_unknown_when_no_refs(self) -> None:
        envelope = bundle_to_agent_envelope(_bundle())
        assert envelope["verification_state"] == "unknown"

    def test_verification_state_verified_when_all_verified(self) -> None:
        refs = [
            SourceReferenceRecord(ref="a", source_type="pr", verification_state="verified"),
            SourceReferenceRecord(ref="b", source_type="pr", verification_state="verified"),
        ]
        envelope = bundle_to_agent_envelope(_bundle(source_refs=refs))
        assert envelope["verification_state"] == "verified"

    def test_verification_state_failed_dominates(self) -> None:
        refs = [
            SourceReferenceRecord(ref="a", source_type="pr", verification_state="verified"),
            SourceReferenceRecord(ref="b", source_type="pr", verification_state="verification_failed"),
        ]
        envelope = bundle_to_agent_envelope(_bundle(source_refs=refs))
        assert envelope["verification_state"] == "verification_failed"

    def test_verification_needs_when_present(self) -> None:
        refs = [
            SourceReferenceRecord(ref="a", source_type="pr", verification_state="needs_verification"),
        ]
        envelope = bundle_to_agent_envelope(_bundle(source_refs=refs))
        assert envelope["verification_state"] == "needs_verification"

    def test_verification_unverified_default(self) -> None:
        refs = [
            SourceReferenceRecord(ref="a", source_type="pr", verification_state="unverified"),
        ]
        envelope = bundle_to_agent_envelope(_bundle(source_refs=refs))
        assert envelope["verification_state"] == "unverified"

    def test_as_of_serialized_when_present(self) -> None:
        request = ContextResolutionRequest(
            pot_id="p", query="q", as_of=datetime(2026, 4, 27, tzinfo=timezone.utc)
        )
        envelope = bundle_to_agent_envelope(IntelligenceBundle(request=request))
        assert envelope["as_of"] == "2026-04-27T00:00:00+00:00"

    def test_as_of_none_when_unset(self) -> None:
        envelope = bundle_to_agent_envelope(_bundle())
        assert envelope["as_of"] is None
