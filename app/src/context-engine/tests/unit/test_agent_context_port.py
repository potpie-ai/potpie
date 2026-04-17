"""Comprehensive tests for domain/agent_context_port.py."""

from __future__ import annotations

import pytest

from domain.agent_context_port import (
    CONTEXT_INTENTS,
    CONTEXT_RECORD_TYPES,
    build_context_record_source_id,
    bundle_to_agent_envelope,
    context_port_manifest,
    context_recipe_for_intent,
    includes_for_request,
    normalize_context_intent,
    normalize_context_values,
    normalize_record_type,
    unsupported_include_values,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# normalize_context_intent
# ---------------------------------------------------------------------------


def test_normalize_context_intent_known_values() -> None:
    for intent in CONTEXT_INTENTS:
        assert normalize_context_intent(intent) == intent


def test_normalize_context_intent_none_returns_unknown() -> None:
    assert normalize_context_intent(None) == "unknown"


def test_normalize_context_intent_empty_returns_unknown() -> None:
    assert normalize_context_intent("") == "unknown"


def test_normalize_context_intent_strips_and_lowercases() -> None:
    assert normalize_context_intent("  DEBUGGING  ") == "debugging"


def test_normalize_context_intent_unrecognized_returns_unknown() -> None:
    assert normalize_context_intent("migration") == "unknown"


# ---------------------------------------------------------------------------
# normalize_context_values
# ---------------------------------------------------------------------------


def test_normalize_context_values_empty_list() -> None:
    assert normalize_context_values([]) == []


def test_normalize_context_values_none() -> None:
    assert normalize_context_values(None) == []


def test_normalize_context_values_deduplicates() -> None:
    result = normalize_context_values(["decisions", "decisions", "owners"])
    assert result == ["decisions", "owners"]


def test_normalize_context_values_strips_and_lowercases() -> None:
    result = normalize_context_values(["  Decisions  ", "OWNERS"])
    assert result == ["decisions", "owners"]


def test_normalize_context_values_skips_blank_entries() -> None:
    result = normalize_context_values(["decisions", "  ", "owners"])
    assert result == ["decisions", "owners"]


def test_normalize_context_values_preserves_order() -> None:
    result = normalize_context_values(["owners", "decisions", "docs"])
    assert result == ["owners", "decisions", "docs"]


# ---------------------------------------------------------------------------
# includes_for_request
# ---------------------------------------------------------------------------


def test_includes_for_request_explicit_overrides_intent_defaults() -> None:
    result = includes_for_request("feature", ["owners", "docs"], [])
    assert result == ["owners", "docs"]


def test_includes_for_request_empty_include_uses_intent_defaults() -> None:
    result = includes_for_request("feature", [], [])
    assert "purpose" in result
    assert "feature_map" in result


def test_includes_for_request_unknown_intent_returns_defaults() -> None:
    result = includes_for_request(None, [], [])
    assert "semantic_search" in result or "recent_changes" in result


def test_includes_for_request_exclude_removes_values() -> None:
    result = includes_for_request("feature", ["owners", "docs", "decisions"], ["docs"])
    assert "docs" not in result
    assert "owners" in result


def test_includes_for_request_exclude_from_intent_defaults() -> None:
    result = includes_for_request("debugging", [], ["owners"])
    assert "owners" not in result


def test_includes_for_request_exclude_all_leaves_empty() -> None:
    result = includes_for_request("docs", [], ["docs", "decisions", "source_status"])
    assert result == []


# ---------------------------------------------------------------------------
# unsupported_include_values
# ---------------------------------------------------------------------------


def test_unsupported_include_values_all_known() -> None:
    result = unsupported_include_values(["decisions", "owners", "docs"])
    assert result == []


def test_unsupported_include_values_detects_unknown() -> None:
    result = unsupported_include_values(["decisions", "spaceship_telemetry"])
    assert result == ["spaceship_telemetry"]


def test_unsupported_include_values_empty_list() -> None:
    assert unsupported_include_values([]) == []


# ---------------------------------------------------------------------------
# context_recipe_for_intent
# ---------------------------------------------------------------------------


def test_context_recipe_for_intent_all_named_intents() -> None:
    for intent in ("feature", "debugging", "review", "operations", "docs", "onboarding"):
        recipe = context_recipe_for_intent(intent)
        assert recipe["intent"] == intent
        assert "include" in recipe
        assert "mode" in recipe
        assert "source_policy" in recipe
        assert "when" in recipe


def test_context_recipe_for_intent_none_returns_unknown_recipe() -> None:
    recipe = context_recipe_for_intent(None)
    assert recipe["intent"] == "unknown"
    assert "include" in recipe


def test_context_recipe_for_intent_unrecognized_returns_unknown() -> None:
    recipe = context_recipe_for_intent("chaos_engineering")
    assert recipe["intent"] == "unknown"


def test_context_recipe_for_intent_returns_copy_not_reference() -> None:
    r1 = context_recipe_for_intent("feature")
    r2 = context_recipe_for_intent("feature")
    r1["include"].append("extra")
    assert "extra" not in r2["include"]


def test_context_recipe_for_intent_debugging_includes_prior_fixes() -> None:
    recipe = context_recipe_for_intent("debugging")
    assert "prior_fixes" in recipe["include"]


def test_context_recipe_for_intent_operations_uses_balanced_mode() -> None:
    recipe = context_recipe_for_intent("operations")
    assert recipe["mode"] == "balanced"


# ---------------------------------------------------------------------------
# normalize_record_type
# ---------------------------------------------------------------------------


def test_normalize_record_type_all_valid() -> None:
    for rt in CONTEXT_RECORD_TYPES:
        assert normalize_record_type(rt) == rt


def test_normalize_record_type_case_insensitive() -> None:
    assert normalize_record_type("Decision") == "decision"
    assert normalize_record_type("FIX") == "fix"


def test_normalize_record_type_strips_whitespace() -> None:
    assert normalize_record_type("  fix  ") == "fix"


def test_normalize_record_type_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported context record type"):
        normalize_record_type("thought")


def test_normalize_record_type_error_message_lists_allowed() -> None:
    with pytest.raises(ValueError) as exc_info:
        normalize_record_type("bad_type")
    assert "decision" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_context_record_source_id
# ---------------------------------------------------------------------------


def test_build_context_record_source_id_uses_idempotency_key() -> None:
    source_id = build_context_record_source_id(
        record_type="decision",
        summary="Use Postgres",
        scope={},
        source_refs=[],
        idempotency_key="my-custom-key",
    )
    assert source_id == "my-custom-key"


def test_build_context_record_source_id_strips_whitespace_from_key() -> None:
    source_id = build_context_record_source_id(
        record_type="decision",
        summary="s",
        scope={},
        source_refs=[],
        idempotency_key="  my-key  ",
    )
    assert source_id == "my-key"


def test_build_context_record_source_id_ignores_empty_idempotency_key() -> None:
    source_id = build_context_record_source_id(
        record_type="decision",
        summary="s",
        scope={},
        source_refs=[],
        idempotency_key="",
    )
    assert source_id.startswith("context_record:decision:")


def test_build_context_record_source_id_hash_format() -> None:
    source_id = build_context_record_source_id(
        record_type="fix",
        summary="Retry on timeout",
        scope={"repo_name": "org/repo"},
        source_refs=["github:pr:42"],
        idempotency_key=None,
    )
    assert source_id.startswith("context_record:fix:")
    suffix = source_id[len("context_record:fix:"):]
    assert len(suffix) == 24
    assert suffix.isalnum()


def test_build_context_record_source_id_is_deterministic() -> None:
    kwargs = dict(
        record_type="decision",
        summary="Pick Redis for queue",
        scope={"pot_id": "p1"},
        source_refs=["github:pr:10"],
        idempotency_key=None,
    )
    id1 = build_context_record_source_id(**kwargs)
    id2 = build_context_record_source_id(**kwargs)
    assert id1 == id2


def test_build_context_record_source_id_differs_for_different_inputs() -> None:
    base = dict(record_type="fix", summary="fix A", scope={}, source_refs=[], idempotency_key=None)
    id_a = build_context_record_source_id(**base)
    id_b = build_context_record_source_id(
        record_type="fix",
        summary="fix B",
        scope={},
        source_refs=[],
        idempotency_key=None,
    )
    assert id_a != id_b


# ---------------------------------------------------------------------------
# bundle_to_agent_envelope
# ---------------------------------------------------------------------------


def _make_bundle(**kwargs):  # type: ignore[no-untyped-def]
    """Build a minimal IntelligenceBundle with overrides."""
    from dataclasses import fields
    from domain.intelligence_models import (
        CapabilitySet,
        ContextBudget,
        ContextResolutionRequest,
        CoverageReport,
        IntelligenceBundle,
        ResolutionMeta,
    )
    from domain.graph_quality import GraphQualityReport
    from domain.source_references import FreshnessReport

    req = ContextResolutionRequest(pot_id="p1", query="test")
    defaults = {
        "request": req,
        "semantic_hits": [],
        "artifacts": [],
        "changes": [],
        "decisions": [],
        "discussions": [],
        "ownership": [],
        "project_map": [],
        "debugging_memory": [],
        "source_refs": [],
        "coverage": CoverageReport(status="complete"),
        "freshness": FreshnessReport(status="unknown"),
        "quality": GraphQualityReport(
            status="unknown",
            issues=[],
            metrics={},
            recommended_maintenance=[],
            policy={},
        ),
        "fallbacks": [],
        "open_conflicts": [],
        "recommended_next_actions": [],
        "errors": [],
        "meta": ResolutionMeta(provider="Test"),
    }
    defaults.update(kwargs)
    return IntelligenceBundle(**defaults)


def test_bundle_to_agent_envelope_structure() -> None:
    bundle = _make_bundle()
    envelope = bundle_to_agent_envelope(bundle)
    required_keys = {
        "ok", "answer", "facts", "evidence", "source_refs",
        "confidence", "as_of", "open_conflicts", "coverage",
        "freshness", "quality", "verification_state", "fallbacks",
        "recommended_next_actions", "errors", "meta", "bundle",
    }
    assert required_keys.issubset(envelope.keys())


def test_bundle_to_agent_envelope_ok_is_true() -> None:
    envelope = bundle_to_agent_envelope(_make_bundle())
    assert envelope["ok"] is True


def test_bundle_to_agent_envelope_confidence_complete() -> None:
    from domain.intelligence_models import CoverageReport

    bundle = _make_bundle(coverage=CoverageReport(status="complete"))
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["confidence"] == pytest.approx(0.82)


def test_bundle_to_agent_envelope_confidence_partial() -> None:
    from domain.intelligence_models import CoverageReport

    bundle = _make_bundle(coverage=CoverageReport(status="partial"))
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["confidence"] == pytest.approx(0.55)


def test_bundle_to_agent_envelope_confidence_empty() -> None:
    from domain.intelligence_models import CoverageReport

    bundle = _make_bundle(coverage=CoverageReport(status="empty"))
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["confidence"] == pytest.approx(0.2)


def test_bundle_to_agent_envelope_verification_state_no_refs() -> None:
    bundle = _make_bundle(source_refs=[])
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["verification_state"] == "unknown"


def test_bundle_to_agent_envelope_verification_state_all_verified() -> None:
    from domain.source_references import SourceReferenceRecord

    refs = [
        SourceReferenceRecord(ref="a", source_type="github", verification_state="verified"),
        SourceReferenceRecord(ref="b", source_type="jira", verification_state="verified"),
    ]
    bundle = _make_bundle(source_refs=refs)
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["verification_state"] == "verified"


def test_bundle_to_agent_envelope_verification_state_has_failed() -> None:
    from domain.source_references import SourceReferenceRecord

    refs = [
        SourceReferenceRecord(ref="a", source_type="github", verification_state="verified"),
        SourceReferenceRecord(ref="b", source_type="github", verification_state="verification_failed"),
    ]
    bundle = _make_bundle(source_refs=refs)
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["verification_state"] == "verification_failed"


def test_bundle_to_agent_envelope_summary_empty_bundle() -> None:
    bundle = _make_bundle()
    envelope = bundle_to_agent_envelope(bundle)
    assert "No matching project context" in envelope["answer"]["summary"]


def test_bundle_to_agent_envelope_summary_with_decisions() -> None:
    from domain.intelligence_models import DecisionRecord

    bundle = _make_bundle(
        decisions=[DecisionRecord(decision="Use Postgres"), DecisionRecord(decision="Use Redis")]
    )
    envelope = bundle_to_agent_envelope(bundle)
    assert "2 decision" in envelope["answer"]["summary"]


def test_bundle_to_agent_envelope_evidence_combines_semantic_and_discussions() -> None:
    from domain.intelligence_models import DiscussionRecord

    bundle = _make_bundle(
        semantic_hits=[{"uuid": "s1", "name": "hit"}],
        discussions=[DiscussionRecord(source_ref="PR #1", summary="Review thread")],
    )
    envelope = bundle_to_agent_envelope(bundle)
    assert len(envelope["evidence"]) == 2


def test_bundle_to_agent_envelope_as_of_none_when_not_set() -> None:
    bundle = _make_bundle()
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["as_of"] is None


def test_bundle_to_agent_envelope_as_of_iso_when_set() -> None:
    from datetime import datetime, timezone
    from domain.intelligence_models import ContextResolutionRequest

    req = ContextResolutionRequest(
        pot_id="p1",
        query="q",
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    bundle = _make_bundle(request=req)
    envelope = bundle_to_agent_envelope(bundle)
    assert "2024-06-01" in envelope["as_of"]


# ---------------------------------------------------------------------------
# context_port_manifest
# ---------------------------------------------------------------------------


def test_context_port_manifest_has_four_tools() -> None:
    manifest = context_port_manifest()
    assert set(manifest["tools"].keys()) == {
        "context_resolve",
        "context_search",
        "context_record",
        "context_status",
    }


def test_context_port_manifest_has_recipes_for_core_intents() -> None:
    manifest = context_port_manifest()
    for intent in ("feature", "debugging", "review", "operations"):
        assert intent in manifest["recipes"]


def test_context_port_manifest_has_rules() -> None:
    manifest = context_port_manifest()
    assert isinstance(manifest["rules"], list)
    assert len(manifest["rules"]) >= 5
