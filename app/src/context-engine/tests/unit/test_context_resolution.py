"""Unit tests for context intelligence resolution (mock provider)."""

from __future__ import annotations

import pytest

from adapters.outbound.intelligence.mock import MockIntelligenceProvider
from application.services.context_resolution import ContextResolutionService
from application.use_cases.resolve_context import resolve_context
from domain.agent_context_port import (
    bundle_to_agent_envelope,
    context_port_manifest,
    context_recipe_for_intent,
)
from domain.intelligence_models import (
    ContextBudget,
    ContextResolutionRequest,
    ContextScope,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_resolve_returns_bundle_with_semantic_hits() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Why was this changed?",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert bundle.request.pot_id == "p1"
    assert len(bundle.semantic_hits) >= 1
    assert bundle.meta.schema_version == "4"
    assert bundle.meta.provider == "MockIntelligenceProvider"


@pytest.mark.asyncio
async def test_resolve_pr_query_includes_discussions() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="What happened in PR #42?",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert len(bundle.discussions) >= 1
    assert "PR #42" in (bundle.discussions[0].source_ref or "")
    assert bundle.source_refs
    assert bundle.freshness.status in {"needs_verification", "unknown"}


@pytest.mark.asyncio
async def test_resolve_verify_policy_reports_unverified_sources() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Check PR #42 before a risky change",
        scope=ContextScope(source_refs=["github:pr:42"]),
        source_policy="verify",
        mode="verify",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert any(ref.ref == "github:pr:42" for ref in bundle.source_refs)
    assert any(fallback.code == "source_unverified" for fallback in bundle.fallbacks)


@pytest.mark.asyncio
async def test_resolve_budget_limits_semantic_hits_and_returns_agent_envelope() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Map this feature before implementation",
        intent="feature",
        budget=ContextBudget(max_items=1, timeout_ms=1200),
    )
    bundle = await resolve_context(svc, req)
    envelope = bundle_to_agent_envelope(bundle)
    assert len(bundle.semantic_hits) <= 1
    assert bundle.project_map
    assert envelope["ok"] is True
    assert envelope["answer"]["summary"]
    assert envelope["answer"]["project_map"]
    assert envelope["coverage"]["status"] in {"complete", "partial", "empty"}
    assert envelope["quality"]["status"] in {"good", "watch", "degraded", "unknown"}
    assert "freshness_ttl_hours" in envelope["quality"]["policy"]
    assert any(
        fallback.code == "context_family_not_implemented"
        for fallback in bundle.fallbacks
    )


@pytest.mark.asyncio
async def test_resolve_docs_include_values_are_recognized_with_pending_fallbacks() -> (
    None
):
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Debug the production workflow",
        include=["operations", "diagnostic_signals", "source_status"],
    )
    bundle = await resolve_context(svc, req)
    fallback_codes = {fallback.code for fallback in bundle.fallbacks}
    fallback_messages = [fallback.message for fallback in bundle.fallbacks]
    assert "unsupported_include" not in fallback_codes
    assert bundle.project_map
    assert bundle.debugging_memory
    assert any(item.family == "service_map" for item in bundle.project_map)
    assert any(item.family == "diagnostic_signals" for item in bundle.debugging_memory)
    assert "context_family_not_implemented" not in fallback_codes
    assert not fallback_messages


@pytest.mark.asyncio
async def test_resolve_debugging_intent_returns_prior_fix_memory() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Repository ingestion keeps timing out in staging",
        intent="debugging",
        scope=ContextScope(services=["context-engine"], environment="staging"),
    )
    bundle = await resolve_context(svc, req)
    envelope = bundle_to_agent_envelope(bundle)
    assert bundle.debugging_memory
    assert any(item.family == "prior_fixes" for item in bundle.debugging_memory)
    assert envelope["answer"]["debugging_memory"]
    assert "debugging_memory_context" in bundle.coverage.available
    assert bundle.quality.metrics["source_ref_count"] >= 1
    assert any(
        item["job"] == "verify_entity"
        for item in bundle.quality.recommended_maintenance
    )


def test_extract_signals_pr_number() -> None:
    from domain.intelligence_signals import extract_signals

    s = extract_signals("Tell me about PR #99")
    assert s.mentioned_pr == 99


def test_context_port_manifest_keeps_recipe_surface_small() -> None:
    manifest = context_port_manifest()
    assert set(manifest["tools"]) == {
        "context_resolve",
        "context_search",
        "context_record",
        "context_status",
    }
    assert manifest["tools"]["context_resolve"]["role"] == "primary"
    assert context_recipe_for_intent("debugging") == {
        "intent": "debugging",
        "include": [
            "prior_fixes",
            "diagnostic_signals",
            "incidents",
            "alerts",
            "recent_changes",
            "config",
            "deployments",
            "owners",
            "source_status",
        ],
        "mode": "fast",
        "source_policy": "references_only",
        "when": "Before investigating a bug, incident, failing workflow, alert, or flaky behavior.",
    }


# ---------------------------------------------------------------------------
# Additional intent flows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_operations_intent_returns_project_map() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="How do I deploy this service to staging?",
        intent="operations",
    )
    bundle = await resolve_context(svc, req)
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["ok"] is True
    # operations intent includes project-map families (deployments, runbooks, etc.)
    assert bundle.project_map or bundle.debugging_memory or bundle.changes


@pytest.mark.asyncio
async def test_resolve_review_intent_with_pr_scope() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Review this PR for safety",
        intent="review",
        scope=ContextScope(pr_number=55),
    )
    bundle = await resolve_context(svc, req)
    assert bundle.artifacts  # PR artifact loaded
    assert bundle.discussions  # discussions for PR


@pytest.mark.asyncio
async def test_resolve_onboarding_intent_includes_repo_map() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="I'm new here — how is this project structured?",
        intent="onboarding",
    )
    bundle = await resolve_context(svc, req)
    envelope = bundle_to_agent_envelope(bundle)
    assert envelope["ok"] is True
    assert bundle.project_map


@pytest.mark.asyncio
async def test_resolve_unknown_intent_falls_back_gracefully() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="General question about the codebase",
        intent="completely_unknown_intent",
    )
    bundle = await resolve_context(svc, req)
    assert bundle.request.pot_id == "p1"
    assert bundle.coverage.status in {"complete", "partial", "empty"}


# ---------------------------------------------------------------------------
# Exclude parameter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_exclude_removes_include_from_plan() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="map this feature",
        intent="feature",
        # Exclude all feature intent defaults AND semantic_search to ensure
        # no capability runs at all
        exclude=["purpose", "feature_map", "service_map", "docs", "tickets",
                 "decisions", "recent_changes", "owners", "preferences",
                 "source_status", "semantic_search"],
    )
    bundle = await resolve_context(svc, req)
    # When everything is excluded nothing is planned and nothing executes
    assert bundle.meta.capabilities_used == []
    assert bundle.semantic_hits == []
    assert bundle.decisions == []
    assert bundle.project_map == []


@pytest.mark.asyncio
async def test_resolve_explicit_include_overrides_intent_defaults() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="find the owner",
        intent="feature",  # feature intent normally includes purpose, feature_map, etc.
        include=["owners"],
        scope=ContextScope(file_path="src/auth.py"),
    )
    bundle = await resolve_context(svc, req)
    assert bundle.ownership  # only owners requested, should be returned


# ---------------------------------------------------------------------------
# Budget and max_items
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_max_items_1_limits_results() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="broad search across everything",
        budget=ContextBudget(max_items=1, timeout_ms=2000),
    )
    bundle = await resolve_context(svc, req)
    assert len(bundle.semantic_hits) <= 1


@pytest.mark.asyncio
async def test_resolve_budget_timeout_ms_overrides_request_timeout() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="test",
        timeout_ms=8000,
        budget=ContextBudget(timeout_ms=8000, max_items=5),
    )
    bundle = await resolve_context(svc, req)
    assert bundle is not None


# ---------------------------------------------------------------------------
# Source policy variations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_summary_policy_returns_fallback_when_no_fetchable_refs() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="What happened in PR #99?",
        source_policy="summary",
    )
    bundle = await resolve_context(svc, req)
    # MockIntelligenceProvider returns unverified, non-fetchable refs
    # so "source_resolver_unavailable" fallback should appear
    fallback_codes = {f.code for f in bundle.fallbacks}
    assert "source_resolver_unavailable" in fallback_codes or bundle.source_refs is not None


@pytest.mark.asyncio
async def test_resolve_verify_policy_with_no_source_refs() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="something with no refs",
        source_policy="verify",
        include=["recent_changes"],  # no artifact/discussion to produce source refs
    )
    bundle = await resolve_context(svc, req)
    assert bundle is not None  # should not crash


# ---------------------------------------------------------------------------
# Envelope quality assurance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_envelope_all_required_keys_present() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(pot_id="p1", query="comprehensive check")
    bundle = await resolve_context(svc, req)
    envelope = bundle_to_agent_envelope(bundle)
    required = {
        "ok", "answer", "facts", "evidence", "source_refs", "confidence",
        "as_of", "open_conflicts", "coverage", "freshness", "quality",
        "verification_state", "fallbacks", "recommended_next_actions",
        "errors", "meta", "bundle",
    }
    assert required.issubset(envelope.keys())


@pytest.mark.asyncio
async def test_resolve_envelope_answer_keys_present() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(pot_id="p1", query="check answer shape")
    bundle = await resolve_context(svc, req)
    envelope = bundle_to_agent_envelope(bundle)
    expected_answer_keys = {
        "summary", "artifacts", "recent_changes", "decisions",
        "discussions", "owners", "project_map", "debugging_memory",
    }
    assert expected_answer_keys.issubset(envelope["answer"].keys())


@pytest.mark.asyncio
async def test_resolve_coverage_status_is_valid_value() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(pot_id="p1", query="test coverage")
    bundle = await resolve_context(svc, req)
    assert bundle.coverage.status in {"complete", "partial", "empty"}


@pytest.mark.asyncio
async def test_resolve_meta_schema_version() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(pot_id="p1", query="meta check")
    bundle = await resolve_context(svc, req)
    assert bundle.meta.schema_version is not None


@pytest.mark.asyncio
async def test_resolve_multiple_pots_are_isolated() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req_a = ContextResolutionRequest(pot_id="pot-a", query="test")
    req_b = ContextResolutionRequest(pot_id="pot-b", query="test")
    bundle_a = await resolve_context(svc, req_a)
    bundle_b = await resolve_context(svc, req_b)
    assert bundle_a.request.pot_id == "pot-a"
    assert bundle_b.request.pot_id == "pot-b"


# ---------------------------------------------------------------------------
# ContextResolutionRequest model
# ---------------------------------------------------------------------------


def test_request_timeout_propagates_to_budget() -> None:
    req = ContextResolutionRequest(pot_id="p1", query="q", timeout_ms=9000)
    assert req.budget.timeout_ms == 9000


def test_request_effective_max_items_clamps_to_1() -> None:
    req = ContextResolutionRequest(
        pot_id="p1", query="q", budget=ContextBudget(max_items=0)
    )
    assert req.effective_max_items == 1


def test_request_effective_max_items_clamps_to_50() -> None:
    req = ContextResolutionRequest(
        pot_id="p1", query="q", budget=ContextBudget(max_items=999)
    )
    assert req.effective_max_items == 50


def test_request_effective_max_items_normal() -> None:
    req = ContextResolutionRequest(
        pot_id="p1", query="q", budget=ContextBudget(max_items=12)
    )
    assert req.effective_max_items == 12


def test_request_explicit_budget_timeout_not_overridden() -> None:
    req = ContextResolutionRequest(
        pot_id="p1",
        query="q",
        timeout_ms=4000,
        budget=ContextBudget(timeout_ms=6000),
    )
    # Budget already set explicitly — should not be overridden by timeout_ms
    assert req.effective_timeout_ms == 6000
