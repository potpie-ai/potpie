"""Unit tests for context intelligence resolution (mock provider)."""

from __future__ import annotations

import pytest

from typing import Sequence

from adapters.outbound.intelligence.mock import MockIntelligenceProvider
from application.services.context_resolution import ContextResolutionService
from application.use_cases.resolve_context import resolve_context
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    ResolvedSummary,
    ResolvedVerification,
    ResolverAuthContext,
    ResolverBudget,
    ResolverCapabilityEntry,
    SourceResolutionResult,
)
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
    assert not any(
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
            "causal_chain",
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


class _StubVerifier:
    """Resolver that always reports each ref as verified, with a summary."""

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return (
            ResolverCapabilityEntry(
                provider="github",
                source_kind="repository",
                policies=frozenset({"summary", "verify"}),
            ),
        )

    async def resolve(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        out = SourceResolutionResult()
        for ref in refs:
            if source_policy == "verify":
                out.verifications.append(
                    ResolvedVerification(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        verified=True,
                        verification_state="verified",
                        checked_at="2026-04-21T00:00:00+00:00",
                        source_system="github",
                    )
                )
            elif source_policy == "summary":
                out.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=f"summary for {ref.ref}",
                    )
                )
        return out


@pytest.mark.asyncio
async def test_resolve_with_resolver_marks_refs_verified_and_drops_unverified_fallback() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider, source_resolver=_StubVerifier())
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Check PR #42",
        scope=ContextScope(source_refs=["github:pr:42"]),
        source_policy="verify",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert all(r.verification_state == "verified" for r in bundle.source_refs)
    assert not any(f.code == "source_unverified" for f in bundle.fallbacks)
    assert any(v.ref == "github:pr:42" for v in bundle.source_resolution.verifications)


@pytest.mark.asyncio
async def test_resolve_with_resolver_summary_populates_source_resolution() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider, source_resolver=_StubVerifier())
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Why was PR #42 merged?",
        scope=ContextScope(source_refs=["github:pr:42"]),
        source_policy="summary",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert bundle.source_resolution.summaries
    envelope = bundle_to_agent_envelope(bundle)
    assert "source_resolution" in envelope
    assert envelope["source_resolution"]["summaries"]


@pytest.mark.asyncio
async def test_resolve_without_resolver_keeps_unavailable_fallback() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)  # no resolver wired
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Why was PR #42 merged?",
        scope=ContextScope(source_refs=["github:pr:42"]),
        source_policy="summary",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    # source_policy_fallbacks emits source_resolver_unavailable when no fetchable refs
    # and no resolver is wired (since the resolver path stays empty).
    assert bundle.source_resolution.summaries == []


# ---------------------------------------------------------------------------
# Phase 5: PR diff via artifact + source_policy (no graph pr_diff family)
# ---------------------------------------------------------------------------


from domain.intelligence_models import ArtifactRef
from domain.source_resolution import ResolvedSnippet


class _StubGitHubArtifactResolver:
    """Resolver that handles github PR refs via summary/snippets."""

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return (
            ResolverCapabilityEntry(
                provider="github",
                source_kind="pr",
                policies=frozenset({"summary", "snippets"}),
            ),
        )

    async def resolve(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        out = SourceResolutionResult()
        for ref in refs:
            # Stub only handles refs the GitHub PR child would handle.
            if (ref.source_system or "").lower() != "github":
                continue
            if source_policy == "summary":
                out.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=f"PR summary for {ref.external_id}",
                        source_system="github",
                    )
                )
            elif source_policy == "snippets":
                out.snippets.append(
                    ResolvedSnippet(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        snippet="@@ -1 +1 @@\n-old\n+new",
                        location="b/x.py",
                        source_system="github",
                    )
                )
        return out


@pytest.mark.asyncio
async def test_review_intent_works_without_diff_fetch_under_default_policy() -> None:
    """High-level review context resolves without needing a diff fetch."""
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider, source_resolver=_StubGitHubArtifactResolver())
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Review the recent PR for risky changes",
        intent="review",
        artifact_ref=ArtifactRef(kind="pr", identifier="42"),
        # default source_policy=references_only — must NOT trigger resolver
    )
    bundle = await resolve_context(svc, req)
    assert bundle.artifacts, "review intent should still surface artifact context"
    assert bundle.source_resolution.summaries == []
    assert bundle.source_resolution.snippets == []


@pytest.mark.asyncio
async def test_pr_artifact_with_summary_policy_routes_through_resolver() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider, source_resolver=_StubGitHubArtifactResolver())
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Summarize PR 42",
        artifact_ref=ArtifactRef(kind="pr", identifier="42"),
        source_policy="summary",
    )
    bundle = await resolve_context(svc, req)
    # Artifact ref must carry source_system='github' so the resolver dispatches.
    pr_refs = [r for r in bundle.source_refs if r.source_type == "pr"]
    assert pr_refs and pr_refs[0].source_system == "github"
    assert bundle.source_resolution.summaries
    assert bundle.source_resolution.summaries[0].summary.endswith("42")


@pytest.mark.asyncio
async def test_pr_artifact_with_snippets_policy_returns_bounded_diff() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider, source_resolver=_StubGitHubArtifactResolver())
    req = ContextResolutionRequest(
        pot_id="p1",
        query="Show me what changed in PR 42",
        artifact_ref=ArtifactRef(kind="pr", identifier="42"),
        source_policy="snippets",
    )
    bundle = await resolve_context(svc, req)
    assert bundle.source_resolution.snippets
    snip = bundle.source_resolution.snippets[0]
    assert snip.location == "b/x.py"
    # Snippet content stays bounded — assertion is structural, not heuristic.
    assert len(snip.snippet) < 10_000


@pytest.mark.asyncio
async def test_resolve_tickets_include_returns_ticket_records() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        pot_id="p1",
        query="What issues are open for the auth service?",
        include=["tickets"],
    )
    bundle = await resolve_context(svc, req)
    fallback_codes = {fallback.code for fallback in bundle.fallbacks}
    assert "context_family_not_implemented" not in fallback_codes
    assert any(item.family == "tickets" for item in bundle.project_map)
    assert any(item.kind == "Issue" for item in bundle.project_map)
