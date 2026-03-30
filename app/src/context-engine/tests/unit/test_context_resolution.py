"""Unit tests for context intelligence resolution (mock provider)."""

from __future__ import annotations

import pytest

from adapters.outbound.intelligence.mock import MockIntelligenceProvider
from application.services.context_resolution import ContextResolutionService
from application.use_cases.resolve_context import resolve_context
from domain.intelligence_models import ContextResolutionRequest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_resolve_returns_bundle_with_semantic_hits() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        project_id="p1",
        query="Why was this changed?",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert bundle.request.project_id == "p1"
    assert len(bundle.semantic_hits) >= 1
    assert bundle.meta.schema_version == "1"
    assert bundle.meta.provider == "MockIntelligenceProvider"


@pytest.mark.asyncio
async def test_resolve_pr_query_includes_discussions() -> None:
    provider = MockIntelligenceProvider()
    svc = ContextResolutionService(provider)
    req = ContextResolutionRequest(
        project_id="p1",
        query="What happened in PR #42?",
        timeout_ms=4000,
    )
    bundle = await resolve_context(svc, req)
    assert len(bundle.discussions) >= 1
    assert "PR #42" in (bundle.discussions[0].source_ref or "")


def test_extract_signals_pr_number() -> None:
    from domain.intelligence_signals import extract_signals

    s = extract_signals("Tell me about PR #99")
    assert s.mentioned_pr == 99
