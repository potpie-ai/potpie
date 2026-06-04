"""Unit tests for the source resolver port types and helpers."""

from __future__ import annotations

import pytest

from domain.source_resolution import (
    NO_SOURCE_REFERENCES,
    RESOLVER_FALLBACK_CODES,
    RESOLVER_UNAVAILABLE,
    ResolvedSnippet,
    ResolvedSummary,
    ResolverBudget,
    ResolverCapabilityEntry,
    ResolverFallback,
    SourceResolutionResult,
    clamp_text,
    source_resolution_to_payload,
)

pytestmark = pytest.mark.unit


def test_resolver_fallback_codes_are_closed_set() -> None:
    assert RESOLVER_UNAVAILABLE in RESOLVER_FALLBACK_CODES
    assert NO_SOURCE_REFERENCES in RESOLVER_FALLBACK_CODES


def test_clamp_text_preserves_short_input() -> None:
    assert clamp_text("hello", 50) == "hello"


def test_clamp_text_trims_with_ellipsis() -> None:
    out = clamp_text("abcdefghij", 5)
    assert out.endswith("…")
    assert len(out) <= 5


def test_clamp_text_handles_none_and_zero() -> None:
    assert clamp_text(None, 100) == ""
    assert clamp_text("abc", 0) == ""


def test_source_resolution_result_extend_and_total_chars() -> None:
    a = SourceResolutionResult(
        summaries=[
            ResolvedSummary(ref="r1", source_type="pr", summary="hi there")
        ]
    )
    b = SourceResolutionResult(
        snippets=[ResolvedSnippet(ref="r2", source_type="doc", snippet="body")]
    )
    a.extend(b)
    assert len(a.summaries) == 1
    assert len(a.snippets) == 1
    assert a.total_chars() == len("hi there") + len("body")


def test_source_resolution_payload_shape() -> None:
    result = SourceResolutionResult(
        summaries=[ResolvedSummary(ref="r1", source_type="pr", summary="ok")],
        fallbacks=[ResolverFallback(code=RESOLVER_UNAVAILABLE, message="no resolver")],
    )
    payload = source_resolution_to_payload(result)
    assert payload["summaries"][0]["summary"] == "ok"
    assert payload["fallbacks"][0]["code"] == RESOLVER_UNAVAILABLE
    assert payload["snippets"] == []
    assert payload["verifications"] == []


def test_resolver_capability_entry_defaults_to_empty_policies() -> None:
    entry = ResolverCapabilityEntry(provider="x", source_kind="y")
    assert entry.policies == frozenset()


def test_resolver_budget_sane_defaults() -> None:
    b = ResolverBudget()
    assert b.max_refs > 0
    assert b.max_chars_per_item > 0
    assert b.max_total_chars >= b.max_chars_per_item
