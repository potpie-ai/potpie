"""Source resolver value types and payload helpers."""

from __future__ import annotations

import pytest

from domain.source_resolution import (
    BUDGET_EXCEEDED,
    NO_SOURCE_REFERENCES,
    PERMISSION_DENIED,
    RESOLVER_ERROR,
    RESOLVER_FALLBACK_CODES,
    RESOLVER_UNAVAILABLE,
    SOURCE_UNREACHABLE,
    STALE_TOKEN,
    UNSUPPORTED_SOURCE_POLICY,
    UNSUPPORTED_SOURCE_TYPE,
    ResolvedSnippet,
    ResolvedSummary,
    ResolvedVerification,
    ResolverAuthContext,
    ResolverBudget,
    ResolverCapabilityEntry,
    ResolverFallback,
    SourceResolutionResult,
    clamp_text,
    resolver_fallbacks_to_payload,
    snippets_to_payload,
    source_resolution_to_payload,
    summaries_to_payload,
    verifications_to_payload,
)

pytestmark = pytest.mark.unit


# --- clamp_text ------------------------------------------------------------


class TestClampText:
    def test_none_returns_empty(self) -> None:
        assert clamp_text(None, 100) == ""

    def test_empty_returns_empty(self) -> None:
        assert clamp_text("", 100) == ""

    def test_zero_max_returns_empty(self) -> None:
        assert clamp_text("hello", 0) == ""

    def test_negative_max_returns_empty(self) -> None:
        assert clamp_text("hello", -3) == ""

    def test_short_text_passes_through_stripped(self) -> None:
        assert clamp_text("  hello  ", 100) == "hello"

    def test_truncates_with_ellipsis(self) -> None:
        out = clamp_text("a" * 100, 10)
        assert out.endswith("…")
        assert len(out) == 10

    def test_max_one_truncates_to_just_ellipsis(self) -> None:
        # max_chars=1 → max(1, 0) = 1 char of text + "…" = 2 chars total.
        # Actually: value[:max(1, 0)] = value[:1] = first char, then rstrip + "…".
        out = clamp_text("hello", 1)
        assert out.endswith("…")


# --- Fallback constants ----------------------------------------------------


class TestFallbackCodes:
    def test_constants_in_set(self) -> None:
        for code in (
            RESOLVER_UNAVAILABLE,
            UNSUPPORTED_SOURCE_TYPE,
            UNSUPPORTED_SOURCE_POLICY,
            PERMISSION_DENIED,
            STALE_TOKEN,
            SOURCE_UNREACHABLE,
            BUDGET_EXCEEDED,
            NO_SOURCE_REFERENCES,
            RESOLVER_ERROR,
        ):
            assert code in RESOLVER_FALLBACK_CODES

    def test_codes_are_strings(self) -> None:
        for code in RESOLVER_FALLBACK_CODES:
            assert isinstance(code, str)
            assert code  # non-empty


# --- Budget / auth dataclass defaults --------------------------------------


class TestBudgetAndAuthDefaults:
    def test_resolver_budget_defaults(self) -> None:
        b = ResolverBudget()
        assert b.max_refs == 6
        assert b.max_chars_per_item == 1200
        assert b.max_total_chars == 6000
        assert b.max_snippets_per_ref == 3
        assert b.timeout_ms == 4000

    def test_resolver_auth_context_defaults(self) -> None:
        a = ResolverAuthContext()
        assert a.user_id is None
        assert a.github_token is None
        assert a.extra == {}

    def test_resolver_capability_entry_defaults(self) -> None:
        e = ResolverCapabilityEntry(provider="github", source_kind="repo")
        assert e.policies == frozenset()
        assert e.reason is None


# --- SourceResolutionResult ------------------------------------------------


class TestSourceResolutionResult:
    def test_extend_concatenates_each_list(self) -> None:
        a = SourceResolutionResult(
            summaries=[ResolvedSummary(ref="a", source_type="pr", summary="A")],
            snippets=[ResolvedSnippet(ref="a", source_type="pr", snippet="snip")],
            verifications=[ResolvedVerification(
                ref="a", source_type="pr", verified=True, verification_state="verified"
            )],
            fallbacks=[ResolverFallback(code=PERMISSION_DENIED, message="m")],
        )
        b = SourceResolutionResult(
            summaries=[ResolvedSummary(ref="b", source_type="pr", summary="B")],
        )
        a.extend(b)
        assert [s.ref for s in a.summaries] == ["a", "b"]
        assert len(a.snippets) == 1  # b had no snippets
        assert len(a.fallbacks) == 1

    def test_total_chars_counts_summaries_and_snippets(self) -> None:
        result = SourceResolutionResult(
            summaries=[
                ResolvedSummary(ref="a", source_type="pr", summary="hello"),
                ResolvedSummary(ref="b", source_type="pr", summary=""),
            ],
            snippets=[
                ResolvedSnippet(ref="a", source_type="pr", snippet="snip!"),
            ],
        )
        assert result.total_chars() == len("hello") + len("snip!")

    def test_total_chars_handles_empty(self) -> None:
        assert SourceResolutionResult().total_chars() == 0


# --- Payload conversion ----------------------------------------------------


class TestPayloadHelpers:
    def test_summaries_to_payload(self) -> None:
        out = summaries_to_payload(
            [ResolvedSummary(ref="a", source_type="pr", summary="x")]
        )
        assert out == [
            {
                "ref": "a",
                "source_type": "pr",
                "summary": "x",
                "title": None,
                "fetched_at": None,
                "source_system": None,
                "retrieval_uri": None,
            }
        ]

    def test_snippets_to_payload(self) -> None:
        out = snippets_to_payload(
            [ResolvedSnippet(ref="a", source_type="pr", snippet="x")]
        )
        assert out[0]["snippet"] == "x"

    def test_verifications_to_payload(self) -> None:
        out = verifications_to_payload(
            [
                ResolvedVerification(
                    ref="a",
                    source_type="pr",
                    verified=True,
                    verification_state="verified",
                )
            ]
        )
        assert out[0]["verified"] is True

    def test_resolver_fallbacks_to_payload(self) -> None:
        out = resolver_fallbacks_to_payload(
            [ResolverFallback(code=PERMISSION_DENIED, message="m")]
        )
        assert out[0]["code"] == PERMISSION_DENIED

    def test_source_resolution_to_payload_aggregates(self) -> None:
        result = SourceResolutionResult(
            summaries=[ResolvedSummary(ref="a", source_type="pr", summary="x")],
            snippets=[ResolvedSnippet(ref="a", source_type="pr", snippet="y")],
            verifications=[
                ResolvedVerification(
                    ref="a",
                    source_type="pr",
                    verified=True,
                    verification_state="verified",
                )
            ],
            fallbacks=[ResolverFallback(code=PERMISSION_DENIED, message="m")],
        )
        payload = source_resolution_to_payload(result)
        assert set(payload.keys()) == {"summaries", "snippets", "verifications", "fallbacks"}
        assert len(payload["summaries"]) == 1
        assert len(payload["fallbacks"]) == 1
