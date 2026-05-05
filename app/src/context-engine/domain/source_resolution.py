"""Source resolver contracts: request/response shapes used behind ``context_resolve``.

The :class:`SourceResolverPort` (in ``domain.ports.source_resolver``) is the
extension point for ``source_policy`` modes ``summary``, ``verify``, and
``snippets``. This module owns the normalized value types. Adapters produce
them; the resolution service merges them back into the agent envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


# Fallback codes returned by resolvers. These are a closed set so agents and
# UIs can branch on them reliably.
RESOLVER_UNAVAILABLE = "source_resolver_unavailable"
UNSUPPORTED_SOURCE_TYPE = "unsupported_source_type"
UNSUPPORTED_SOURCE_POLICY = "unsupported_source_policy"
PERMISSION_DENIED = "permission_denied"
STALE_TOKEN = "stale_token"
SOURCE_UNREACHABLE = "source_unreachable"
BUDGET_EXCEEDED = "budget_exceeded"
NO_SOURCE_REFERENCES = "no_source_references"
RESOLVER_ERROR = "source_resolver_error"

RESOLVER_FALLBACK_CODES = frozenset(
    {
        RESOLVER_UNAVAILABLE,
        UNSUPPORTED_SOURCE_TYPE,
        UNSUPPORTED_SOURCE_POLICY,
        PERMISSION_DENIED,
        STALE_TOKEN,
        SOURCE_UNREACHABLE,
        BUDGET_EXCEEDED,
        NO_SOURCE_REFERENCES,
        RESOLVER_ERROR,
    }
)


@dataclass(slots=True)
class ResolverBudget:
    """Per-request budget for source-backed reads.

    Resolvers must clamp themselves to these limits. ``max_refs`` caps how
    many refs are fetched, ``max_chars_per_item`` bounds any single
    summary/snippet, and ``max_total_chars`` is the global soft cap across
    all resolved items in one request.
    """

    max_refs: int = 6
    max_chars_per_item: int = 1200
    max_total_chars: int = 6000
    max_snippets_per_ref: int = 3
    timeout_ms: int = 4000


@dataclass(slots=True)
class ResolverAuthContext:
    """Caller auth/permissions threaded into the resolver.

    The host fills this from the request scope so resolvers can make
    permission-aware fetches (e.g. use the caller's GitHub token). The engine
    never inspects the values — it only forwards them.
    """

    user_id: str | None = None
    github_token: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedSummary:
    """Compact source-backed summary for one source ref."""

    ref: str
    source_type: str
    summary: str
    title: str | None = None
    fetched_at: str | None = None
    source_system: str | None = None
    retrieval_uri: str | None = None


@dataclass(slots=True)
class ResolvedSnippet:
    """Bounded snippet of source content for one source ref."""

    ref: str
    source_type: str
    snippet: str
    location: str | None = None
    fetched_at: str | None = None
    source_system: str | None = None


@dataclass(slots=True)
class ResolvedVerification:
    """Verification state for one source ref, checked against source of truth."""

    ref: str
    source_type: str
    verified: bool
    verification_state: str  # "verified" | "verification_failed" | "needs_verification"
    checked_at: str | None = None
    source_system: str | None = None
    reason: str | None = None  # e.g. "PR #42 is merged"; empty when verified=True


@dataclass(slots=True)
class ResolverFallback:
    """Structured fallback returned by a resolver when a ref is unresolvable."""

    code: str
    message: str
    ref: str | None = None
    source_type: str | None = None
    impact: str | None = None


@dataclass(slots=True)
class SourceResolutionResult:
    """Aggregate output from a resolver call across all requested refs."""

    summaries: list[ResolvedSummary] = field(default_factory=list)
    snippets: list[ResolvedSnippet] = field(default_factory=list)
    verifications: list[ResolvedVerification] = field(default_factory=list)
    fallbacks: list[ResolverFallback] = field(default_factory=list)

    def extend(self, other: "SourceResolutionResult") -> None:
        self.summaries.extend(other.summaries)
        self.snippets.extend(other.snippets)
        self.verifications.extend(other.verifications)
        self.fallbacks.extend(other.fallbacks)

    def total_chars(self) -> int:
        n = 0
        for s in self.summaries:
            n += len(s.summary or "")
        for sn in self.snippets:
            n += len(sn.snippet or "")
        return n


@dataclass(slots=True)
class ResolverCapabilityEntry:
    """What one resolver can do for a ``(provider, source_kind)`` pairing.

    ``provider`` matches the pot source provider (e.g. ``github``,
    ``notion``). ``source_kind`` matches the pot source kind (e.g.
    ``repository``, ``docs_space``). ``policies`` lists the ``source_policy``
    values the resolver can fulfill for that pairing.
    """

    provider: str
    source_kind: str
    policies: frozenset[str] = field(default_factory=frozenset)
    reason: str | None = None  # Populated only when no policies are supported.


def clamp_text(value: str | None, max_chars: int) -> str:
    if not value:
        return ""
    if max_chars <= 0:
        return ""
    value = value.strip()
    if len(value) <= max_chars:
        return value
    # Preserve a trailing ellipsis so downstream renderers know the text was cut.
    return value[: max(1, max_chars - 1)].rstrip() + "…"


def summaries_to_payload(items: Iterable[ResolvedSummary]) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]


def snippets_to_payload(items: Iterable[ResolvedSnippet]) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]


def verifications_to_payload(
    items: Iterable[ResolvedVerification],
) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]


def resolver_fallbacks_to_payload(
    items: Iterable[ResolverFallback],
) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]


def source_resolution_to_payload(result: SourceResolutionResult) -> dict[str, Any]:
    return {
        "summaries": summaries_to_payload(result.summaries),
        "snippets": snippets_to_payload(result.snippets),
        "verifications": verifications_to_payload(result.verifications),
        "fallbacks": resolver_fallbacks_to_payload(result.fallbacks),
    }
