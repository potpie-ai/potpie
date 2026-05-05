"""Documentation URI source resolver.

Lightweight resolver for ``documentation`` / ``docs`` / ``uri`` refs that
carry an ``http(s)`` ``retrieval_uri``. It does not fetch remote content by
default — that requires host-side auth and network policy. It:

* For ``summary``: re-emits the graph-stored ``title`` + ``summary`` under
  a fresh timestamp so agents see source-grouped summaries. If a host wants
  live fetches it can plug in a ``content_fetcher`` callable.
* For ``verify``: marks refs with a reachable-looking ``http(s)``
  ``retrieval_uri`` as ``verified`` (link integrity, not content diff).
* For ``snippets``: returns nothing and emits an
  ``unsupported_source_policy`` fallback — docs snippets need live fetches,
  which is opt-in.

When a ``content_fetcher`` is provided, ``summary`` and ``snippets`` call it
and clamp to the request budget. The fetcher may raise ``PermissionError``,
``ConnectionError``, or ``TimeoutError`` and the resolver converts them to
structured fallbacks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable, Sequence
from urllib.parse import urlparse

from domain.source_references import SourceReferenceRecord, normalize_source_policy
from domain.source_resolution import (
    PERMISSION_DENIED,
    RESOLVER_ERROR,
    SOURCE_UNREACHABLE,
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
)

logger = logging.getLogger(__name__)

DOC_SOURCE_TYPES = frozenset(
    {
        "documentation",
        "docs",
        "doc",
        "uri",
        "url",
        "http",
        "https",
        "notion",
        "confluence",
    }
)

# Async fetcher: ``(uri, auth) -> fetched text | None``.
ContentFetcher = Callable[[str, ResolverAuthContext], Awaitable[str | None]]


class DocumentationUriResolver:
    """Resolver for documentation URI refs."""

    def __init__(
        self,
        *,
        content_fetcher: ContentFetcher | None = None,
        supported_source_kinds: Sequence[str] = ("docs_space",),
        providers: Sequence[str] = ("documentation", "docs", "notion", "confluence"),
    ) -> None:
        self._content_fetcher = content_fetcher
        self._source_kinds = tuple(supported_source_kinds)
        self._providers = tuple(providers)

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        entries: list[ResolverCapabilityEntry] = []
        policies = {"summary", "verify"}
        if self._content_fetcher is not None:
            policies.add("snippets")
        for provider in self._providers:
            for kind in self._source_kinds:
                entries.append(
                    ResolverCapabilityEntry(
                        provider=provider,
                        source_kind=kind,
                        policies=frozenset(policies),
                    )
                )
        return entries

    async def resolve(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        policy = normalize_source_policy(source_policy)
        out = SourceResolutionResult()
        remaining_chars = budget.max_total_chars

        for ref in refs:
            if not _is_doc_ref(ref):
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="DocumentationUriResolver only handles documentation refs.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            uri = ref.retrieval_uri or ref.uri
            if policy == "verify":
                if _looks_fetchable(uri):
                    out.verifications.append(
                        ResolvedVerification(
                            ref=ref.ref,
                            source_type=ref.source_type,
                            verified=True,
                            verification_state="verified",
                            checked_at=datetime.now(timezone.utc).isoformat(),
                            source_system=ref.source_system or "documentation",
                            reason=None,
                        )
                    )
                else:
                    out.verifications.append(
                        ResolvedVerification(
                            ref=ref.ref,
                            source_type=ref.source_type,
                            verified=False,
                            verification_state="verification_failed",
                            checked_at=datetime.now(timezone.utc).isoformat(),
                            source_system=ref.source_system or "documentation",
                            reason="No reachable http(s) retrieval_uri for this ref.",
                        )
                    )
                continue

            if policy == "summary":
                text = await self._fetch_or_fallback(
                    ref=ref,
                    uri=uri,
                    auth=auth,
                    budget=budget,
                    remaining_chars=remaining_chars,
                    out=out,
                )
                if text is None:
                    continue
                out.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=text,
                        title=ref.title,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        source_system=ref.source_system or "documentation",
                        retrieval_uri=uri,
                    )
                )
                remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break
                continue

            if policy == "snippets":
                if self._content_fetcher is None:
                    out.fallbacks.append(
                        ResolverFallback(
                            code=UNSUPPORTED_SOURCE_POLICY,
                            message=(
                                "DocumentationUriResolver has no content fetcher; "
                                "snippets require live fetches."
                            ),
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                    continue
                if not uri:
                    out.fallbacks.append(
                        ResolverFallback(
                            code=UNSUPPORTED_SOURCE_TYPE,
                            message="Ref has no retrieval_uri for snippets.",
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                    continue
                try:
                    fetched = await self._content_fetcher(uri, auth)
                except PermissionError as exc:
                    out.fallbacks.append(
                        ResolverFallback(
                            code=PERMISSION_DENIED,
                            message=str(exc) or "Access denied.",
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                    continue
                except (ConnectionError, TimeoutError) as exc:
                    out.fallbacks.append(
                        ResolverFallback(
                            code=SOURCE_UNREACHABLE,
                            message=f"Doc source unreachable: {exc}",
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                    continue
                except Exception as exc:
                    logger.exception("doc fetcher failed: %s", exc)
                    out.fallbacks.append(
                        ResolverFallback(
                            code=RESOLVER_ERROR,
                            message=f"Doc fetcher raised: {exc}",
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                    continue
                clamped = clamp_text(
                    fetched,
                    min(budget.max_chars_per_item, remaining_chars),
                )
                if not clamped:
                    continue
                out.snippets.append(
                    ResolvedSnippet(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        snippet=clamped,
                        location=uri,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        source_system=ref.source_system or "documentation",
                    )
                )
                remaining_chars = max(0, remaining_chars - len(clamped))
                if remaining_chars <= 0:
                    break
                continue

            out.fallbacks.append(
                ResolverFallback(
                    code=UNSUPPORTED_SOURCE_POLICY,
                    message=f"DocumentationUriResolver does not handle policy={policy!r}.",
                    ref=ref.ref,
                    source_type=ref.source_type,
                )
            )

        return out

    async def _fetch_or_fallback(
        self,
        *,
        ref: SourceReferenceRecord,
        uri: str | None,
        auth: ResolverAuthContext,
        budget: ResolverBudget,
        remaining_chars: int,
        out: SourceResolutionResult,
    ) -> str | None:
        """Fetch live when a fetcher is wired; otherwise reuse stored summary."""
        if self._content_fetcher is not None and uri:
            try:
                fetched = await self._content_fetcher(uri, auth)
            except PermissionError as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=PERMISSION_DENIED,
                        message=str(exc) or "Access denied.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                return None
            except (ConnectionError, TimeoutError) as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=SOURCE_UNREACHABLE,
                        message=f"Doc source unreachable: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                return None
            except Exception as exc:
                logger.exception("doc fetcher failed: %s", exc)
                out.fallbacks.append(
                    ResolverFallback(
                        code=RESOLVER_ERROR,
                        message=f"Doc fetcher raised: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                return None
            clamped = clamp_text(
                fetched,
                min(budget.max_chars_per_item, remaining_chars),
            )
            return clamped or None

        # Fallback: reuse graph-stored title/summary so agents still get a
        # source-grouped summary block.
        stored = " ".join(p for p in (ref.title, ref.summary) if p)
        if not stored:
            return None
        return clamp_text(stored, min(budget.max_chars_per_item, remaining_chars)) or None


def _is_doc_ref(ref: SourceReferenceRecord) -> bool:
    src_type = (ref.source_type or "").lower()
    src_system = (ref.source_system or "").lower()
    return src_type in DOC_SOURCE_TYPES or src_system in DOC_SOURCE_TYPES


def _looks_fetchable(uri: str | None) -> bool:
    if not uri:
        return False
    try:
        parsed = urlparse(uri)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
