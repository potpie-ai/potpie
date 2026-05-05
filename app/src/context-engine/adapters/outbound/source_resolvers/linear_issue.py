"""Linear issue source resolver.

Handles ``source_policy`` modes ``summary``, ``verify``, and ``snippets`` for
refs that identify a Linear issue. Refs are matched by
``source_system == "linear"`` or ``source_type in {"linear_issue", "issue"}``
and parsed from common shapes (``linear:issue:ENG-123``, ``linear:ENG-123``,
``ENG-123``, the raw Linear UUID, or an ``external_id`` carrying the
identifier).

The resolver depends on :class:`LinearIssueFetcher` — hosts inject an
implementation that calls the Linear GraphQL API. Tests can pass a simple
dict-backed fake.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Sequence

from domain.linear_events import LinearIssue, linear_issue_from_payload
from domain.ports.linear_source import LinearIssueFetcher
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


_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^linear:issue:([A-Z][A-Z0-9]*-\d+)$", re.IGNORECASE),
    re.compile(r"^linear:([A-Z][A-Z0-9]*-\d+)$", re.IGNORECASE),
    re.compile(r"^issue:([A-Z][A-Z0-9]*-\d+)$", re.IGNORECASE),
    re.compile(r"^([A-Z][A-Z0-9]*-\d+)$", re.IGNORECASE),
    # Linear URL: https://linear.app/<org>/issue/ENG-123/...
    re.compile(r"linear\.app/[^/]+/issue/([A-Z][A-Z0-9]*-\d+)", re.IGNORECASE),
)

_COMPLETED_STATE_TYPES = frozenset({"completed", "canceled"})


class LinearIssueResolver:
    """Resolver for Linear issue refs."""

    def __init__(self, *, fetcher: LinearIssueFetcher) -> None:
        self._fetcher = fetcher

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return (
            ResolverCapabilityEntry(
                provider="linear",
                source_kind="linear_issue",
                policies=frozenset({"summary", "verify", "snippets"}),
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
        _ = pot_id, auth  # Linear auth is attached to the injected fetcher.
        policy = normalize_source_policy(source_policy)
        out = SourceResolutionResult()
        if policy not in {"summary", "verify", "snippets"}:
            out.fallbacks.append(
                ResolverFallback(
                    code=UNSUPPORTED_SOURCE_POLICY,
                    message=f"LinearIssueResolver does not handle policy={policy!r}.",
                )
            )
            return out

        remaining_chars = budget.max_total_chars
        for ref in refs:
            if not _is_linear_ref(ref):
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="Not a Linear issue ref.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            identifier = _parse_identifier(ref)
            if not identifier:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="Could not parse a Linear identifier (e.g. ENG-123) from ref.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            try:
                payload = self._fetcher.get_issue(identifier)
            except PermissionError as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=PERMISSION_DENIED,
                        message=str(exc) or "Linear rejected the request.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except (ConnectionError, TimeoutError) as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=SOURCE_UNREACHABLE,
                        message=f"Linear unreachable: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except Exception as exc:
                logger.exception("linear issue fetch failed: %s", exc)
                out.fallbacks.append(
                    ResolverFallback(
                        code=RESOLVER_ERROR,
                        message=f"Linear fetch raised: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            if payload is None:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message=f"Linear issue {identifier} not found.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            issue = linear_issue_from_payload(payload)
            now_iso = datetime.now(timezone.utc).isoformat()

            if policy == "summary":
                text = _compose_issue_summary(issue, budget.max_chars_per_item)
                if len(text) > remaining_chars:
                    text = clamp_text(text, remaining_chars)
                if not text:
                    continue
                out.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=text,
                        title=issue.title or None,
                        fetched_at=now_iso,
                        source_system="linear",
                        retrieval_uri=issue.url or ref.retrieval_uri,
                    )
                )
                remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break

            elif policy == "verify":
                state_name = issue.state.name if issue.state else None
                state_type = (issue.state.type or "").lower() if issue.state else ""
                verified = bool(state_name)
                out.verifications.append(
                    ResolvedVerification(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        verified=verified,
                        verification_state="verified" if verified else "verification_failed",
                        checked_at=now_iso,
                        source_system="linear",
                        reason=(
                            f"Linear issue {issue.identifier} is in state "
                            f"{state_name!r}"
                            + (" (terminal)" if state_type in _COMPLETED_STATE_TYPES else "")
                        )
                        if verified
                        else "Linear returned no state for this issue.",
                    )
                )

            elif policy == "snippets":
                chunks = _issue_snippet_chunks(
                    issue,
                    per_item=budget.max_chars_per_item,
                    max_chunks=budget.max_snippets_per_ref,
                )
                for chunk, location in chunks:
                    if remaining_chars <= 0:
                        break
                    text = clamp_text(chunk, min(budget.max_chars_per_item, remaining_chars))
                    if not text:
                        continue
                    out.snippets.append(
                        ResolvedSnippet(
                            ref=ref.ref,
                            source_type=ref.source_type,
                            snippet=text,
                            location=location,
                            fetched_at=now_iso,
                            source_system="linear",
                        )
                    )
                    remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break

        return out


def _is_linear_ref(ref: SourceReferenceRecord) -> bool:
    if (ref.source_system or "").lower() == "linear":
        return True
    if (ref.source_type or "").lower() in {"linear_issue", "issue"}:
        # ``issue`` is generic; only treat it as Linear when the ref string hints so.
        return bool(_parse_identifier(ref))
    if ref.ref and ref.ref.lower().startswith(("linear:", "linear/")):
        return True
    return False


def _parse_identifier(ref: SourceReferenceRecord) -> str | None:
    for candidate in (
        ref.external_id,
        ref.ref,
        ref.uri,
        ref.retrieval_uri,
        (ref.resolver_hint or {}).get("linear_identifier")
        if isinstance(ref.resolver_hint, dict)
        else None,
    ):
        if not candidate:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        for pattern in _IDENTIFIER_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(1).upper()
    return None


def _compose_issue_summary(issue: LinearIssue, max_chars: int) -> str:
    parts: list[str] = []
    header = issue.title or issue.identifier or "Linear issue"
    parts.append(f"{issue.identifier}: {header}" if issue.identifier and issue.title else header)
    meta: list[str] = []
    if issue.state and issue.state.name:
        meta.append(f"state={issue.state.name}")
    if issue.assignee:
        meta.append(f"assignee={issue.assignee.name}")
    if issue.priority is not None:
        meta.append(f"priority={issue.priority}")
    if issue.labels:
        meta.append("labels=" + ",".join(lbl.name for lbl in issue.labels[:6]))
    if meta:
        parts.append("(" + ", ".join(meta) + ")")
    if issue.description:
        parts.append(issue.description)
    return clamp_text(" ".join(parts), max_chars)


def _issue_snippet_chunks(
    issue: LinearIssue,
    *,
    per_item: int,
    max_chunks: int,
) -> list[tuple[str, str | None]]:
    """Return up to ``max_chunks`` snippets: description first, then comments.

    Each snippet is labelled with a stable location (``description``,
    ``comment:<id>``) so agents can cite where text came from.
    """
    if max_chunks <= 0:
        return []
    chunks: list[tuple[str, str | None]] = []
    desc = (issue.description or "").strip()
    if desc:
        chunks.append((clamp_text(desc, per_item), "description"))
    for comment in issue.comments:
        if len(chunks) >= max_chunks:
            break
        body = (comment.body or "").strip()
        if not body:
            continue
        author = comment.author.name if comment.author else "unknown"
        chunks.append(
            (clamp_text(f"{author}: {body}", per_item), f"comment:{comment.id}")
        )
    return chunks[:max_chunks]
