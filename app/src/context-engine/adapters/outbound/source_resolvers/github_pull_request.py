"""GitHub pull-request source resolver.

Handles ``source_policy`` modes ``summary``, ``verify``, and ``snippets`` for
refs that identify a GitHub pull request. The host injects a factory that
returns a ``SourceControlPort`` for a given repo name; the resolver calls
``get_pull_request``/``get_pull_request_commits`` and clamps output to the
caller's :class:`ResolverBudget`.

Refs are matched by ``source_system == "github"`` or ``source_type == "pr"``
and parsed from common shapes (``github:pr:42``, ``PR #42``, or an
``external_id`` that parses as an integer). A ``repo_resolver`` callable
maps ``pot_id`` → ``repo_name``; hosts that attach a single GitHub repo to
each pot can return that directly, multi-repo hosts should disambiguate via
the ref's ``resolver_hint``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

from domain.ports.source_control import SourceControlPort
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

# RepoResolver can be sync or async: ``(pot_id, ref) -> repo_name | None``.
# Returning ``None`` emits an UNSUPPORTED_SOURCE_TYPE fallback for that ref.
RepoResolver = Callable[[str, SourceReferenceRecord], "str | None | Awaitable[str | None]"]
# SourceControlFactory: ``(repo_name) -> SourceControlPort``. Typically the
# same callable as ``ContextEngineContainer.source_for_repo``.
SourceControlFactory = Callable[[str], SourceControlPort]


class GitHubPullRequestResolver:
    """Resolver for GitHub pull-request refs."""

    _PR_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"^github:pr:(\d+)$", re.IGNORECASE),
        re.compile(r"^pr\s*#?(\d+)$", re.IGNORECASE),
        re.compile(r"pull/(\d+)"),
        re.compile(r"^(\d+)$"),
    )

    def __init__(
        self,
        *,
        source_for_repo: SourceControlFactory,
        repo_resolver: RepoResolver,
    ) -> None:
        self._source_for_repo = source_for_repo
        self._repo_resolver = repo_resolver

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return (
            ResolverCapabilityEntry(
                provider="github",
                source_kind="repository",
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
        policy = normalize_source_policy(source_policy)
        out = SourceResolutionResult()
        if policy not in {"summary", "verify", "snippets"}:
            out.fallbacks.append(
                ResolverFallback(
                    code=UNSUPPORTED_SOURCE_POLICY,
                    message=f"GitHubPullRequestResolver does not handle policy={policy!r}.",
                )
            )
            return out

        remaining_chars = budget.max_total_chars
        for ref in refs:
            pr_number = self._parse_pr_number(ref)
            if pr_number is None:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="Could not parse a PR number from ref.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            repo_name = await self._resolve_repo(pot_id, ref)
            if not repo_name:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="No GitHub repo could be resolved for this pot/ref.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            try:
                client = self._source_for_repo(repo_name)
                want_diff = policy == "snippets"
                pr_data = client.get_pull_request(
                    repo_name, pr_number, include_diff=want_diff
                )
            except PermissionError as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=PERMISSION_DENIED,
                        message=str(exc) or "GitHub rejected the request.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except (ConnectionError, TimeoutError) as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=SOURCE_UNREACHABLE,
                        message=f"GitHub unreachable: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except Exception as exc:
                logger.exception("github PR fetch failed: %s", exc)
                out.fallbacks.append(
                    ResolverFallback(
                        code=RESOLVER_ERROR,
                        message=f"GitHub fetch raised: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            now_iso = datetime.now(timezone.utc).isoformat()

            if policy == "summary":
                text = _compose_pr_summary(pr_data, budget.max_chars_per_item)
                if len(text) > remaining_chars:
                    text = clamp_text(text, remaining_chars)
                if not text:
                    continue
                out.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=text,
                        title=_str_field(pr_data, "title"),
                        fetched_at=now_iso,
                        source_system="github",
                        retrieval_uri=_str_field(pr_data, "html_url") or ref.retrieval_uri,
                    )
                )
                remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break

            elif policy == "verify":
                merged = bool(pr_data.get("merged"))
                state = str(pr_data.get("state") or "").lower()
                verification_state = (
                    "verified" if (merged or state in {"open", "closed"}) else "verification_failed"
                )
                out.verifications.append(
                    ResolvedVerification(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        verified=verification_state == "verified",
                        verification_state=verification_state,
                        checked_at=now_iso,
                        source_system="github",
                        reason=(
                            f"PR state={state!r}, merged={merged}"
                            if verification_state != "verified"
                            else None
                        ),
                    )
                )

            elif policy == "snippets":
                diff_text = _str_field(pr_data, "diff") or ""
                body_text = _str_field(pr_data, "body") or ""
                chunks = _split_diff_chunks(
                    diff_text or body_text,
                    per_item=budget.max_chars_per_item,
                    max_chunks=budget.max_snippets_per_ref,
                )
                if not chunks and body_text:
                    chunks = [clamp_text(body_text, budget.max_chars_per_item)]
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
                            source_system="github",
                        )
                    )
                    remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break

        return out

    async def _resolve_repo(
        self,
        pot_id: str,
        ref: SourceReferenceRecord,
    ) -> str | None:
        result = self._repo_resolver(pot_id, ref)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        return result if isinstance(result, str) and result else None

    def _parse_pr_number(self, ref: SourceReferenceRecord) -> int | None:
        for candidate in (ref.external_id, ref.ref, ref.uri, ref.retrieval_uri):
            n = self._parse_int(candidate)
            if n is not None:
                return n
        return None

    @classmethod
    def _parse_int(cls, value: str | None) -> int | None:
        if not value:
            return None
        value = value.strip()
        for pattern in cls._PR_PATTERNS:
            m = pattern.search(value)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None
        return None


def _compose_pr_summary(pr: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    title = _str_field(pr, "title")
    state = _str_field(pr, "state")
    merged = pr.get("merged")
    if title:
        parts.append(title)
    meta: list[str] = []
    if state:
        meta.append(f"state={state}")
    if merged is not None:
        meta.append(f"merged={'yes' if merged else 'no'}")
    author = _str_field(pr, "user_login") or _str_field(pr, "author")
    if author:
        meta.append(f"author={author}")
    if meta:
        parts.append("(" + ", ".join(meta) + ")")
    body = _str_field(pr, "body")
    if body:
        parts.append(body)
    return clamp_text(" ".join(parts), max_chars)


def _split_diff_chunks(
    text: str,
    *,
    per_item: int,
    max_chunks: int,
) -> list[tuple[str, str | None]]:
    """Split a unified diff into at most ``max_chunks`` per-file hunks.

    Falls back to a single clamped chunk when the input is not a recognizable
    diff. Each chunk is labelled by its ``+++`` file path when available so
    agents can cite the location.
    """
    if not text or max_chunks <= 0:
        return []
    if "diff --git" not in text and "+++ " not in text:
        return [(clamp_text(text, per_item), None)]
    chunks: list[tuple[str, str | None]] = []
    current_lines: list[str] = []
    current_path: str | None = None
    for line in text.splitlines():
        if line.startswith("diff --git"):
            if current_lines:
                chunks.append(
                    (clamp_text("\n".join(current_lines), per_item), current_path)
                )
                if len(chunks) >= max_chunks:
                    return chunks
            current_lines = [line]
            current_path = None
            continue
        if line.startswith("+++ "):
            current_path = line[4:].strip()
        current_lines.append(line)
    if current_lines:
        chunks.append((clamp_text("\n".join(current_lines), per_item), current_path))
    return chunks[:max_chunks]


def _str_field(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
