"""GitLab merge-request source resolver.

Handles ``source_policy`` modes ``summary``, ``verify``, and ``snippets``
for refs that identify a GitLab merge request — the counterpart of
``connectors/github/resolver.py``. The host injects a factory that
returns a ``GitLabReadPort`` for a given project path; the resolver calls
``get_merge_request`` and clamps output to the caller's
:class:`ResolverBudget`.

Refs are matched by ``source_system == "gitlab"`` or a
``merge_request``/``mr`` source type and parsed from the common shapes
(``gitlab:mr:group/project:42``, ``!42``, a ``/-/merge_requests/42``
URL, or an ``external_id`` that parses as an integer). A
``project_resolver`` callable maps ``pot_id`` → project path; hosts that
attach a single GitLab project to each pot can return that directly,
multi-project hosts should disambiguate via the ref's ``resolver_hint``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

from potpie_context_engine.adapters.outbound.connectors._diff_chunks import (
    split_diff_chunks,
)
from potpie_context_engine.adapters.outbound.connectors.gitlab.api_client import (
    GitLabReadPort,
)
from potpie_context_core.source_references import (
    SourceReferenceRecord,
    normalize_source_policy,
)
from potpie_context_engine.domain.source_resolution import (
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

# ProjectResolver can be sync or async: ``(pot_id, ref) -> project | None``.
# Returning ``None`` emits an UNSUPPORTED_SOURCE_TYPE fallback for that ref.
ProjectResolver = Callable[
    [str, SourceReferenceRecord], "str | None | Awaitable[str | None]"
]
# SourceControlFactory: ``(project) -> GitLabReadPort``.
SourceControlFactory = Callable[[str], GitLabReadPort]


class GitLabMergeRequestResolver:
    """Resolver for GitLab merge-request refs."""

    _MR_PATTERNS: tuple[re.Pattern[str], ...] = (
        # gitlab:mr:group/sub/project:42 — the iid is the last segment.
        re.compile(r"^gitlab:mr:.+:(\d+)$", re.IGNORECASE),
        re.compile(r"^gitlab:mr:(\d+)$", re.IGNORECASE),
        re.compile(r"^!(\d+)$"),
        re.compile(r"^mr\s*[!#]?(\d+)$", re.IGNORECASE),
        re.compile(r"merge_requests/(\d+)"),
        re.compile(r"^(\d+)$"),
    )

    def __init__(
        self,
        *,
        source_for_project: SourceControlFactory,
        project_resolver: ProjectResolver,
    ) -> None:
        self._source_for_project = source_for_project
        self._project_resolver = project_resolver

    def capabilities(self) -> Sequence[ResolverCapabilityEntry]:
        return (
            ResolverCapabilityEntry(
                provider="gitlab",
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
                    message=(
                        f"GitLabMergeRequestResolver does not handle policy={policy!r}."
                    ),
                )
            )
            return out

        remaining_chars = budget.max_total_chars
        for ref in refs:
            iid = self._parse_mr_iid(ref)
            if iid is None:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="Could not parse a merge request iid from ref.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            project = await self._resolve_project(pot_id, ref)
            if not project:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_TYPE,
                        message="No GitLab project could be resolved for this pot/ref.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            try:
                client = self._source_for_project(project)
                want_diff = policy == "snippets"
                mr_data = client.get_merge_request(project, iid, include_diff=want_diff)
            except PermissionError as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=PERMISSION_DENIED,
                        message=str(exc) or "GitLab rejected the request.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except (ConnectionError, TimeoutError) as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=SOURCE_UNREACHABLE,
                        message=f"GitLab unreachable: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except Exception as exc:
                logger.exception("gitlab MR fetch failed: %s", exc)
                out.fallbacks.append(
                    ResolverFallback(
                        code=RESOLVER_ERROR,
                        message=f"GitLab fetch raised: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue

            now_iso = datetime.now(timezone.utc).isoformat()

            if policy == "summary":
                text = _compose_mr_summary(mr_data, budget.max_chars_per_item)
                if len(text) > remaining_chars:
                    text = clamp_text(text, remaining_chars)
                if not text:
                    continue
                out.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=text,
                        title=_str_field(mr_data, "title"),
                        fetched_at=now_iso,
                        source_system="gitlab",
                        retrieval_uri=_str_field(mr_data, "url") or ref.retrieval_uri,
                    )
                )
                remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break

            elif policy == "verify":
                merged = bool(mr_data.get("merged"))
                state = str(mr_data.get("state") or "").lower()
                verification_state = (
                    "verified"
                    if (merged or state in {"opened", "closed", "merged", "locked"})
                    else "verification_failed"
                )
                out.verifications.append(
                    ResolvedVerification(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        verified=verification_state == "verified",
                        verification_state=verification_state,
                        checked_at=now_iso,
                        source_system="gitlab",
                        reason=(
                            f"MR state={state!r}, merged={merged}"
                            if verification_state != "verified"
                            else None
                        ),
                    )
                )

            elif policy == "snippets":
                diff_text = _files_as_diff(mr_data)
                body_text = _str_field(mr_data, "body") or ""
                chunks = split_diff_chunks(
                    diff_text or body_text,
                    per_item=budget.max_chars_per_item,
                    max_chunks=budget.max_snippets_per_ref,
                )
                if not chunks and body_text:
                    chunks = [(clamp_text(body_text, budget.max_chars_per_item), None)]
                for chunk, location in chunks:
                    if remaining_chars <= 0:
                        break
                    text = clamp_text(
                        chunk, min(budget.max_chars_per_item, remaining_chars)
                    )
                    if not text:
                        continue
                    out.snippets.append(
                        ResolvedSnippet(
                            ref=ref.ref,
                            source_type=ref.source_type,
                            snippet=text,
                            location=location,
                            fetched_at=now_iso,
                            source_system="gitlab",
                        )
                    )
                    remaining_chars = max(0, remaining_chars - len(text))
                if remaining_chars <= 0:
                    break

        return out

    async def _resolve_project(
        self,
        pot_id: str,
        ref: SourceReferenceRecord,
    ) -> str | None:
        result = self._project_resolver(pot_id, ref)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        return result if isinstance(result, str) and result else None

    def _parse_mr_iid(self, ref: SourceReferenceRecord) -> int | None:
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
        for pattern in cls._MR_PATTERNS:
            m = pattern.search(value)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None
        return None


def _files_as_diff(mr: dict[str, Any]) -> str:
    """Rebuild a unified diff from the API's per-file ``files`` entries.

    ``split_diff_chunks`` splits on ``diff --git`` headers, which GitLab's
    per-file payload does not carry — synthesizing them keeps one chunking
    implementation shared with the GitHub resolver.
    """
    files = mr.get("files")
    if not isinstance(files, list):
        return ""
    parts: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        patch = entry.get("patch")
        if not patch:
            continue
        filename = entry.get("filename") or entry.get("previous_filename") or ""
        previous = entry.get("previous_filename") or filename
        parts.append(f"diff --git a/{previous} b/{filename}")
        parts.append(f"--- a/{previous}")
        parts.append(f"+++ b/{filename}")
        parts.append(str(patch).rstrip("\n"))
    return "\n".join(parts)


def _compose_mr_summary(mr: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    title = _str_field(mr, "title")
    state = _str_field(mr, "state")
    merged = mr.get("merged")
    if title:
        parts.append(title)
    meta: list[str] = []
    if state:
        meta.append(f"state={state}")
    if merged is not None:
        meta.append(f"merged={'yes' if merged else 'no'}")
    author = _str_field(mr, "author")
    if author:
        meta.append(f"author={author}")
    if meta:
        parts.append("(" + ", ".join(meta) + ")")
    body = _str_field(mr, "body")
    if body:
        parts.append(body)
    return clamp_text(" ".join(parts), max_chars)


def _str_field(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
