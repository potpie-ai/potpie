"""Notion docs connector — smoke test for the connector contract.

Implements :class:`SourceConnectorPort` for Notion to prove the interface is
right by adding a source the engine has not seen before. The connector is
intentionally minimal:

- :meth:`capabilities` advertises ``("notion", "page")`` with policies
  ``summary`` and ``snippets``.
- :meth:`fetch` resolves Notion page refs to summaries / snippets via an
  injected :class:`NotionPageFetcher` (the host wires up the actual
  Notion HTTP client; tests pass a fake).
- :meth:`normalize_webhook` returns ``None`` because Notion's webhooks
  are still in beta and we have not committed to a payload shape; the
  capability flag advertises ``webhook_capable=False``.
- :meth:`list_artifacts` is supported when the host's fetcher exposes a
  list operation; otherwise empty.

Rebuild plan P0: removed ``propose_plan`` (deterministic page-event plan
compilation). Fetched page content is evidence for the harness or parked LLM
planner; this connector does not write graph claims directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence

from potpie_context_engine.domain.context_events import ContextEvent
from potpie_context_engine.domain.ports.source_connector import SourceConnectorPort
from potpie_context_engine.domain.source_connector import (
    ConnectorScope,
    SourceCapability,
)
from potpie_context_engine.domain.source_references import SourceReferenceRecord
from potpie_context_engine.domain.source_resolution import (
    PERMISSION_DENIED,
    SOURCE_UNREACHABLE,
    UNSUPPORTED_SOURCE_POLICY,
    ResolvedSnippet,
    ResolvedSummary,
    ResolverAuthContext,
    ResolverBudget,
    ResolverFallback,
    SourceResolutionResult,
    clamp_text,
)

logger = logging.getLogger(__name__)


class NotionPageFetcher(Protocol):
    """Connector-internal read surface for Notion pages.

    Implementations call the Notion HTTP API (or a recorded fixture for
    tests) and return a normalized dict with ``title`` and ``content``.
    Returning ``None`` signals "not found"; raising is reserved for
    transport / auth errors so the connector can translate to fallback
    codes.
    """

    def get_page(self, page_id: str) -> dict[str, Any] | None: ...

    def list_pages(self, workspace_id: str) -> Iterable[dict[str, Any]]: ...


class NotionConnector(SourceConnectorPort):
    """Phase 2 smoke-test connector for Notion docs."""

    KIND = "notion"

    def __init__(self, *, fetcher: NotionPageFetcher | None = None) -> None:
        self._fetcher = fetcher

    # ------------------------------------------------------------------
    # SourceConnectorPort
    # ------------------------------------------------------------------
    def kind(self) -> str:
        return self.KIND

    def capabilities(self) -> Sequence[SourceCapability]:
        if self._fetcher is None:
            return (
                SourceCapability(
                    provider="notion",
                    source_kind="page",
                    policies=frozenset(),
                    fetch_capable=False,
                    list_capable=False,
                    webhook_capable=False,
                    sync_capable=False,
                    notes="notion fetcher not configured",
                ),
            )
        return (
            SourceCapability(
                provider="notion",
                source_kind="page",
                policies=frozenset({"summary", "snippets"}),
                fetch_capable=True,
                list_capable=True,
                webhook_capable=False,
                sync_capable=False,
            ),
        )

    def list_artifacts(
        self,
        scope: ConnectorScope,
    ) -> Iterable[SourceReferenceRecord]:
        if self._fetcher is None:
            return ()
        workspace_id = (scope.scope.get("workspace_id") or "").strip()
        if not workspace_id:
            return ()
        out: list[SourceReferenceRecord] = []
        try:
            for page in self._fetcher.list_pages(workspace_id):
                page_id = str(page.get("id") or "").strip()
                if not page_id:
                    continue
                out.append(
                    SourceReferenceRecord(
                        ref=f"notion:page:{page_id}",
                        source_type="page",
                        source_system="notion",
                        external_id=page_id,
                        title=str(page.get("title") or "") or None,
                        uri=str(page.get("url") or "") or None,
                        retrieval_uri=str(page.get("url") or "") or None,
                        fetchable=True,
                        access="allowed",
                    )
                )
        except Exception as exc:
            logger.warning("notion list_artifacts failed: %s", exc)
        return out

    def normalize_webhook(
        self,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        del payload, headers  # Notion webhooks are not consumed yet.
        return None

    async def fetch(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        del pot_id, auth  # Notion auth is held inside the fetcher.
        result = SourceResolutionResult()
        if self._fetcher is None:
            for ref in refs:
                result.fallbacks.append(
                    ResolverFallback(
                        code=SOURCE_UNREACHABLE,
                        message="notion fetcher not configured",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
            return result
        if source_policy not in {"summary", "snippets"}:
            for ref in refs:
                result.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_POLICY,
                        message=f"notion: policy={source_policy!r} not supported",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
            return result

        fetched_at = datetime.now(timezone.utc).isoformat()
        chars_used = 0
        for ref in refs[: budget.max_refs]:
            page_id = _page_id_from_ref(ref)
            if not page_id:
                continue
            try:
                page = self._fetcher.get_page(page_id)
            except PermissionError:
                result.fallbacks.append(
                    ResolverFallback(
                        code=PERMISSION_DENIED,
                        message="notion permission denied",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            except Exception as exc:
                result.fallbacks.append(
                    ResolverFallback(
                        code=SOURCE_UNREACHABLE,
                        message=f"notion fetch failed: {exc}",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
                continue
            if page is None:
                continue
            title = str(page.get("title") or "") or None
            content = str(page.get("content") or "")
            allowance = max(0, budget.max_total_chars - chars_used)
            if allowance <= 0:
                break
            text = clamp_text(content, min(budget.max_chars_per_item, allowance))
            chars_used += len(text)
            if source_policy == "summary":
                result.summaries.append(
                    ResolvedSummary(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        summary=text,
                        title=title,
                        fetched_at=fetched_at,
                        source_system="notion",
                        retrieval_uri=ref.retrieval_uri or ref.uri,
                    )
                )
            else:
                result.snippets.append(
                    ResolvedSnippet(
                        ref=ref.ref,
                        source_type=ref.source_type,
                        snippet=text,
                        location=title,
                        fetched_at=fetched_at,
                        source_system="notion",
                    )
                )
        return result


def _page_id_from_ref(ref: SourceReferenceRecord) -> str | None:
    if ref.external_id:
        return ref.external_id
    parts = (ref.ref or "").split(":")
    if len(parts) >= 3 and parts[0].lower() == "notion":
        return parts[-1]
    return None


__all__ = ["NotionConnector", "NotionPageFetcher"]
