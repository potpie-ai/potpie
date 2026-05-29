"""Linear source connector — single point of contact for everything Linear.

Phase 2 of the rebuild collapsed three Linear-shaped surfaces into this
one class:

- ``LinearIssueResolver`` (resolve refs) → :meth:`fetch`
- ``application/use_cases/normalize_linear_webhook.py`` →
  :meth:`normalize_webhook`
- ``adapters/outbound/reconciliation/linear_issue_plan.py`` →
  :meth:`propose_plan`

The application layer no longer imports anything in this module.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping, Sequence

from adapters.outbound.connectors.linear.events import linear_issue_from_payload
from adapters.outbound.connectors.linear.fetcher import LinearIssueFetcher
from adapters.outbound.connectors.linear.plan import build_linear_issue_plan
from adapters.outbound.connectors.linear.resolver import LinearIssueResolver
from adapters.outbound.connectors.linear.webhook import (
    LinearWebhookError,
    linear_payload_to_event,
)
from domain.context_events import ContextEvent, EventRef
from domain.ports.source_connector import SourceConnectorPort
from domain.reconciliation import ReconciliationPlan
from domain.source_connector import ConnectorScope, SourceCapability
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    SourceResolutionResult,
)

logger = logging.getLogger(__name__)


class LinearConnector(SourceConnectorPort):
    """The unified Linear connector.

    Constructed with an injected :class:`LinearIssueFetcher` (the host's
    Linear GraphQL client). Tests pass a fake fetcher.
    """

    KIND = "linear"

    def __init__(self, *, fetcher: LinearIssueFetcher) -> None:
        self._fetcher = fetcher
        self._resolver = LinearIssueResolver(fetcher=fetcher)

    # ------------------------------------------------------------------
    # SourceConnectorPort
    # ------------------------------------------------------------------
    def kind(self) -> str:
        return self.KIND

    def capabilities(self) -> Sequence[SourceCapability]:
        out: list[SourceCapability] = []
        for cap in self._resolver.capabilities():
            out.append(
                SourceCapability(
                    provider=cap.provider,
                    source_kind=cap.source_kind,
                    policies=cap.policies,
                    fetch_capable=True,
                    list_capable=False,
                    webhook_capable=True,
                    plan_capable=True,
                    sync_capable=False,
                )
            )
        return out

    def list_artifacts(
        self,
        scope: ConnectorScope,
    ) -> Iterable[SourceReferenceRecord]:
        # Phase 2: Linear backfill is not yet wired through list_artifacts.
        del scope
        return ()

    def normalize_webhook(
        self,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        del headers  # Linear does not require signature checks today.
        try:
            body = json.loads(payload.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return None
        try:
            return linear_payload_to_event(body)
        except LinearWebhookError as exc:
            logger.info("linear webhook ignored: %s", exc)
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
        return await self._resolver.resolve(
            pot_id=pot_id,
            refs=refs,
            source_policy=source_policy,
            budget=budget,
            auth=auth,
        )

    def propose_plan(
        self,
        event: ContextEvent,
        context_graph: object,
    ) -> ReconciliationPlan | None:
        del context_graph
        if event.event_type not in {"linear_issue", "linear_comment"}:
            return None

        payload = event.payload or {}
        action = str(payload.get("action") or event.action or "").strip().lower()
        issue_payload = payload.get("issue")
        if not isinstance(issue_payload, dict):
            return None
        try:
            issue = linear_issue_from_payload(issue_payload)
        except Exception as exc:
            logger.warning("linear propose_plan: parse failed: %s", exc)
            return None

        comment_payload = payload.get("comment")
        comment = None
        if isinstance(comment_payload, dict):
            try:
                # Reuse the issue parser's comment helper by feeding a one-shot dict.
                from adapters.outbound.connectors.linear.events import (
                    LinearComment,
                    parse_linear_datetime,
                )

                comment = LinearComment(
                    id=str(comment_payload.get("id") or ""),
                    body=str(comment_payload.get("body") or ""),
                    created_at=parse_linear_datetime(
                        comment_payload.get("createdAt")
                    ),
                )
            except Exception as exc:
                logger.warning("linear propose_plan: comment parse failed: %s", exc)

        prev_state_payload = payload.get("previous_state")
        previous_state = None
        if isinstance(prev_state_payload, dict):
            from adapters.outbound.connectors.linear.events import LinearState

            previous_state = LinearState(
                name=str(prev_state_payload.get("name") or ""),
                type=prev_state_payload.get("type"),
            )

        from adapters.outbound.connectors.linear.events import LinearIssueEvent

        issue_event: Any = LinearIssueEvent(
            action=action or "update",
            issue=issue,
            comment=comment,
            previous_state=previous_state,
        )

        event_ref = EventRef(
            event_id=event.event_id,
            source_system="linear",
            pot_id=event.pot_id,
        )
        return build_linear_issue_plan(
            event_ref=event_ref,
            issue_event=issue_event,
        )


__all__ = ["LinearConnector"]
