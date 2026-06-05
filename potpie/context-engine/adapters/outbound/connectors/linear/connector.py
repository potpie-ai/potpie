"""Linear source connector — single point of contact for everything Linear.

Bundles three Linear-shaped surfaces behind one class:

- ``LinearIssueResolver`` (resolve refs) → :meth:`fetch`
- ``application/use_cases/normalize_linear_webhook.py`` →
  :meth:`normalize_webhook`
- Backfill enumeration → :meth:`list_artifacts`

The application layer no longer imports anything in this module.

Rebuild plan P0: removed ``propose_plan`` (deterministic issue-event
plan compilation). Webhooks now produce raw :class:`ContextEvent`\\s;
the P5 deterministic activity layer + LLM reconciliation agent turn
them into :RELATES_TO claims.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Iterable, Mapping, Sequence

from adapters.outbound.connectors.linear.fetcher import LinearIssueFetcher
from adapters.outbound.connectors.linear.resolver import LinearIssueResolver
from adapters.outbound.connectors.linear.webhook import (
    LinearWebhookError,
    linear_payload_to_event,
)
from domain.context_events import ContextEvent
from domain.ports.source_connector import SourceConnectorPort
from domain.source_connector import ConnectorScope, SourceCapability
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    SourceResolutionResult,
)

logger = logging.getLogger(__name__)


def _verify_linear_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """Constant-time check of Linear's ``Linear-Signature`` header.

    Linear signs the **raw** request body with HMAC-SHA256 and sends the
    hex digest (no scheme prefix). A missing/empty signature is rejected.
    """
    if not signature:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature.strip(), expected)


class LinearConnector(SourceConnectorPort):
    """The unified Linear connector.

    Constructed with an injected :class:`LinearIssueFetcher` (the host's
    Linear GraphQL client). Tests pass a fake fetcher.
    """

    KIND = "linear"

    def __init__(
        self,
        *,
        fetcher: LinearIssueFetcher,
        webhook_secret: str | None = None,
        allow_unsigned: bool = False,
    ) -> None:
        self._fetcher = fetcher
        self._resolver = LinearIssueResolver(fetcher=fetcher)
        self._webhook_secret = (webhook_secret or "").strip() or None
        self._allow_unsigned = allow_unsigned
        self._unsigned_warned = False

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
        signature = headers.get("Linear-Signature") or headers.get("linear-signature")
        if self._webhook_secret is None:
            # Fail closed, same posture as the GitHub connector: an unsigned
            # webhook is an unauthenticated graph write + a free trigger for
            # expensive agent work.
            if not self._allow_unsigned:
                raise PermissionError(
                    "linear webhook signature required: LINEAR_WEBHOOK_SECRET "
                    "is not configured (set it, or set "
                    "CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS=1 for local dev "
                    "only)"
                )
            if not self._unsigned_warned:
                logger.warning(
                    "SECURITY: LINEAR_WEBHOOK_SECRET is unset and "
                    "CONTEXT_ENGINE_ALLOW_UNSIGNED_WEBHOOKS is enabled — "
                    "linear webhooks are being accepted UNAUTHENTICATED. "
                    "Never use this in a network-reachable deployment."
                )
                self._unsigned_warned = True
        elif not _verify_linear_signature(payload, signature, self._webhook_secret):
            raise PermissionError("linear webhook signature mismatch")
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


__all__ = ["LinearConnector"]
