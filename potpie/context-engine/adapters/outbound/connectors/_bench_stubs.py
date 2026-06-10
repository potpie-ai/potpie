"""Passive stub source connectors for the benchmark harness.

These connectors register a `SourceConnectorPort` under a stable `kind`
so the bench harness can submit events tagged with that source. They
hold no live read or fetch capability — the bench advertises
`fetch_capable=False` everywhere.

Rebuild plan P0: removed `propose_plan` (the deterministic event-driven
plan compiler). Bench scenarios now flow through the canonical
ingestion path:

    POST /events/reconcile  →  LLM reconciliation agent  →
        canonical claim writer (:RELATES_TO edges)

The agent is responsible for extracting entities + claims from the
payload. There are no more "stub plan seeds" the agent gets a chance
to enrich; the agent owns extraction outright.

If a future bench needs seed graph state, the harness should submit explicit
semantic mutations with evidence instead of adding a per-connector plan
compiler.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

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


class _PassiveStubConnector(SourceConnectorPort):
    """Common shape for the bench stub connectors.

    Subclasses override ``KIND`` and ``SOURCE_KIND``. The connector
    advertises no capabilities — the bench submits events directly to
    ``/events/reconcile``; this registration only exists so the
    `source_system` lookup in `find_for_event` resolves cleanly.
    """

    KIND = "_passive_stub"
    SOURCE_KIND = "record"

    def kind(self) -> str:
        return self.KIND

    def capabilities(self) -> Sequence[SourceCapability]:
        return (
            SourceCapability(
                provider=self.KIND,
                source_kind=self.SOURCE_KIND,
                policies=frozenset(),
                fetch_capable=False,
                list_capable=False,
                webhook_capable=False,
                sync_capable=False,
                notes=f"bench stub: passive {self.KIND} reader (no live read access)",
            ),
        )

    def list_artifacts(
        self,
        scope: ConnectorScope,
    ) -> Iterable[SourceReferenceRecord]:
        del scope
        return ()

    def normalize_webhook(
        self,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        del payload, headers
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
        del pot_id, refs, source_policy, budget, auth
        return SourceResolutionResult()


class SlackStubConnector(_PassiveStubConnector):
    KIND = "slack"
    SOURCE_KIND = "message"


class RepoDocsStubConnector(_PassiveStubConnector):
    KIND = "repo_docs"
    SOURCE_KIND = "document"


class AlertingStubConnector(_PassiveStubConnector):
    KIND = "alerting"
    SOURCE_KIND = "alert"


class DeployStubConnector(_PassiveStubConnector):
    KIND = "deploy"
    SOURCE_KIND = "deployment"


__all__ = [
    "SlackStubConnector",
    "RepoDocsStubConnector",
    "AlertingStubConnector",
    "DeployStubConnector",
]
