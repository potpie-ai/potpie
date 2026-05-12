"""The unified source-connector port.

A ``SourceConnector`` is the single extension point the engine exposes for
adding a new source — GitHub, Linear, Notion, Slack, Sentry, PagerDuty, a
documentation site, a CI pipeline. It bundles five concerns behind one
contract:

- Read access to artifacts → :meth:`fetch` / :meth:`list_artifacts`
- Reference-to-live-data resolution → :meth:`fetch`
- Webhook normalization → :meth:`normalize_webhook`
- Deterministic plan compilation → :meth:`propose_plan`
- Status manifest contribution → :meth:`capabilities` aggregated by the registry

All five verbs are optional. A passive connector (e.g. a documentation
URL resolver) implements only :meth:`fetch` and :meth:`capabilities`; a
deterministic event-driven connector implements all five. Connectors
declare exactly what they support via :class:`SourceCapability` so the
registry can route requests cleanly.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

from domain.context_events import ContextEvent
from domain.reconciliation import ReconciliationPlan
from domain.source_connector import ConnectorScope, SourceCapability
from domain.source_references import SourceReferenceRecord
from domain.source_resolution import (
    ResolverAuthContext,
    ResolverBudget,
    SourceResolutionResult,
)


class SourceConnectorPort(Protocol):
    """One source backend behind a stable contract."""

    def kind(self) -> str:
        """Stable connector identifier (e.g. ``"github"``, ``"linear"``, ``"notion"``)."""
        ...

    def capabilities(self) -> Sequence[SourceCapability]:
        """Advertise which ``(provider, source_kind)`` pairs this connector serves."""
        ...

    def list_artifacts(
        self,
        scope: ConnectorScope,
    ) -> Iterable[SourceReferenceRecord]:
        """Enumerate artifacts in scope (used by backfill).

        Connectors that cannot enumerate (passive resolvers) yield an empty
        iterable and set ``list_capable=False`` in their capability manifest.
        """
        ...

    def normalize_webhook(
        self,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        """Turn a raw webhook delivery into a canonical :class:`ContextEvent`.

        Return ``None`` for events the connector does not care about. Raise
        on signature failures so the inbound adapter can return 401.
        """
        ...

    async def fetch(
        self,
        *,
        pot_id: str,
        refs: Sequence[SourceReferenceRecord],
        source_policy: str,
        budget: ResolverBudget,
        auth: ResolverAuthContext,
    ) -> SourceResolutionResult:
        """Resolve refs into summaries / snippets / verifications under ``source_policy``."""
        ...

    def propose_plan(
        self,
        event: ContextEvent,
        context_graph: object,  # ContextGraphPort — typed dynamically to avoid cycles
    ) -> ReconciliationPlan | None:
        """Propose a deterministic :class:`ReconciliationPlan` for an event.

        Connectors with deterministic mappings (e.g. GitHub PR merged →
        canonical entities + edges) return a plan. Connectors that only
        contribute resolution return ``None``; the reconciliation agent
        plans from the raw event.
        """
        ...
