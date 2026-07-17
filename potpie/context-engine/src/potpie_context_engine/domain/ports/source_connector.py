"""The unified source-connector port.

A ``SourceConnector`` is the single extension point the engine exposes for
adding a new source — GitHub, Linear, Notion, Slack, Sentry, PagerDuty, a
documentation site, a CI pipeline. It bundles four concerns behind one
contract:

- Read access to artifacts → :meth:`fetch` / :meth:`list_artifacts`
- Reference-to-live-data resolution → :meth:`fetch`
- Webhook normalization → :meth:`normalize_webhook`
- Status manifest contribution → :meth:`capabilities` aggregated by the registry

All verbs are optional. A passive connector (e.g. a documentation URL
resolver) implements only :meth:`fetch` and :meth:`capabilities`; an
event-driven connector also implements :meth:`normalize_webhook` and
:meth:`list_artifacts`. Connectors declare exactly what they support via
:class:`SourceCapability` so the registry can route requests cleanly.

Rebuild plan P0: removed ``propose_plan`` (deterministic event-driven plan
compilation). Webhooks now produce raw :class:`ContextEvent`\\s that the
P5 deterministic activity layer + LLM reconciliation agent turn into
claims; no per-connector plan compiler.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

from potpie_context_engine.domain.context_events import ContextEvent
from potpie_context_engine.domain.source_connector import ConnectorScope, SourceCapability
from potpie_context_engine.domain.source_references import SourceReferenceRecord
from potpie_context_engine.domain.source_resolution import (
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
