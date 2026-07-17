"""Source-connector registry — single dispatch point for all source verbs.

Phase 2 collapsed the per-source ports / resolvers / planners / webhook
normalizers into one :class:`SourceConnectorPort` and routed every call
through this registry. The application layer never imports a concrete
connector; it asks the registry "who handles this ref / webhook / kind?"
and dispatches.

Replaces (Phase 2 deletions):

- ``adapters/outbound/source_resolvers/composite.py`` — the registry now
  is the resolver fan-out.
- ``domain/ports/pot_source_listing.py`` — status manifest is built from
  registered connectors via :meth:`manifest_for_pot`.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

from potpie_context_core.context_events import ContextEvent
from potpie_context_engine.domain.ports.source_connector import SourceConnectorPort
from potpie_context_engine.domain.source_connector import (
    ConnectorManifest,
    SourceCapability,
    merge_capability_policies,
)
from potpie_context_core.source_references import SourceReferenceRecord, normalize_source_policy
from potpie_context_engine.domain.source_resolution import (
    BUDGET_EXCEEDED,
    RESOLVER_ERROR,
    UNSUPPORTED_SOURCE_POLICY,
    UNSUPPORTED_SOURCE_TYPE,
    ResolverAuthContext,
    ResolverBudget,
    ResolverFallback,
    SourceResolutionResult,
)

logger = logging.getLogger(__name__)


class SourceConnectorRegistry:
    """Look up connectors by kind, by ref, by webhook source.

    Connectors are registered explicitly during container bootstrap. The
    registry never instantiates connectors itself — that is the host's
    responsibility, so per-pot credentials and scope live in the
    connector boundary, not in shared application code.
    """

    def __init__(self) -> None:
        self._by_kind: dict[str, SourceConnectorPort] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, connector: SourceConnectorPort) -> None:
        kind = connector.kind().lower()
        if kind in self._by_kind:
            logger.warning("Replacing already-registered connector kind=%s", kind)
        self._by_kind[kind] = connector

    def all(self) -> Sequence[SourceConnectorPort]:
        return list(self._by_kind.values())

    def get(self, kind: str) -> SourceConnectorPort | None:
        return self._by_kind.get((kind or "").lower())

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def find_for_ref(
        self,
        ref: SourceReferenceRecord,
        policy: str,
    ) -> SourceConnectorPort | None:
        """Pick the connector whose capabilities cover this ref + policy."""
        provider_key = (ref.source_system or ref.source_type or "").lower()
        type_key = (ref.source_type or "").lower()
        for connector in self._by_kind.values():
            for cap in connector.capabilities():
                if not cap.fetch_capable:
                    continue
                if policy not in cap.policies:
                    continue
                cap_provider = cap.provider.lower()
                cap_kind = cap.source_kind.lower()
                if cap_provider in {provider_key, type_key} or cap_kind in {
                    provider_key,
                    type_key,
                }:
                    return connector
        return None

    def find_for_webhook(self, source_kind: str) -> SourceConnectorPort | None:
        """Look up the connector responsible for a webhook source kind."""
        connector = self.get(source_kind)
        if connector is not None:
            for cap in connector.capabilities():
                if cap.webhook_capable:
                    return connector
        return None

    def find_for_event(self, event: ContextEvent) -> SourceConnectorPort | None:
        """Pick the connector that owns an event's source system."""
        ss = (event.source_system or "").lower()
        return self.get(ss)

    # ------------------------------------------------------------------
    # Aggregated views
    # ------------------------------------------------------------------
    def aggregated_capabilities(self) -> list[SourceCapability]:
        """All registered connectors' capabilities, deduped by ``(provider, source_kind)``."""
        flat: list[SourceCapability] = []
        for c in self._by_kind.values():
            flat.extend(c.capabilities())
        return merge_capability_policies(flat)

    def capabilities(self) -> list[SourceCapability]:
        """Fetch-capable subset of :meth:`aggregated_capabilities`.

        This is the read surface ``context_status`` uses to advertise which
        ``(provider, source_kind)`` pairings can actually resolve refs under
        a non-``references_only`` policy.
        """
        return [cap for cap in self.aggregated_capabilities() if cap.fetch_capable]

    def manifest_for_pot(self, pot_id: str) -> list[ConnectorManifest]:
        """Build the per-pot connector manifest used by ``context_status``."""
        del pot_id  # Phase 2 returns the global manifest; Phase 5 will scope per pot.
        out: list[ConnectorManifest] = []
        for connector in self._by_kind.values():
            caps = tuple(connector.capabilities())
            out.append(
                ConnectorManifest(
                    kind=connector.kind(),
                    enabled=bool(caps),
                    capabilities=caps,
                    health="ready" if caps else "no_capabilities",
                )
            )
        return sorted(out, key=lambda m: m.kind)

    # ------------------------------------------------------------------
    # Resolver fan-out (replaces CompositeSourceResolver)
    # ------------------------------------------------------------------
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
        if policy == "references_only":
            return out

        unmatched: list[SourceReferenceRecord] = []
        by_connector: dict[str, list[SourceReferenceRecord]] = {}
        connector_index: dict[str, SourceConnectorPort] = {}
        for ref in refs:
            connector = self.find_for_ref(ref, policy)
            if connector is None:
                unmatched.append(ref)
                continue
            kind = connector.kind()
            by_connector.setdefault(kind, []).append(ref)
            connector_index[kind] = connector

        for ref in unmatched:
            out.fallbacks.append(
                ResolverFallback(
                    code=UNSUPPORTED_SOURCE_TYPE,
                    message=(
                        f"No connector registered for source_type={ref.source_type!r} "
                        f"under policy={policy!r}."
                    ),
                    ref=ref.ref,
                    source_type=ref.source_type,
                )
            )

        remaining_chars = budget.max_total_chars
        refs_served = 0

        for kind, child_refs in by_connector.items():
            connector = connector_index[kind]
            allowed = max(0, budget.max_refs - refs_served)
            if allowed <= 0:
                for ref in child_refs:
                    out.fallbacks.append(
                        ResolverFallback(
                            code=BUDGET_EXCEEDED,
                            message="Request hit the max_refs budget.",
                            ref=ref.ref,
                            source_type=ref.source_type,
                        )
                    )
                continue
            served = child_refs[:allowed]
            overflow = child_refs[allowed:]
            for ref in overflow:
                out.fallbacks.append(
                    ResolverFallback(
                        code=BUDGET_EXCEEDED,
                        message="Request hit the max_refs budget.",
                        ref=ref.ref,
                        source_type=ref.source_type,
                    )
                )
            child_budget = ResolverBudget(
                max_refs=len(served),
                max_chars_per_item=budget.max_chars_per_item,
                max_total_chars=max(0, remaining_chars),
                max_snippets_per_ref=budget.max_snippets_per_ref,
                timeout_ms=budget.timeout_ms,
            )
            try:
                child_result = await connector.fetch(
                    pot_id=pot_id,
                    refs=served,
                    source_policy=policy,
                    budget=child_budget,
                    auth=auth,
                )
            except NotImplementedError as exc:
                out.fallbacks.append(
                    ResolverFallback(
                        code=UNSUPPORTED_SOURCE_POLICY,
                        message=str(exc) or f"policy={policy!r} not implemented",
                    )
                )
                continue
            except Exception as exc:
                logger.exception("connector fetch failed: %s", exc)
                out.fallbacks.append(
                    ResolverFallback(
                        code=RESOLVER_ERROR,
                        message=f"Connector raised: {exc}",
                    )
                )
                continue

            out.extend(child_result)
            refs_served += len(served)
            remaining_chars = max(0, remaining_chars - child_result.total_chars())
            if remaining_chars <= 0:
                break

        return out

    # ------------------------------------------------------------------
    # Webhook dispatch
    # ------------------------------------------------------------------
    def normalize_webhook(
        self,
        source_kind: str,
        payload: bytes,
        headers: Mapping[str, str],
    ) -> ContextEvent | None:
        connector = self.find_for_webhook(source_kind)
        if connector is None:
            return None
        return connector.normalize_webhook(payload, headers)

    def list_artifacts(
        self,
        kind: str,
        scope_payload: dict,
        pot_id: str,
    ) -> Iterable[SourceReferenceRecord]:
        from potpie_context_engine.domain.source_connector import ConnectorScope

        connector = self.get(kind)
        if connector is None:
            return ()
        return connector.list_artifacts(
            ConnectorScope(pot_id=pot_id, scope=scope_payload)
        )
