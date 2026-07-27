"""Public backend authoring kit and reusable in-memory runtime fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from potpie_context_core.api import (
    ClaimQueryFilter,
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
    GraphInboxItem,
    GraphMutationPlanRecord,
    GraphRuntime,
    build_graph_runtime,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.testing.conformance import (
    GraphBackendConformanceMixin,
    run_graph_backend_conformance,
)


@dataclass(slots=True)
class InMemoryGraphPlanStore:
    _records: dict[tuple[str, str], GraphMutationPlanRecord] = field(
        default_factory=dict
    )

    def save(self, record: GraphMutationPlanRecord) -> None:
        self._records[(record.pot_id, record.plan_id)] = record

    def get(self, *, pot_id: str, plan_id: str) -> GraphMutationPlanRecord | None:
        return self._records.get((pot_id, plan_id))

    def list(
        self,
        *,
        pot_id: str,
        plan_id: str | None = None,
        mutation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphMutationPlanRecord, ...]:
        records = [
            record
            for (record_pot, _), record in self._records.items()
            if record_pot == pot_id
            and (plan_id is None or record.plan_id == plan_id)
            and (mutation_id is None or record.mutation_id == mutation_id)
            and (since is None or record.created_at >= since)
            and (until is None or record.created_at <= until)
        ]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return tuple(records[:limit] if limit is not None else records)

    async def save_async(self, record: GraphMutationPlanRecord) -> None:
        self.save(record)

    async def get_async(
        self, *, pot_id: str, plan_id: str
    ) -> GraphMutationPlanRecord | None:
        return self.get(pot_id=pot_id, plan_id=plan_id)

    async def list_async(self, **kwargs):
        return self.list(**kwargs)


@dataclass(slots=True)
class InMemoryGraphInboxStore:
    _items: dict[tuple[str, str], GraphInboxItem] = field(default_factory=dict)

    def save(self, item: GraphInboxItem) -> None:
        self._items[(item.pot_id, item.item_id)] = item

    def get(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        return self._items.get((pot_id, item_id))

    def list(
        self,
        *,
        pot_id: str,
        status: tuple[str, ...] = (),
        claimed_by: str | None = None,
        suspected_subgraph: str | None = None,
        source_ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphInboxItem, ...]:
        items = [
            item
            for (item_pot, _), item in self._items.items()
            if item_pot == pot_id
            and (not status or item.status in status)
            and (claimed_by is None or item.claimed_by == claimed_by)
            and (
                suspected_subgraph is None
                or suspected_subgraph in item.suspected_subgraphs
            )
            and (source_ref is None or source_ref in item.source_refs)
            and (since is None or item.created_at >= since)
            and (until is None or item.created_at <= until)
        ]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(items[:limit] if limit is not None else items)

    async def save_async(self, item: GraphInboxItem) -> None:
        self.save(item)

    async def get_async(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        return self.get(pot_id=pot_id, item_id=item_id)

    async def list_async(self, **kwargs):
        return self.list(**kwargs)


def build_test_backend(
    *, definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
) -> InMemoryGraphBackend:
    return InMemoryGraphBackend(definition=definition)


def build_test_graph_runtime(
    *, definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
) -> GraphRuntime:
    return build_graph_runtime(
        build_test_backend(definition=definition),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        definition,
    )


def assert_graph_backend_conforms(
    backend_factory: Callable[[], object],
) -> None:
    """Small dependency-free smoke suite reusable by third-party backends.

    The package's pytest conformance module layers the full behavioral suite
    over this startup/capability/isolation check.
    """

    backend = backend_factory().bind_definition(DEFAULT_GRAPH_DEFINITION)
    capabilities = backend.capabilities()
    if not capabilities.mutation or not capabilities.claim_query:
        raise AssertionError("backend must implement mutation and claim_query")
    first = backend.mutation.readiness("conformance:first")
    second = backend.mutation.readiness("conformance:second")
    if not first.ready or not second.ready:
        raise AssertionError("backend must be ready for two independent pots")
    backend.mutation.reset_pot("conformance:first")
    leaked = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="conformance:second")
    )
    if any(row.pot_id != "conformance:second" for row in leaked):
        raise AssertionError("backend claim queries are not pot-isolated")


__all__ = [
    "GraphBackendConformanceMixin",
    "InMemoryGraphBackend",
    "InMemoryGraphInboxStore",
    "InMemoryGraphPlanStore",
    "assert_graph_backend_conforms",
    "build_test_backend",
    "build_test_graph_runtime",
    "run_graph_backend_conformance",
]
