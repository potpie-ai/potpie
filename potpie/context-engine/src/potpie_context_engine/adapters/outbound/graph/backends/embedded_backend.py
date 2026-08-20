"""Embedded ``GraphBackend`` — the OSS local default (JSON-persisted POC).

The intended ``pip install potpie`` default: a local, no-Docker, persistent store
so the CLI round-trips ``record`` → ``resolve`` across separate invocations
(each ``potpie`` call is a fresh process). This POC reuses the in-memory
capability adapters over a claim store that loads from / saves to a JSON file at
``<home>/graph.json``, persisting after every mutation.

It is a real, working backend — the POC found that an ephemeral ``in_memory``
store cannot back a process-per-call CLI, and ``embedded`` is the documented
answer. It is intentionally simple (JSON file, naive vectors); the real embedded
store is SQLite + a local vector index.

    TODO(stage-N): replace the JSON file with SQLite + a local vector index and
    real embeddings; keep this capability surface unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from potpie_context_engine.adapters.outbound.graph._local_json_atomic import (
    locked_json_store,
)
from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionRegistry,
    execution_receipts_from_json,
    execution_receipts_to_json,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    dump_store,
    load_store,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
)
from potpie_context_engine.adapters.outbound.local_paths import default_home
from potpie_context_engine.core.definition import DEFAULT_GRAPH_DEFINITION, GraphDefinition
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.lifecycle import DONE, SetupPlan, StepResult
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.mutation import MutationExecutionState
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig

_PROFILE = "embedded"
_T = TypeVar("_T")


@dataclass(slots=True)
class _EmbeddedMutation:
    backend: "EmbeddedGraphBackend"

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        return self.backend._apply_once(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
            reconciliation_config=reconciliation_config,
        )

    async def apply_async(self, *args: Any, **kwargs: Any) -> MutationResult:
        import asyncio

        return await asyncio.to_thread(self.apply, *args, **kwargs)

    def lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ):
        return self.backend._lookup_execution(
            plan,
            expected_pot_id=expected_pot_id,
            mutation_id=mutation_id,
        )

    def invalidate(self, **kwargs: Any) -> int:
        return self.backend._transact(lambda inner: inner.mutation.invalidate(**kwargs))

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        return self.backend._transact(lambda inner: inner.mutation.reset_pot(pot_id))

    def readiness(self, pot_id: str):
        return self.backend._inner.mutation.readiness(pot_id)


@dataclass(slots=True)
class _EmbeddedAnalytics:
    backend: "EmbeddedGraphBackend"

    def counts(self, pot_id: str):
        return self.backend._inner.analytics.counts(pot_id)

    def freshness(self, pot_id: str):
        return self.backend._inner.analytics.freshness(pot_id)

    def quality(self, pot_id: str):
        return self.backend._inner.analytics.quality(pot_id)

    def repair(self, pot_id: str, **kwargs: Any):
        return self.backend._transact(
            lambda inner: inner.analytics.repair(pot_id, **kwargs)
        )


@dataclass(slots=True)
class _EmbeddedSnapshot:
    backend: "EmbeddedGraphBackend"

    def export(self, **kwargs: Any):
        return self.backend._inner.snapshot.export(**kwargs)

    def import_(self, **kwargs: Any):
        return self.backend._transact(lambda inner: inner.snapshot.import_(**kwargs))


@dataclass(slots=True)
class EmbeddedGraphBackend:
    """Local JSON-persisted backend; delegates capabilities to the in-memory
    adapters and saves the store after each mutation."""

    home: Path = field(default_factory=default_home)
    embedder: EmbedderPort | None = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    shared_store: InMemoryClaimQueryStore | None = field(default=None, repr=False)
    shared_execution_registry: MutationExecutionRegistry | None = field(
        default=None,
        repr=False,
    )
    _inner: InMemoryGraphBackend = field(init=False)
    _mutation: _EmbeddedMutation = field(init=False)
    _analytics: _EmbeddedAnalytics = field(init=False)
    _snapshot: _EmbeddedSnapshot = field(init=False)

    def __post_init__(self) -> None:
        loaded_store, loaded_registry = self._load_state()
        store = self.shared_store or loaded_store
        registry = self.shared_execution_registry or loaded_registry
        store.embedder = self.embedder
        self._inner = InMemoryGraphBackend(
            store=store,
            profile_name=_PROFILE,
            embedder=self.embedder,
            definition=self.definition,
            execution_registry=registry,
        )
        self._mutation = _EmbeddedMutation(self)
        self._analytics = _EmbeddedAnalytics(self)
        self._snapshot = _EmbeddedSnapshot(self)

    @property
    def match_mode(self) -> str:
        return self._inner.match_mode

    @property
    def _path(self) -> Path:
        return self.home / "graph.json"

    def _load_state(
        self,
    ) -> tuple[InMemoryClaimQueryStore, MutationExecutionRegistry]:
        try:
            with open(self._path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return (
            load_store(payload),
            MutationExecutionRegistry(
                execution_receipts_from_json(payload.get("mutation_receipts"))
            ),
        )

    def _write_state(
        self,
        store: InMemoryClaimQueryStore,
        registry: MutationExecutionRegistry,
    ) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        payload = dump_store(store)
        payload["mutation_receipts"] = execution_receipts_to_json(registry.completed())
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(self._path)

    def _apply_once(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None,
        reconciliation_config: ReconciliationConfig | None,
    ) -> MutationResult:
        mutation_id = (
            provenance_context.mutation_id if provenance_context is not None else None
        )
        with locked_json_store(self._path):
            store, registry = self._load_state()
            if mutation_id:
                lookup = registry.lookup(
                    plan,
                    expected_pot_id=expected_pot_id,
                    mutation_id=mutation_id,
                )
                if lookup.state == MutationExecutionState.completed.value:
                    assert lookup.result is not None
                    self._adopt_state(store, registry)
                    return lookup.result
            inner = self._transaction_backend(store, registry)
            result = inner.mutation.apply(
                plan,
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
                reconciliation_config=reconciliation_config,
            )
            self._write_state(inner.store, registry)
            self._adopt_state(inner.store, registry)
            return result

    def _transact(
        self,
        operation: Callable[[InMemoryGraphBackend], _T],
    ) -> _T:
        """Reload, mutate, and atomically persist one current disk snapshot."""

        with locked_json_store(self._path):
            store, registry = self._load_state()
            inner = self._transaction_backend(store, registry)
            result = operation(inner)
            self._write_state(inner.store, registry)
            self._adopt_state(inner.store, registry)
            return result

    def _transaction_backend(
        self,
        store: InMemoryClaimQueryStore,
        registry: MutationExecutionRegistry,
    ) -> InMemoryGraphBackend:
        return InMemoryGraphBackend(
            store=store,
            profile_name=_PROFILE,
            embedder=self.embedder,
            definition=self.definition,
            execution_registry=registry,
        )

    def _adopt_state(
        self,
        store: InMemoryClaimQueryStore,
        registry: MutationExecutionRegistry,
    ) -> None:
        _replace_store(self._inner.store, store, embedder=self.embedder)
        self._inner.execution_registry.replace_completed(registry.completed())

    def _lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ):
        with locked_json_store(self._path):
            store, registry = self._load_state()
            lookup = registry.lookup(
                plan,
                expected_pot_id=expected_pot_id,
                mutation_id=mutation_id,
            )
            self._adopt_state(store, registry)
        return lookup

    # --- capability bundle (delegate to the in-memory adapters) -------------
    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def claim_query(self):
        return self._inner.claim_query

    @property
    def mutation(self):
        return self._mutation

    @property
    def semantic(self):
        return self._inner.semantic

    @property
    def inspection(self):
        return self._inner.inspection

    @property
    def analytics(self):
        return self._analytics

    @property
    def snapshot(self):
        return self._snapshot

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=_PROFILE,
            mutation=True,
            claim_query=True,
            semantic=True,
            inspection=True,
            analytics=True,
            snapshot=True,
        )

    def bind_definition(self, definition: GraphDefinition) -> EmbeddedGraphBackend:
        return EmbeddedGraphBackend(
            home=self.home,
            embedder=self.embedder,
            definition=definition,
            shared_store=self._inner.store,
            shared_execution_registry=self._inner.execution_registry,
        )

    def provision(self, plan: SetupPlan) -> StepResult:
        # Stand up the local store: ensure the home dir + persist the (possibly
        # empty) store file so resolve works immediately after setup. Idempotent.
        self._transact(lambda _inner: None)
        return StepResult(
            step="backend.provision",
            state=DONE,
            detail=f"embedded store at {self._path}",
            metadata={"profile": _PROFILE, "path": str(self._path)},
        )


def _replace_store(
    target: InMemoryClaimQueryStore,
    source: InMemoryClaimQueryStore,
    *,
    embedder: EmbedderPort | None,
) -> None:
    """Refresh a shared store without breaking definition-bound backend views."""

    target.rows[:] = source.rows
    target.entity_label_index.clear()
    target.entity_label_index.update(source.entity_label_index)
    target.entity_property_index.clear()
    target.entity_property_index.update(source.entity_property_index)
    target.embedder = embedder


__all__ = ["EmbeddedGraphBackend"]
