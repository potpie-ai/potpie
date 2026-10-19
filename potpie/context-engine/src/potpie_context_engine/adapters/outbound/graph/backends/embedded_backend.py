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
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.mutation_execution import (
    MutationCapabilities,
    MutationPersistenceError,
)
from potpie_context_engine.core.ports.graph.mutation import MutationExecutionLookup
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.core.reconciliation import MutationBatch, MutationResult
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig

from potpie_context_engine.adapters.outbound.graph._local_json_atomic import (
    locked_json_store,
    durable_json_replace,
    strict_json_load,
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
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.provisioning import BackendProvisionResult

_PROFILE = "embedded"
_T = TypeVar("_T")


@dataclass(slots=True)
class _EmbeddedMutation:
    backend: EmbeddedGraphBackend

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

    @property
    def capabilities(self) -> MutationCapabilities:
        return MutationCapabilities(
            atomic_writes=True,
            revision_check=True,
            execution_lookup=True,
            durable_receipts=True,
        )

    async def current_version_async(self, pot_id: str) -> int:
        import asyncio

        return await asyncio.to_thread(self.current_version, pot_id)

    async def compare_and_apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        expected_version: int,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        import asyncio

        return await asyncio.to_thread(
            self.compare_and_apply,
            plan,
            expected_pot_id=expected_pot_id,
            expected_version=expected_version,
            provenance_context=provenance_context,
            reconciliation_config=reconciliation_config,
        )

    async def lookup_execution_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        import asyncio

        return await asyncio.to_thread(
            self.lookup_execution,
            plan,
            expected_pot_id=expected_pot_id,
            mutation_id=mutation_id,
        )

    def current_version(self, pot_id: str) -> int:
        return self.backend._current_version(pot_id)

    def compare_and_apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        expected_version: int,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        return self.backend._apply_once(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
            reconciliation_config=reconciliation_config,
            expected_version=expected_version,
        )

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        import asyncio

        return await asyncio.to_thread(
            self.apply,
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
            reconciliation_config=reconciliation_config,
        )

    def lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        return self.backend._lookup_execution(
            plan,
            expected_pot_id=expected_pot_id,
            mutation_id=mutation_id,
        )

    def invalidate(self, **kwargs: Any) -> int:
        return self.backend._transact(lambda inner: inner.mutation.invalidate(**kwargs))

    def reset_pot(
        self, pot_id: str, *, mutation_id: str | None = None
    ) -> dict[str, Any]:
        result = self.apply(
            MutationBatch(reset_pot=True),
            expected_pot_id=pot_id,
            provenance_context=ProvenanceContext(mutation_id=mutation_id),
        )
        return {
            "removed_claims": result.administrative_summary["removed_claims"],
            "mutation_id": result.mutation_id,
            "execution": result.execution.to_dict()
            if result.execution is not None
            else None,
        }

    async def reset_pot_async(
        self, pot_id: str, *, mutation_id: str | None = None
    ) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self.reset_pot, pot_id, mutation_id=mutation_id)

    def readiness(self, pot_id: str):
        return self.backend._inner.mutation.readiness(pot_id)


@dataclass(slots=True)
class _EmbeddedAnalytics:
    backend: EmbeddedGraphBackend

    def counts(self, pot_id: str):
        return self.backend._read(lambda inner: inner.analytics.counts(pot_id))

    def freshness(self, pot_id: str):
        return self.backend._read(lambda inner: inner.analytics.freshness(pot_id))

    def quality(self, pot_id: str):
        return self.backend._read(lambda inner: inner.analytics.quality(pot_id))

    def repair(self, pot_id: str, **kwargs: Any):
        return self.backend._transact(
            lambda inner: inner.analytics.repair(pot_id, **kwargs)
        )


@dataclass(slots=True)
class _EmbeddedSnapshot:
    backend: EmbeddedGraphBackend

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        payload = self.export_data(pot_id=pot_id)
        with open(destination, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
        return SnapshotManifest(
            pot_id=pot_id,
            location=destination,
            format_version="2",
            entity_count=len(payload["entities"]),
            claim_count=len(payload["claims"]),
        )

    def export_data(self, *, pot_id: str) -> dict[str, Any]:
        return self.backend._read(
            lambda inner: inner.snapshot.export_data(pot_id=pot_id)
        )

    def import_(
        self, *, pot_id: str, source: str, mutation_id: str | None = None
    ) -> SnapshotManifest:
        with open(source, encoding="utf-8") as fh:
            payload = json.load(fh)
        result = self.import_data(
            pot_id=pot_id, payload=payload, mutation_id=mutation_id
        )
        return SnapshotManifest(
            pot_id=result.pot_id,
            location=source,
            format_version=result.format_version,
            entity_count=result.entity_count,
            claim_count=result.claim_count,
            metadata=result.metadata,
        )

    def import_data(
        self,
        *,
        pot_id: str,
        payload: Mapping[str, Any],
        mutation_id: str | None = None,
    ) -> SnapshotManifest:
        from potpie_context_engine.adapters.outbound.graph.memory_admin import (
            import_snapshot,
        )

        return import_snapshot(
            self.backend.mutation,
            pot_id=pot_id,
            payload=payload,
            mutation_id=mutation_id,
        )


@dataclass(slots=True)
class _EmbeddedState:
    store: InMemoryClaimQueryStore
    registry: MutationExecutionRegistry
    revisions: dict[str, int]
    journal_data: dict[str, Any]


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
    _revisions: dict[str, int] = field(default_factory=dict, init=False)
    _journal_data: dict[str, Any] = field(default_factory=dict, init=False)
    _state_observed: bool = field(default=False, init=False, repr=False)
    resource_exists: Any = None

    def __post_init__(self) -> None:
        with locked_json_store(self._path):
            state = self._load_state()
        store = self.shared_store or state.store
        registry = self.shared_execution_registry or state.registry
        store.embedder = self.embedder
        self._inner = InMemoryGraphBackend(
            store=store,
            profile_name=_PROFILE,
            embedder=self.embedder,
            definition=self.definition,
            execution_registry=registry,
            revisions=dict(state.revisions),
            journal_data=deepcopy(state.journal_data),
            resource_exists=self.resource_exists,
        )
        self._revisions = dict(state.revisions)
        self._journal_data = deepcopy(state.journal_data)
        store.before_read = self._refresh
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
    ) -> _EmbeddedState:
        loaded = False
        try:
            with open(self._path, encoding="utf-8") as fh:
                payload = strict_json_load(fh)
            loaded = True
        except FileNotFoundError as exc:
            if self._state_observed:
                raise MutationPersistenceError(
                    "previously observed embedded graph state is missing"
                ) from exc
            # A fresh process has no external lifecycle metadata with which to
            # distinguish a new home from an externally deleted store. Store
            # durability requires that applications preserve its state files.
            payload = {}
        except (ValueError, UnicodeError) as exc:
            raise MutationPersistenceError(
                "embedded graph state is malformed JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise MutationPersistenceError("embedded graph state must be an object")
        if "schema_version" in payload:
            if (
                type(payload["schema_version"]) is not int
                or payload["schema_version"] != 1
            ):
                raise MutationPersistenceError(
                    "unsupported embedded graph schema version"
                )
            required = {
                "rows": list,
                "labels": list,
                "entity_properties": list,
                "graph_revisions": dict,
                "mutation_receipts": list,
                "graph_journal": dict,
            }
            for name, kind in required.items():
                if not isinstance(payload.get(name), kind):
                    raise MutationPersistenceError(
                        f"embedded {name} must be a {kind.__name__}"
                    )
        from potpie_context_engine.core.graph_journal import decode_journal

        try:
            journal_data = (
                decode_journal(payload["graph_journal"])
                if "graph_journal" in payload
                else {}
            )
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise MutationPersistenceError("invalid embedded graph journal") from exc
        if not isinstance(journal_data, dict):
            raise MutationPersistenceError(
                "embedded graph journal must decode to a map"
            )
        revisions = payload.get("graph_revisions", {})
        if not isinstance(revisions, dict) or any(
            not isinstance(pot_id, str)
            or not pot_id
            or type(version) is not int
            or version < 0
            for pot_id, version in revisions.items()
        ):
            raise MutationPersistenceError(
                "embedded graph revisions must map pot IDs to nonnegative integers"
            )
        receipts = execution_receipts_from_json(payload.get("mutation_receipts", []))
        for receipt in receipts:
            if receipt.result.execution is not None and (
                receipt.pot_id not in revisions
                or revisions[receipt.pot_id] < receipt.result.execution.version_after
            ):
                raise MutationPersistenceError(
                    "embedded graph revision is missing or older than its execution receipt"
                )
        state = _EmbeddedState(
            load_store(payload),
            MutationExecutionRegistry(receipts),
            dict(revisions),
            journal_data,
        )
        if loaded:
            self._state_observed = True
        return state

    def _write_state(
        self,
        store: InMemoryClaimQueryStore,
        registry: MutationExecutionRegistry,
        *,
        revisions: dict[str, int] | None = None,
        journal_data: dict[str, Any] | None = None,
    ) -> None:
        payload = {"schema_version": 1, **dump_store(store)}
        payload["mutation_receipts"] = execution_receipts_to_json(registry.completed())
        payload["graph_revisions"] = dict(
            revisions if revisions is not None else self._revisions
        )
        from potpie_context_engine.core.graph_journal import encode_journal

        payload["graph_journal"] = encode_journal(
            journal_data if journal_data is not None else self._journal_data
        )
        try:
            durable_json_replace(self._path, payload)
        finally:
            # Replacement can succeed before directory synchronization raises.
            # That complete file is authoritative despite a failed acknowledgement.
            if self._path.exists():
                self._state_observed = True

    def _apply_once(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None,
        reconciliation_config: ReconciliationConfig | None,
        expected_version: int | None = None,
    ) -> MutationResult:
        with locked_json_store(self._path):
            state = self._load_state()
            receipt_count = len(state.registry.completed())
            inner = self._transaction_backend(state)
            kwargs = dict(
                expected_pot_id=expected_pot_id,
                provenance_context=provenance_context,
                reconciliation_config=reconciliation_config,
            )
            result = (
                inner.mutation.apply(plan, **kwargs)
                if expected_version is None
                else inner.mutation.compare_and_apply(
                    plan, expected_version=expected_version, **kwargs
                )
            )
            if len(inner.execution_registry.completed()) != receipt_count:
                self._persist(inner)
            self._adopt_state(inner)
            return result

    def _current_version(self, pot_id: str) -> int:
        return self._read(lambda inner: inner.mutation.current_version(pot_id))

    def _transact(
        self,
        operation: Callable[[InMemoryGraphBackend], _T],
    ) -> _T:
        """Reload, mutate, persist, then publish one complete current snapshot."""
        with locked_json_store(self._path):
            inner = self._transaction_backend(self._load_state())
            result = operation(inner)
            self._persist(inner)
            self._adopt_state(inner)
            return result

    def _persist(self, inner: InMemoryGraphBackend) -> None:
        self._write_state(
            inner.store,
            inner.execution_registry,
            revisions=inner.revisions,
            journal_data=inner.journal_data,
        )

    def _read(self, operation: Callable[[InMemoryGraphBackend], _T]) -> _T:
        with locked_json_store(self._path):
            inner = self._transaction_backend(self._load_state())
            result = operation(inner)
            self._adopt_state(inner)
            return result

    def _refresh(self) -> None:
        self._read(lambda _inner: None)

    def _transaction_backend(self, state: _EmbeddedState) -> InMemoryGraphBackend:
        return InMemoryGraphBackend(
            store=state.store,
            profile_name=_PROFILE,
            embedder=self.embedder,
            definition=self.definition,
            execution_registry=state.registry,
            revisions=state.revisions,
            journal_data=state.journal_data,
            resource_exists=self.resource_exists,
        )

    def _adopt_state(self, inner: InMemoryGraphBackend) -> None:
        # Claim, semantic, inspection and bulk-property readers use this lock.
        with self._inner.state_lock:
            _replace_store(self._inner.store, inner.store, embedder=self.embedder)
            self._inner.execution_registry.replace_completed(
                inner.execution_registry.completed()
            )
            self._inner.revisions.clear()
            self._inner.revisions.update(inner.revisions)
            self._inner.journal_data.clear()
            self._inner.journal_data.update(deepcopy(inner.journal_data))
            self._revisions = dict(inner.revisions)
            self._journal_data = deepcopy(inner.journal_data)

    def _lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ):
        return self._read(
            lambda inner: inner.mutation.lookup_execution(
                plan, expected_pot_id=expected_pot_id, mutation_id=mutation_id
            )
        )

    # --- capability bundle (delegate to the in-memory adapters) -------------
    @property
    def profile(self) -> str:
        return _PROFILE

    @property
    def journal(self):
        from potpie_context_engine.adapters.outbound.graph.embedded_journal import (
            EmbeddedJournal,
        )

        return EmbeddedJournal(self)

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
            resource_exists=self.resource_exists,
        )

    def provision(self) -> BackendProvisionResult:
        # Stand up the local store: ensure the home dir + persist the (possibly
        # empty) store file so resolve works immediately after setup. Idempotent.
        self._transact(lambda _inner: None)
        return BackendProvisionResult(
            ok=True,
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
