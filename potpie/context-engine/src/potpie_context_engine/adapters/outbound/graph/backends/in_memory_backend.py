"""In-memory ``GraphBackend`` — the conformance reference + POC substrate.

Implements all six capability ports over one shared
:class:`InMemoryClaimQueryStore`. It is a *real* backend (not a dummy): records
written through ``mutation.apply`` become claim rows that ``claim_query`` and
the readers see, which is what lets the POC prove the resolve → record → resolve
round trip without Neo4j. It is the backend the conformance suite runs against.

Not a substitute for a persistent backend — there is no durability, no real
embeddings, and traversal is naive — but the *contract* is complete.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from potpie_context_engine.core.definition import (
    DEFAULT_GRAPH_DEFINITION,
    GraphDefinition,
)
from potpie_context_engine.core.claim_lifecycle import (
    ClaimLifecycleConflict,
    apply_claim_to_rows,
    resolve_claim_assertion,
)
from potpie_context_engine.core.claim_preparation import assertion_row
from potpie_context_engine.core.claim_codec import stored_temporal_diagnostics
from potpie_context_engine.core.claim_storage import (
    StoredClaimSource,
    preserve_claim_source,
    stored_claim_record_id,
)
from potpie_context_engine.core.record_identity import resolve_entity_identity
from potpie_context_engine.core.claim_corrections import apply_claim_closure
from potpie_context_engine.core.graph_contract import evidence_strength_for_truth
from potpie_context_engine.core.graph_entity_summary import (
    merge_entity_display_properties,
    normalize_entity_properties,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EdgeDelete,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import MutationCapabilities
from potpie_context_engine.core.graph_snapshot import (
    build_snapshot_payload,
    snapshot_record_id,
)
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter, ClaimRow
from potpie_context_engine.core.ports.graph.analytics import RepairReport
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.inspection import (
    GraphEdge,
    GraphNode,
    GraphSlice,
)
from potpie_context_engine.core.ports.graph.mutation import (
    BackendReadiness,
    MutationExecutionLookup,
)
from potpie_context_engine.core.ports.graph.snapshot import SnapshotManifest
from potpie_context_engine.core.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)
from potpie_context_engine.core.reconciliation_config import ReconciliationConfig

from potpie_context_engine.adapters.outbound.graph._mutation_execution import (
    MutationExecutionRegistry,
)
from potpie_context_engine.adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
    card_for_row,
    coherent_store_read,
)
from potpie_context_engine.domain.ports.embedder import EmbedderPort
from potpie_context_engine.domain.ports.provisioning import BackendProvisionResult

_PROFILE = "in_memory"


@dataclass(slots=True)
class _Mutation:
    store: InMemoryClaimQueryStore
    on_change: Any = None
    profile: str = _PROFILE
    embedder: EmbedderPort | None = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry
    )
    state_lock: threading.RLock = field(default_factory=threading.RLock)
    revisions: dict[str, int] = field(default_factory=dict)
    journal_data: dict[str, Any] = field(default_factory=dict)
    resource_exists: Any = None

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        from potpie_context_engine.adapters.outbound.graph.memory_transaction import (
            apply_transaction,
        )

        return apply_transaction(
            self,
            plan,
            expected_pot_id=expected_pot_id,
            provenance=provenance_context,
            reconciliation_config=reconciliation_config,
        )

    @property
    def capabilities(self) -> MutationCapabilities:
        return MutationCapabilities(
            atomic_writes=True,
            revision_check=True,
            execution_lookup=True,
            durable_receipts=False,
        )

    def current_version(self, pot_id: str) -> int:
        with self.state_lock:
            return self.revisions.get(pot_id, 0)

    def compare_and_apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        expected_version: int,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        from potpie_context_engine.adapters.outbound.graph.memory_transaction import (
            apply_transaction,
        )

        return apply_transaction(
            self,
            plan,
            expected_pot_id=expected_pot_id,
            provenance=provenance_context,
            reconciliation_config=reconciliation_config,
            expected_version=expected_version,
        )

    async def current_version_async(self, pot_id: str) -> int:
        return self.current_version(pot_id)

    async def compare_and_apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        expected_version: int,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        return self.compare_and_apply(
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
        return self.lookup_execution(
            plan,
            expected_pot_id=expected_pot_id,
            mutation_id=mutation_id,
        )

    def lookup_execution(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
    ) -> MutationExecutionLookup:
        with self.state_lock:
            return self.execution_registry.lookup(
                plan,
                expected_pot_id=expected_pot_id,
                mutation_id=mutation_id,
            )

    def _apply_uncached(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        mutation_id: str,
        provenance: ProvenanceContext | None = None,
        applied_at: datetime | None = None,
    ) -> MutationResult:
        from potpie_context_engine.adapters.outbound.graph.memory_admin import (
            apply_reset,
            apply_snapshot_import,
        )

        if plan.reset_pot:
            return apply_reset(self, pot_id=expected_pot_id, mutation_id=mutation_id)
        if plan.snapshot_import is not None:
            return apply_snapshot_import(
                self,
                plan.snapshot_import,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
            )
        applied_at = applied_at or datetime.now(timezone.utc)
        anchor = (
            provenance.prepared.prepared_at
            if provenance and provenance.prepared
            else applied_at
        )
        summary = MutationSummary()
        for ent in plan.entity_upserts:
            endpoint = (expected_pot_id, ent.entity_key)
            exists = (
                endpoint in self.store.entity_label_index
                or endpoint in self.store.entity_property_index
            )
            properties = resolve_entity_identity(
                self.store.entity_property_index.get(endpoint, {}) if exists else None,
                ent.properties,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                entity_key=ent.entity_key,
            )
            if ent.preserve_labels:
                if (
                    endpoint not in self.store.entity_label_index
                    and endpoint not in self.store.entity_property_index
                ):
                    raise ClaimLifecycleConflict("entity projection target is missing")
            else:
                self.store.set_entity_label(
                    pot_id=expected_pot_id, entity_key=ent.entity_key, labels=ent.labels
                )
            self.store.set_entity_properties(
                pot_id=expected_pot_id,
                entity_key=ent.entity_key,
                properties=(
                    properties
                    if ent.preserve_labels
                    else merge_entity_display_properties(
                        properties,
                        existing=self.store.entity_property_index.get(
                            (expected_pot_id, ent.entity_key), {}
                        ),
                        entity_key=ent.entity_key,
                    )
                ),
            )
            summary.entity_upserts_applied += 1
        summary.edge_deletes_applied = self._apply_closures(
            plan.edge_deletes, pot_id=expected_pot_id
        )
        summary.invalidations_applied = self._apply_closures(
            plan.invalidations, pot_id=expected_pot_id
        )
        for edge in plan.edge_upserts:
            for key in (edge.from_entity_key, edge.to_entity_key):
                endpoint = (expected_pot_id, key)
                if (
                    endpoint not in self.store.entity_label_index
                    and endpoint not in self.store.entity_property_index
                ) or self.store.entity_property_index.get(endpoint, {}).get(
                    "retired", False
                ):
                    raise ClaimLifecycleConflict("claim endpoint is missing or retired")
            row = self._build_claim_row(
                edge,
                pot_id=expected_pot_id,
                mutation_id=mutation_id,
                anchor=anchor,
            )
            self._upsert_claim_row(row, edge=edge, anchor=anchor)
            summary.edge_upserts_applied += 1
        return MutationResult(
            ok=True, mutation_id=mutation_id, mutation_summary=summary
        )

    def _build_claim_row(
        self,
        edge: EdgeUpsert,
        *,
        pot_id: str,
        mutation_id: str,
        anchor: datetime,
    ) -> ClaimRow:
        assert edge.resolution is not None
        props = dict(edge.properties)
        props["mutation_id"] = mutation_id
        row = assertion_row(
            replace(edge, properties=props),
            pot_id=pot_id,
            record_id=edge.resolution.record_id,
            anchor=anchor,
        )
        # Embed the retrieval card on write so reads use a real vector.
        if self.embedder is not None and row.fact_embedding is None:
            try:
                row = replace(
                    row, fact_embedding=self.embedder.embed(card_for_row(row))
                )
            except Exception:  # noqa: BLE001 - graph writes should survive embedder failure.
                pass
        return row

    def _upsert_claim_row(
        self, row: ClaimRow, *, edge: EdgeUpsert, anchor: datetime
    ) -> None:
        row, resolution = resolve_claim_assertion(
            row,
            self.store.rows,
            singleton=row.predicate in self.definition.singleton_predicates,
            anchor=anchor,
            resolution=edge.resolution,
            metadata_record_id=edge.metadata_review_record_id,
        )
        if any(
            pot_id == row.pot_id
            and row.record_id == snapshot_record_id({"properties": props})
            for (pot_id, _), props in self.store.entity_property_index.items()
        ):
            raise ClaimLifecycleConflict(
                "claim incarnation collides with an entity identity"
            )
        self.store.rows[:] = apply_claim_to_rows(self.store.rows, row, resolution)

    def _apply_closures(
        self, operations: Iterable[EdgeDelete | InvalidationOp], *, pot_id: str
    ) -> int:
        count = 0
        for operation in operations:
            assert operation.closures is not None
            for closure in operation.closures:
                matches = [
                    index
                    for index, row in enumerate(self.store.rows)
                    if row.pot_id == pot_id
                    and row.record_id == closure.target.record_id
                ]
                if len(matches) != 1:
                    raise ClaimLifecycleConflict(
                        "frozen correction incarnation missing or ambiguous"
                    )
                index = matches[0]
                previous = self.store.rows[index]
                updated = apply_claim_closure(previous, closure)
                self.store.rows[index] = updated
                count += int(updated != previous)
        return count

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
        reconciliation_config: ReconciliationConfig | None = None,
    ) -> MutationResult:
        # In-memory mutations are pure-sync CPU work (no I/O to await); the
        # async door just delegates so async callers get a uniform surface.
        return self.apply(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
            reconciliation_config=reconciliation_config,
        )

    def invalidate(
        self, *, pot_id: str, claim_keys: Sequence[str], reason: str | None = None
    ) -> int:
        from potpie_context_engine.core.graph_mutations import InvalidationOp

        if not claim_keys:
            return 0
        result = self.apply(
            MutationBatch(
                invalidations=[
                    InvalidationOp(
                        target_entity_key=None,
                        target_edge=None,
                        reason=reason or "direct invalidation",
                        target_claim_keys=tuple(claim_keys),
                    )
                ]
            ),
            expected_pot_id=pot_id,
        )
        return result.mutation_summary.invalidations_applied

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
        return self.reset_pot(pot_id, mutation_id=mutation_id)

    def readiness(self, pot_id: str) -> BackendReadiness:
        return BackendReadiness(
            profile=self.profile,
            ready=True,
            capability_ready={
                "mutation": True,
                "claim_query": True,
                "semantic": True,
                "inspection": True,
                "analytics": True,
                "snapshot": True,
            },
        )


@dataclass(slots=True)
class _Semantic:
    store: InMemoryClaimQueryStore

    @coherent_store_read
    def search(
        self,
        *,
        pot_id: str,
        query: str,
        k: int = 10,
        filter_: ClaimQueryFilter | None = None,
    ) -> list[ClaimRow]:
        base = filter_ or ClaimQueryFilter(pot_id=pot_id)
        # Re-derive with the semantic query + k bound applied.
        scored = self.store.find_claims(
            ClaimQueryFilter(
                pot_id=pot_id,
                predicate_in=base.predicate_in,
                subject_key_in=base.subject_key_in,
                object_key_in=base.object_key_in,
                subject_label=base.subject_label,
                object_label=base.object_label,
                include_invalidated=base.include_invalidated,
                source_system_in=base.source_system_in,
                subgraph_in=base.subgraph_in,
                subgraph_not_in=base.subgraph_not_in,
                fact_query=query,
                limit=k,
            )
        )
        return scored


@dataclass(slots=True)
class _Inspection:
    store: InMemoryClaimQueryStore

    @coherent_store_read
    def neighborhood(
        self,
        *,
        pot_id: str,
        entity_key: str,
        depth: int = 1,
        direction: str = "both",
        predicates: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> GraphSlice:
        if self.store.entity_property_index.get((pot_id, entity_key), {}).get(
            "retired", False
        ):
            return GraphSlice(pot_id=pot_id)
        seen_nodes: dict[str, GraphNode] = {}
        identity = (pot_id, entity_key)
        if (
            identity in self.store.entity_label_index
            or identity in self.store.entity_property_index
            or any(
                row.pot_id == pot_id and entity_key in (row.subject_key, row.object_key)
                for row in self.store.rows
            )
        ):
            seen_nodes[entity_key] = self._node(pot_id, entity_key)
        edges: list[GraphEdge] = []
        seen_edges: set[tuple[str, ...]] = set()
        frontier = {entity_key}
        max_edges = max(0, int(limit)) if limit is not None else None
        predicate_set = {p.upper() for p in predicates if p}
        walk_out = direction in ("out", "both")
        walk_in = direction in ("in", "both")
        truncated = False
        visited_frontier: set[str] = set()
        for _ in range(max(1, depth)):
            next_frontier: set[str] = set()
            current = frontier - visited_frontier
            if not current:
                break
            visited_frontier.update(current)
            for row in self.store.rows:
                if row.retired or row.pot_id != pot_id:
                    continue
                if row.invalid_at is not None:
                    # Invalidated claims are history, not current structure; the
                    # FalkorDB inspection path excludes them too.
                    continue
                if predicate_set and row.predicate.upper() not in predicate_set:
                    continue
                follows_out = walk_out and row.subject_key in current
                follows_in = walk_in and row.object_key in current
                if not (follows_out or follows_in):
                    continue
                edge_key = (
                    row.subject_key,
                    row.predicate,
                    row.object_key,
                    row.record_id or str(row.claim_key or ""),
                )
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(_edge(row))
                    if max_edges is not None and len(edges) >= max_edges:
                        truncated = True
                for key in (row.subject_key, row.object_key):
                    if key not in seen_nodes:
                        seen_nodes[key] = self._node(pot_id, key)
                if follows_out:
                    next_frontier.add(row.object_key)
                if follows_in:
                    next_frontier.add(row.subject_key)
                if truncated:
                    break
            if truncated:
                break
            frontier = next_frontier - visited_frontier
            if not frontier:
                break
        return GraphSlice(
            pot_id=pot_id,
            nodes=tuple(seen_nodes.values()),
            edges=tuple(edges),
            truncated=truncated,
        )

    @coherent_store_read
    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        # Naive BFS over undirected claim edges.
        adjacency: dict[str, list[ClaimRow]] = {}
        for row in self.store.rows:
            if row.retired or row.invalid_at is not None or row.pot_id != pot_id:
                continue
            adjacency.setdefault(row.subject_key, []).append(row)
            adjacency.setdefault(row.object_key, []).append(row)
        queue: list[tuple[str, list[ClaimRow]]] = [(from_key, [])]
        visited = {from_key}
        while queue:
            node, trail = queue.pop(0)
            if node == to_key:
                nodes = {from_key, to_key}
                for r in trail:
                    nodes.update({r.subject_key, r.object_key})
                return GraphSlice(
                    pot_id=pot_id,
                    nodes=tuple(self._node(pot_id, k) for k in nodes),
                    edges=tuple(_edge(r) for r in trail),
                )
            if len(trail) >= max_depth:
                continue
            for row in adjacency.get(node, []):
                nxt = row.object_key if row.subject_key == node else row.subject_key
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, trail + [row]))
        return GraphSlice(pot_id=pot_id)

    @coherent_store_read
    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        return self.store.entity_labels(pot_id=pot_id, entity_keys=entity_keys)

    @coherent_store_read
    def slice(self, *, pot_id: str, filter_: ClaimQueryFilter) -> GraphSlice:
        rows = self.store.find_claims(filter_)
        node_keys = {k for r in rows for k in (r.subject_key, r.object_key)}
        return GraphSlice(
            pot_id=pot_id,
            nodes=tuple(self._node(pot_id, k) for k in node_keys),
            edges=tuple(_edge(r) for r in rows),
        )

    def _node(self, pot_id: str, key: str) -> GraphNode:
        labels = self.store.entity_label_index.get((pot_id, key), ())
        props = normalize_entity_properties(
            self.store.entity_properties(pot_id=pot_id, entity_key=key),
            entity_key=key,
        )
        return GraphNode(key=key, labels=labels, properties=props)


@dataclass(slots=True)
class _Analytics:
    store: InMemoryClaimQueryStore
    mutation: _Mutation
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION

    def _rows(self, pot_id: str) -> list[ClaimRow]:
        return [r for r in self.store.rows if r.pot_id == pot_id and not r.retired]

    @coherent_store_read
    def counts(self, pot_id: str) -> Mapping[str, int]:
        rows = self._rows(pot_id)
        entities = {k for r in rows for k in (r.subject_key, r.object_key)}
        predicates = {r.predicate for r in rows}
        return {
            "claims": len(rows),
            "entities": len(entities),
            "predicates": len(predicates),
            "invalidated": sum(1 for r in rows if r.invalid_at is not None),
        }

    @coherent_store_read
    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        stamps = [r.valid_at for r in self._rows(pot_id) if r.valid_at is not None]
        return {
            "oldest": min(stamps).isoformat() if stamps else None,
            "newest": max(stamps).isoformat() if stamps else None,
            "stamped_claims": len(stamps),
        }

    @coherent_store_read
    def quality(self, pot_id: str) -> Mapping[str, Any]:
        rows = self._rows(pot_id)
        return {
            "status": "ok" if rows else "empty",
            "open_conflicts": 0,
            "claim_count": len(rows),
        }

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        from potpie_context_engine.adapters.outbound.graph.memory_admin import (
            repair_entities,
        )

        return repair_entities(self, pot_id, targets)


@dataclass(slots=True)
class _Snapshot:
    store: InMemoryClaimQueryStore
    mutation: _Mutation

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

    @coherent_store_read
    def export_data(self, *, pot_id: str) -> dict[str, Any]:
        rows = [
            r
            for r in self.store.rows
            if r.pot_id == pot_id
            and not r.retired
            and not any(
                self.store.entity_property_index.get((pot_id, key), {}).get(
                    "retired", False
                )
                for key in (r.subject_key, r.object_key)
            )
        ]
        keys = {key for row in rows for key in (row.subject_key, row.object_key)}
        keys.update(key for pid, key in self.store.entity_label_index if pid == pot_id)
        keys.update(
            key for pid, key in self.store.entity_property_index if pid == pot_id
        )
        keys = {
            key
            for key in keys
            if not self.store.entity_property_index.get((pot_id, key), {}).get(
                "retired", False
            )
        }
        entities = [
            {
                "key": key,
                "labels": list(self.store.entity_label_index.get((pot_id, key), ())),
                "properties": self.store.entity_properties(
                    pot_id=pot_id, entity_key=key
                ),
            }
            for key in keys
        ]
        claims = [
            {
                **{
                    k: v
                    for k, v in _row_to_dict(row).items()
                    if k != "retired" and (k != "record_id" or v)
                },
                "evidence_strength": row.evidence_strength,
            }
            for row in rows
        ]
        return build_snapshot_payload(pot_id=pot_id, entities=entities, claims=claims)

    def import_(
        self, *, pot_id: str, source: str, mutation_id: str | None = None
    ) -> SnapshotManifest:
        with open(source, encoding="utf-8") as fh:
            payload = json.load(fh)
        return replace(
            self.import_data(pot_id=pot_id, payload=payload, mutation_id=mutation_id),
            location=source,
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
            self.mutation, pot_id=pot_id, payload=payload, mutation_id=mutation_id
        )


@dataclass(slots=True)
class InMemoryGraphBackend:
    """A complete ``GraphBackend`` over one in-memory claim store.

    ``on_change`` is an optional postcommit observer. Durable backends persist
    their staged state before publication rather than using this observer.
    """

    store: InMemoryClaimQueryStore = field(default_factory=InMemoryClaimQueryStore)
    profile_name: str = _PROFILE
    on_change: Any = None
    embedder: EmbedderPort | None = None
    definition: GraphDefinition = DEFAULT_GRAPH_DEFINITION
    execution_registry: MutationExecutionRegistry = field(
        default_factory=MutationExecutionRegistry,
        repr=False,
    )
    revisions: dict[str, int] = field(default_factory=dict)
    journal_data: dict[str, Any] = field(default_factory=dict)
    resource_exists: Any = None
    state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _mutation: _Mutation = field(init=False)
    _semantic: _Semantic = field(init=False)
    _inspection: _Inspection = field(init=False)
    _analytics: _Analytics = field(init=False)
    _snapshot: _Snapshot = field(init=False)

    def __post_init__(self) -> None:
        # Store readers and all definition-bound writers share one publication lock.
        self.state_lock = self.store.state_lock
        # The embedder powers vector search on read and embed-on-write; share it
        # with the store so ``find_claims`` and ``mutation.apply`` agree on mode.
        if self.embedder is not None and self.store.embedder is None:
            self.store.embedder = self.embedder
        self._mutation = _Mutation(
            self.store,
            on_change=self.on_change,
            profile=self.profile_name,
            embedder=self.store.embedder,
            definition=self.definition,
            execution_registry=self.execution_registry,
            revisions=self.revisions,
            state_lock=self.state_lock,
            journal_data=self.journal_data,
            resource_exists=self.resource_exists,
        )
        self._semantic = _Semantic(self.store)
        self._inspection = _Inspection(self.store)
        self._analytics = _Analytics(
            self.store,
            mutation=self._mutation,
            definition=self.definition,
        )
        self._snapshot = _Snapshot(self.store, self._mutation)

    @property
    def match_mode(self) -> str:
        return self.store.match_mode

    @property
    def profile(self) -> str:
        return self.profile_name

    @property
    def mutation(self) -> _Mutation:
        return self._mutation

    @property
    def journal(self):
        from potpie_context_engine.adapters.outbound.graph.memory_journal import (
            MemoryJournal,
        )

        return MemoryJournal(self._mutation)

    @property
    def claim_query(self) -> InMemoryClaimQueryStore:
        return self.store

    @property
    def semantic(self) -> _Semantic:
        return self._semantic

    @property
    def inspection(self) -> _Inspection:
        return self._inspection

    @property
    def analytics(self) -> _Analytics:
        return self._analytics

    @property
    def snapshot(self) -> _Snapshot:
        return self._snapshot

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=self.profile_name,
            mutation=True,
            claim_query=True,
            semantic=True,
            inspection=True,
            analytics=True,
            snapshot=True,
        )

    def provision(self) -> BackendProvisionResult:
        # Ephemeral store: nothing durable to stand up.
        return BackendProvisionResult(
            ok=True,
            detail=f"'{self.profile_name}' backend ready (ephemeral, no store to provision)",
            metadata={"profile": self.profile_name},
        )

    def bind_definition(self, definition: GraphDefinition) -> InMemoryGraphBackend:
        """Return a definition-bound facade without mutating this backend."""
        return InMemoryGraphBackend(
            store=self.store,
            profile_name=self.profile_name,
            on_change=self.on_change,
            embedder=self.embedder,
            definition=definition,
            execution_registry=self.execution_registry,
            revisions=self.revisions,
            state_lock=self.state_lock,
            journal_data=self.journal_data,
            resource_exists=self.resource_exists,
        )


def _edge(row: ClaimRow) -> GraphEdge:
    properties = {
        **dict(row.properties),
        "claim_key": row.claim_key,
        "record_id": row.record_id,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "confidence": row.confidence,
        "description": row.description,
        "environment": row.environment,
        "fact": row.fact,
        "source_system": row.source_system,
        "source_ref": row.source_ref,
        "source_refs": list(row.source_refs),
        "valid_at": _dt_iso(row.valid_at),
        "valid_until": _dt_iso(row.valid_until),
        "observed_at": _dt_iso(row.observed_at),
        "mutation_id": row.mutation_id,
    }
    return GraphEdge(
        predicate=row.predicate,
        from_key=row.subject_key,
        to_key=row.object_key,
        properties={
            key: value for key, value in properties.items() if value is not None
        },
    )


def _coerce_dt(value: Any) -> datetime | None:
    """Coerce an ISO string or datetime into a tz-aware datetime (or None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _evidence_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _dt_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _row_to_dict(row: ClaimRow) -> dict[str, Any]:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "valid_at": _dt_iso(row.valid_at),
        "invalid_at": _dt_iso(row.invalid_at),
        "source_system": row.source_system,
        "source_ref": row.source_ref,
        "fact": row.fact,
        "properties": dict(row.properties),
        "fact_embedding": list(row.fact_embedding) if row.fact_embedding else None,
        # V1.5 first-class metadata.
        "claim_key": row.claim_key,
        "subgraph": row.subgraph,
        "truth": row.truth,
        "confidence": row.confidence,
        "description": row.description,
        "environment": row.environment,
        "observed_at": _dt_iso(row.observed_at),
        "valid_until": _dt_iso(row.valid_until),
        "mutation_id": row.mutation_id,
        "source_refs": list(row.source_refs),
        "evidence": [dict(item) for item in row.evidence],
        "graph_contract_version": row.graph_contract_version,
        "ontology_version": row.ontology_version,
        "record_id": row.record_id,
        "retired": row.retired,
    }


def _row_from_dict(pot_id: str, raw: Mapping[str, Any]) -> ClaimRow:
    embedding = raw.get("fact_embedding")
    source_refs = raw.get("source_refs") or ()
    truth = _coerce_str(raw.get("truth"))
    temporal_diagnostics = stored_temporal_diagnostics(
        raw.get("properties", {}), canonical_fields=raw
    )
    row = ClaimRow(
        pot_id=pot_id,
        predicate=raw["predicate"],
        subject_key=raw["subject_key"],
        object_key=raw["object_key"],
        valid_at=_coerce_dt(raw.get("valid_at")),
        invalid_at=_coerce_dt(raw.get("invalid_at")),
        evidence_strength=_coerce_str(raw.get("evidence_strength"))
        or evidence_strength_for_truth(truth),
        source_system=_coerce_str(raw.get("source_system")),
        source_ref=_coerce_str(raw.get("source_ref")),
        fact=_coerce_str(raw.get("fact")),
        properties=dict(raw.get("properties", {})),
        fact_embedding=tuple(embedding) if embedding else None,
        claim_key=_coerce_str(raw.get("claim_key")),
        subgraph=_coerce_str(raw.get("subgraph")),
        truth=truth,
        confidence=_coerce_float(raw.get("confidence")),
        description=_coerce_str(raw.get("description")),
        environment=_coerce_str(raw.get("environment")),
        observed_at=_coerce_dt(raw.get("observed_at")),
        valid_until=_coerce_dt(raw.get("valid_until")),
        mutation_id=_coerce_str(raw.get("mutation_id")),
        source_refs=tuple(source_refs),
        evidence=_evidence_tuple(raw.get("evidence")),
        graph_contract_version=_coerce_str(raw.get("graph_contract_version")),
        ontology_version=_coerce_str(raw.get("ontology_version")),
        record_id=stored_claim_record_id(raw, raw.get("properties", {})),
        retired=bool(raw.get("retired", False)),
        temporal_diagnostics=temporal_diagnostics,
    )
    hydrated = _row_to_dict(row)
    hydrated.pop("properties")
    return replace(
        row,
        stored_source=StoredClaimSource(
            fields=deepcopy(
                {
                    key: value
                    for key, value in raw.items()
                    if key
                    not in {
                        "pot_id",
                        "properties",
                    }
                }
            ),
            hydrated_fields=hydrated,
        ),
    )


def _stored_row_to_dict(row: ClaimRow) -> dict[str, Any]:
    current = _row_to_dict(row)
    properties = current.pop("properties")
    stored = preserve_claim_source(current, row.stored_source)
    stored["properties"] = properties
    return stored


def dump_store(store: InMemoryClaimQueryStore) -> dict[str, Any]:
    """Serialize a whole (multi-pot) claim store for persistence."""
    return {
        "format_version": "2",
        "rows": [{"pot_id": r.pot_id, **_stored_row_to_dict(r)} for r in store.rows],
        "labels": [
            [pot_id, key, list(labels)]
            for (pot_id, key), labels in store.entity_label_index.items()
        ],
        "entity_properties": [
            [pot_id, key, props]
            for (pot_id, key), props in store.entity_property_index.items()
        ],
    }


def load_store(data: dict[str, Any]) -> InMemoryClaimQueryStore:
    """Rebuild a claim store from :func:`dump_store` output."""
    store = InMemoryClaimQueryStore()
    for raw in data.get("rows", []):
        store.add(_row_from_dict(raw["pot_id"], raw))
    for pot_id, key, labels in data.get("labels", []):
        store.set_entity_label(pot_id=pot_id, entity_key=key, labels=labels)
    for pot_id, key, props in data.get("entity_properties", []):
        store.set_entity_properties(pot_id=pot_id, entity_key=key, properties=props)
    return store


__all__ = ["InMemoryGraphBackend", "dump_store", "load_store"]
