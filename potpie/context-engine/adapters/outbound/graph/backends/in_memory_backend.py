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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from adapters.outbound.graph.in_memory_reader import InMemoryClaimQueryStore
from domain.graph_mutations import ProvenanceContext
from domain.lifecycle import DONE, SetupPlan, StepResult
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.graph.analytics import RepairReport
from domain.ports.graph.backend import BackendCapabilities
from domain.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice
from domain.ports.graph.mutation import BackendReadiness
from domain.ports.graph.snapshot import SnapshotManifest
from domain.reconciliation import (
    MutationSummary,
    ReconciliationPlan,
    ReconciliationResult,
)

_PROFILE = "in_memory"


@dataclass(slots=True)
class _Mutation:
    store: InMemoryClaimQueryStore
    on_change: Any = None
    profile: str = _PROFILE

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def apply(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        summary = MutationSummary()
        for ent in plan.entity_upserts:
            self.store.set_entity_label(
                pot_id=expected_pot_id, entity_key=ent.entity_key, labels=ent.labels
            )
            summary.entity_upserts_applied += 1
        for edge in plan.edge_upserts:
            props = dict(edge.properties)
            self.store.add(
                ClaimRow(
                    pot_id=expected_pot_id,
                    predicate=edge.edge_type,
                    subject_key=edge.from_entity_key,
                    object_key=edge.to_entity_key,
                    valid_at=props.get("valid_at"),
                    evidence_strength=props.get("evidence_strength", "stated"),
                    source_system=props.get("source_system"),
                    source_ref=props.get("source_ref"),
                    fact=props.get("fact") or plan.summary,
                    properties=props,
                )
            )
            summary.edge_upserts_applied += 1
        self._notify()
        return ReconciliationResult(
            ok=True, mutation_id=uuid.uuid4().hex, mutation_summary=summary
        )

    async def apply_async(
        self,
        plan: ReconciliationPlan,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> ReconciliationResult:
        # In-memory mutations are pure-sync CPU work (no I/O to await); the
        # async door just delegates so async callers get a uniform surface.
        return self.apply(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def invalidate(
        self, *, pot_id: str, claim_keys: Sequence[str], reason: str | None = None
    ) -> int:
        keys = set(claim_keys)
        invalidated = 0
        now = datetime.now(timezone.utc)
        for i, row in enumerate(self.store.rows):
            if row.pot_id != pot_id or row.invalid_at is not None:
                continue
            if row.subject_key in keys or row.object_key in keys:
                self.store.rows[i] = _with_invalid_at(row, now)
                invalidated += 1
        self._notify()
        return invalidated

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        before = len(self.store.rows)
        self.store.rows = [r for r in self.store.rows if r.pot_id != pot_id]
        for key in [k for k in self.store.entity_label_index if k[0] == pot_id]:
            self.store.entity_label_index.pop(key, None)
        self._notify()
        return {"removed_claims": before - len(self.store.rows)}

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
                fact_query=query,
                limit=k,
            )
        )
        return scored


@dataclass(slots=True)
class _Inspection:
    store: InMemoryClaimQueryStore

    def neighborhood(
        self, *, pot_id: str, entity_key: str, depth: int = 1
    ) -> GraphSlice:
        seen_nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        frontier = {entity_key}
        for _ in range(max(1, depth)):
            next_frontier: set[str] = set()
            for row in self.store.rows:
                if row.pot_id != pot_id:
                    continue
                if row.subject_key in frontier or row.object_key in frontier:
                    edges.append(_edge(row))
                    for key in (row.subject_key, row.object_key):
                        if key not in seen_nodes:
                            seen_nodes[key] = self._node(pot_id, key)
                            next_frontier.add(key)
            frontier = next_frontier - set(seen_nodes)
            if not frontier:
                break
        return GraphSlice(
            pot_id=pot_id, nodes=tuple(seen_nodes.values()), edges=tuple(edges)
        )

    def path(
        self, *, pot_id: str, from_key: str, to_key: str, max_depth: int = 4
    ) -> GraphSlice:
        # Naive BFS over undirected claim edges.
        adjacency: dict[str, list[ClaimRow]] = {}
        for row in self.store.rows:
            if row.pot_id != pot_id:
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

    def labels(
        self, *, pot_id: str, entity_keys: Iterable[str]
    ) -> Mapping[str, tuple[str, ...]]:
        return self.store.entity_labels(pot_id=pot_id, entity_keys=entity_keys)

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
        return GraphNode(key=key, labels=labels)


@dataclass(slots=True)
class _Analytics:
    store: InMemoryClaimQueryStore

    def _rows(self, pot_id: str) -> list[ClaimRow]:
        return [r for r in self.store.rows if r.pot_id == pot_id]

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

    def freshness(self, pot_id: str) -> Mapping[str, Any]:
        stamps = [r.valid_at for r in self._rows(pot_id) if r.valid_at is not None]
        return {
            "oldest": min(stamps).isoformat() if stamps else None,
            "newest": max(stamps).isoformat() if stamps else None,
            "stamped_claims": len(stamps),
        }

    def quality(self, pot_id: str) -> Mapping[str, Any]:
        rows = self._rows(pot_id)
        return {
            "status": "ok" if rows else "empty",
            "open_conflicts": 0,
            "claim_count": len(rows),
        }

    def repair(self, pot_id: str, *, targets: Sequence[str] = ()) -> RepairReport:
        # Projections are derived on read here, so repair is a no-op report.
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired={},
            detail="in_memory projections are computed on read; nothing to rebuild",
        )


@dataclass(slots=True)
class _Snapshot:
    store: InMemoryClaimQueryStore

    def export(self, *, pot_id: str, destination: str) -> SnapshotManifest:
        rows = [r for r in self.store.rows if r.pot_id == pot_id]
        payload = {
            "format_version": "1",
            "pot_id": pot_id,
            "claims": [_row_to_dict(r) for r in rows],
            "labels": {
                f"{k[1]}": list(v)
                for k, v in self.store.entity_label_index.items()
                if k[0] == pot_id
            },
        }
        with open(destination, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        entities = {k for r in rows for k in (r.subject_key, r.object_key)}
        return SnapshotManifest(
            pot_id=pot_id,
            location=destination,
            entity_count=len(entities),
            claim_count=len(rows),
        )

    def import_(self, *, pot_id: str, source: str) -> SnapshotManifest:
        with open(source, encoding="utf-8") as fh:
            payload = json.load(fh)
        claims = payload.get("claims", [])
        for raw in claims:
            self.store.add(_row_from_dict(pot_id, raw))
        for key, labels in payload.get("labels", {}).items():
            self.store.set_entity_label(pot_id=pot_id, entity_key=key, labels=labels)
        entities = {
            k for raw in claims for k in (raw["subject_key"], raw["object_key"])
        }
        return SnapshotManifest(
            pot_id=pot_id,
            location=source,
            entity_count=len(entities),
            claim_count=len(claims),
        )


@dataclass(slots=True)
class InMemoryGraphBackend:
    """A complete ``GraphBackend`` over one in-memory claim store.

    ``on_change`` is an optional hook invoked after every mutation — the
    embedded backend uses it to persist the store to disk, reusing all the
    capability adapters here unchanged.
    """

    store: InMemoryClaimQueryStore = field(default_factory=InMemoryClaimQueryStore)
    profile_name: str = _PROFILE
    on_change: Any = None
    _mutation: _Mutation = field(init=False)
    _semantic: _Semantic = field(init=False)
    _inspection: _Inspection = field(init=False)
    _analytics: _Analytics = field(init=False)
    _snapshot: _Snapshot = field(init=False)

    def __post_init__(self) -> None:
        self._mutation = _Mutation(
            self.store, on_change=self.on_change, profile=self.profile_name
        )
        self._semantic = _Semantic(self.store)
        self._inspection = _Inspection(self.store)
        self._analytics = _Analytics(self.store)
        self._snapshot = _Snapshot(self.store)

    @property
    def profile(self) -> str:
        return self.profile_name

    @property
    def mutation(self) -> _Mutation:
        return self._mutation

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

    def provision(self, plan: SetupPlan) -> StepResult:
        # Ephemeral store: nothing durable to stand up.
        return StepResult(
            step="backend.provision",
            state=DONE,
            detail=f"'{self.profile_name}' backend ready (ephemeral, no store to provision)",
            metadata={"profile": self.profile_name},
        )


def _edge(row: ClaimRow) -> GraphEdge:
    return GraphEdge(
        predicate=row.predicate,
        from_key=row.subject_key,
        to_key=row.object_key,
        properties=dict(row.properties),
    )


def _with_invalid_at(row: ClaimRow, when: datetime) -> ClaimRow:
    return ClaimRow(
        pot_id=row.pot_id,
        predicate=row.predicate,
        subject_key=row.subject_key,
        object_key=row.object_key,
        valid_at=row.valid_at,
        invalid_at=when,
        evidence_strength=row.evidence_strength,
        source_system=row.source_system,
        source_ref=row.source_ref,
        fact=row.fact,
        properties=row.properties,
        fact_embedding=row.fact_embedding,
    )


def _row_to_dict(row: ClaimRow) -> dict[str, Any]:
    return {
        "predicate": row.predicate,
        "subject_key": row.subject_key,
        "object_key": row.object_key,
        "valid_at": row.valid_at.isoformat() if row.valid_at else None,
        "evidence_strength": row.evidence_strength,
        "source_system": row.source_system,
        "source_ref": row.source_ref,
        "fact": row.fact,
        "properties": dict(row.properties),
    }


def _row_from_dict(pot_id: str, raw: Mapping[str, Any]) -> ClaimRow:
    valid_at = raw.get("valid_at")
    return ClaimRow(
        pot_id=pot_id,
        predicate=raw["predicate"],
        subject_key=raw["subject_key"],
        object_key=raw["object_key"],
        valid_at=datetime.fromisoformat(valid_at) if valid_at else None,
        evidence_strength=raw.get("evidence_strength", "stated"),
        source_system=raw.get("source_system"),
        source_ref=raw.get("source_ref"),
        fact=raw.get("fact"),
        properties=dict(raw.get("properties", {})),
    )


def dump_store(store: InMemoryClaimQueryStore) -> dict[str, Any]:
    """Serialize a whole (multi-pot) claim store for persistence."""
    return {
        "format_version": "1",
        "rows": [{"pot_id": r.pot_id, **_row_to_dict(r)} for r in store.rows],
        "labels": [
            [pot_id, key, list(labels)]
            for (pot_id, key), labels in store.entity_label_index.items()
        ],
    }


def load_store(data: dict[str, Any]) -> InMemoryClaimQueryStore:
    """Rebuild a claim store from :func:`dump_store` output."""
    store = InMemoryClaimQueryStore()
    for raw in data.get("rows", []):
        store.add(_row_from_dict(raw["pot_id"], raw))
    for pot_id, key, labels in data.get("labels", []):
        store.set_entity_label(pot_id=pot_id, entity_key=key, labels=labels)
    return store


__all__ = ["InMemoryGraphBackend", "dump_store", "load_store"]
