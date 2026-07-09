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

from adapters.outbound.graph.in_memory_reader import (
    InMemoryClaimQueryStore,
    card_for_row,
)
from adapters.outbound.graph.canonical_claim_query import CONTRACT_EDGE_KEYS
from adapters.outbound.graph.entity_summary_repair import (
    ENTITY_SUMMARY_TARGET,
    repaired_entity_properties,
    wants_entity_summary_repair,
)
from adapters.outbound.graph.semantic_index_repair import (
    SEMANTIC_INDEX_TARGET,
    wants_semantic_index_repair,
)
from application.services.reconciliation_validation import (
    validate_reconciliation_plan,
)
from domain.graph_contract import evidence_strength_for_truth
from domain.graph_entity_summary import (
    merge_entity_display_properties,
    normalize_entity_properties,
)
from domain.graph_mutations import ProvenanceContext
from domain.lifecycle import DONE, SetupPlan, StepResult
from domain.ports.claim_query import ClaimQueryFilter, ClaimRow
from domain.ports.embedder import EmbedderPort
from domain.ports.graph.analytics import RepairReport
from domain.ports.graph.backend import BackendCapabilities
from domain.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice
from domain.ports.graph.mutation import BackendReadiness
from domain.ports.graph.snapshot import SnapshotManifest
from domain.reconciliation import (
    MutationBatch,
    MutationResult,
    MutationSummary,
)

_PROFILE = "in_memory"


@dataclass(slots=True)
class _Mutation:
    store: InMemoryClaimQueryStore
    on_change: Any = None
    profile: str = _PROFILE
    embedder: EmbedderPort | None = None

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        # Validate before mutating, exactly like the Neo4j/FalkorDB writers do
        # (both route apply through apply_mutation_batch → this validator). Keeps
        # the substrate from silently accepting malformed or cross-pot batches.
        validate_reconciliation_plan(plan, expected_pot_id)
        summary = MutationSummary()
        mutation_id = uuid.uuid4().hex
        for ent in plan.entity_upserts:
            self.store.set_entity_label(
                pot_id=expected_pot_id, entity_key=ent.entity_key, labels=ent.labels
            )
            self.store.set_entity_properties(
                pot_id=expected_pot_id,
                entity_key=ent.entity_key,
                properties=merge_entity_display_properties(
                    ent.properties,
                    existing=self.store.entity_properties(
                        pot_id=expected_pot_id, entity_key=ent.entity_key
                    ),
                    entity_key=ent.entity_key,
                ),
            )
            summary.entity_upserts_applied += 1
        for edge in plan.edge_upserts:
            self._upsert_claim_row(
                self._build_claim_row(
                    edge,
                    pot_id=expected_pot_id,
                    mutation_id=mutation_id,
                    fallback_fact=plan.summary,
                )
            )
            summary.edge_upserts_applied += 1
        summary.edge_deletes_applied = self._apply_edge_deletes(
            plan, pot_id=expected_pot_id
        )
        summary.invalidations_applied = self._apply_invalidations(
            plan, pot_id=expected_pot_id
        )
        self._notify()
        return MutationResult(
            ok=True, mutation_id=mutation_id, mutation_summary=summary
        )

    def _build_claim_row(
        self,
        edge: Any,
        *,
        pot_id: str,
        mutation_id: str,
        fallback_fact: str,
    ) -> ClaimRow:
        props = dict(edge.properties)
        props.setdefault("mutation_id", mutation_id)
        source_refs = props.get("source_refs")
        truth = _coerce_str(props.get("truth"))
        row = ClaimRow(
            pot_id=pot_id,
            predicate=edge.edge_type,
            subject_key=edge.from_entity_key,
            object_key=edge.to_entity_key,
            valid_at=_coerce_dt(props.get("valid_at")),
            evidence_strength=evidence_strength_for_truth(truth),
            source_system=_coerce_str(props.get("source_system")),
            source_ref=_coerce_str(props.get("source_ref")),
            fact=_coerce_str(props.get("fact")) or fallback_fact or None,
            properties=_reader_extras(props),
            fact_embedding=_vector_tuple(props.get("fact_embedding")),
            claim_key=_coerce_str(props.get("claim_key")),
            subgraph=_coerce_str(props.get("subgraph")),
            truth=truth,
            confidence=_coerce_float(props.get("confidence")),
            description=_coerce_str(props.get("description")),
            environment=_coerce_str(props.get("environment")),
            observed_at=_coerce_dt(props.get("observed_at")),
            valid_until=_coerce_dt(props.get("valid_until")),
            mutation_id=mutation_id,
            source_refs=tuple(source_refs)
            if isinstance(source_refs, (list, tuple))
            else (),
            evidence=_evidence_tuple(props.get("evidence")),
            graph_contract_version=_coerce_str(props.get("graph_contract_version")),
            ontology_version=_coerce_str(props.get("ontology_version")),
        )
        # Embed the retrieval card on write (R1/R2) so reads use a real vector.
        if self.embedder is not None and row.fact_embedding is None:
            try:
                row = _with_embedding(row, self.embedder.embed(card_for_row(row)))
            except Exception:  # noqa: BLE001 - graph writes should survive embedder failure.
                pass
        return row

    def _upsert_claim_row(self, row: ClaimRow) -> None:
        """MERGE the claim by identity instead of blindly appending.

        The canonical Neo4j/FalkorDB writers ``MERGE`` on claim identity, so
        re-applying the same batch updates the same edge. Mirror that here: an
        existing live row with the same ``claim_key`` (or the same
        source_ref + predicate + endpoints) is replaced, not duplicated.
        """
        for i, existing in enumerate(self.store.rows):
            if existing.pot_id != row.pot_id or existing.invalid_at is not None:
                continue
            same_claim = bool(row.claim_key and existing.claim_key == row.claim_key)
            same_source_edge = bool(
                row.source_ref
                and existing.source_ref == row.source_ref
                and existing.predicate == row.predicate
                and existing.subject_key == row.subject_key
                and existing.object_key == row.object_key
            )
            if same_claim or same_source_edge:
                self.store.rows[i] = row
                return
        self.store.add(row)

    def _apply_edge_deletes(self, plan: MutationBatch, *, pot_id: str) -> int:
        """Drop claims targeted by ``plan.edge_deletes``.

        The shared application path records ``summary.edge_deletes_applied``
        after the writer deletes edges; the substrate must actually remove the
        rows so it does not keep stale claims the canonical backends would drop.
        """
        if not plan.edge_deletes:
            return 0
        targets = {
            (edge.edge_type.upper(), edge.from_entity_key, edge.to_entity_key)
            for edge in plan.edge_deletes
        }
        before = len(self.store.rows)
        self.store.rows = [
            row
            for row in self.store.rows
            if not (
                row.pot_id == pot_id
                and (row.predicate.upper(), row.subject_key, row.object_key) in targets
            )
        ]
        return before - len(self.store.rows)

    def _apply_invalidations(self, plan: MutationBatch, *, pot_id: str) -> int:
        if not plan.invalidations:
            return 0
        now = datetime.now(timezone.utc)
        # Preserve each op's own ``valid_to`` (the canonical writers stamp it when
        # present); fall back to apply-time ``now`` only when it is missing.
        edge_targets: dict[tuple[str, str, str], datetime] = {}
        entity_targets: dict[str, datetime] = {}
        for inv in plan.invalidations:
            invalid_at = _coerce_dt(inv.valid_to) or now
            if inv.target_edge:
                pred, subj, obj = inv.target_edge
                edge_targets[(pred.upper(), subj, obj)] = invalid_at
            if inv.target_entity_key:
                entity_targets[inv.target_entity_key] = invalid_at
        count = 0
        for i, row in enumerate(self.store.rows):
            if row.pot_id != pot_id or row.invalid_at is not None:
                continue
            triple = (row.predicate.upper(), row.subject_key, row.object_key)
            invalid_at = edge_targets.get(triple)
            if invalid_at is None:
                invalid_at = entity_targets.get(row.subject_key) or entity_targets.get(
                    row.object_key
                )
            if invalid_at is not None:
                self.store.rows[i] = _with_invalid_at(row, invalid_at)
                count += 1
        return count

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
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
            # Match by first-class claim_key, or by endpoint key (legacy).
            if (
                (row.claim_key and row.claim_key in keys)
                or row.subject_key in keys
                or row.object_key in keys
            ):
                self.store.rows[i] = _with_invalid_at(row, now)
                invalidated += 1
        self._notify()
        return invalidated

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        before = len(self.store.rows)
        self.store.rows = [r for r in self.store.rows if r.pot_id != pot_id]
        for key in [k for k in self.store.entity_label_index if k[0] == pot_id]:
            self.store.entity_label_index.pop(key, None)
        for key in [k for k in self.store.entity_property_index if k[0] == pot_id]:
            self.store.entity_property_index.pop(key, None)
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
        self,
        *,
        pot_id: str,
        entity_key: str,
        depth: int = 1,
        direction: str = "both",
        predicates: tuple[str, ...] = (),
        limit: int | None = None,
    ) -> GraphSlice:
        seen_nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()
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
                if row.pot_id != pot_id:
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
                edge_key = (row.subject_key, row.predicate, row.object_key)
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
        props = normalize_entity_properties(
            self.store.entity_properties(pot_id=pot_id, entity_key=key),
            entity_key=key,
        )
        return GraphNode(key=key, labels=labels, properties=props)


@dataclass(slots=True)
class _Analytics:
    store: InMemoryClaimQueryStore
    on_change: Any = None

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
        repaired: dict[str, int] = {}
        details: list[str] = []
        changed = False
        if wants_semantic_index_repair(targets) and self.store.embedder is not None:
            reembedded = self._repair_semantic_index(pot_id)
            repaired[SEMANTIC_INDEX_TARGET] = reembedded
            details.append(f"re-embedded {reembedded} stale claim(s)")
            changed = changed or bool(reembedded)
        if wants_entity_summary_repair(targets):
            count = self._repair_entity_summaries(pot_id)
            repaired[ENTITY_SUMMARY_TARGET] = count
            details.append(f"repaired {count} entity summaries")
            changed = changed or bool(count)
        if changed and self.on_change is not None:
            self.on_change()
        if repaired or details:
            return RepairReport(
                pot_id=pot_id,
                targets=tuple(targets),
                repaired=repaired,
                detail="; ".join(details) or None,
            )
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired={},
            detail="in_memory projections are computed on read; nothing to rebuild",
        )

    def _repair_semantic_index(self, pot_id: str) -> int:
        """Re-embed rows whose stored vector is missing or mis-sized."""
        embedder = self.store.embedder
        assert embedder is not None
        dims = int(getattr(embedder, "dimensions", 0))
        repaired = 0
        for idx, row in enumerate(self.store.rows):
            if row.pot_id != pot_id:
                continue
            if row.fact_embedding is not None and len(row.fact_embedding) == dims:
                continue
            try:
                embedding = tuple(
                    float(x) for x in embedder.embed(card_for_row(row))
                )
            except Exception:  # noqa: BLE001 - repair must not die on one row.
                continue
            self.store.rows[idx] = _with_embedding(row, embedding)
            repaired += 1
        return repaired

    def _repair_entity_summaries(self, pot_id: str) -> int:
        entity_keys = {
            key
            for row in self._rows(pot_id)
            for key in (row.subject_key, row.object_key)
        }
        entity_keys.update(
            key for pid, key in self.store.entity_property_index if pid == pot_id
        )
        repaired = 0
        for entity_key in entity_keys:
            props = self.store.entity_properties(pot_id=pot_id, entity_key=entity_key)
            fixed = repaired_entity_properties(entity_key, props)
            if fixed is None:
                continue
            self.store.set_entity_properties(
                pot_id=pot_id, entity_key=entity_key, properties=fixed
            )
            repaired += 1
        return repaired


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
    embedder: EmbedderPort | None = None
    _mutation: _Mutation = field(init=False)
    _semantic: _Semantic = field(init=False)
    _inspection: _Inspection = field(init=False)
    _analytics: _Analytics = field(init=False)
    _snapshot: _Snapshot = field(init=False)

    def __post_init__(self) -> None:
        # The embedder powers vector search on read and embed-on-write; share it
        # with the store so ``find_claims`` and ``mutation.apply`` agree on mode.
        if self.embedder is not None and self.store.embedder is None:
            self.store.embedder = self.embedder
        self._mutation = _Mutation(
            self.store,
            on_change=self.on_change,
            profile=self.profile_name,
            embedder=self.store.embedder,
        )
        self._semantic = _Semantic(self.store)
        self._inspection = _Inspection(self.store)
        self._analytics = _Analytics(self.store, on_change=self.on_change)
        self._snapshot = _Snapshot(self.store)

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
    properties = {
        **dict(row.properties),
        "claim_key": row.claim_key,
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


def _with_invalid_at(row: ClaimRow, when: datetime) -> ClaimRow:
    import dataclasses

    return dataclasses.replace(row, invalid_at=when)


def _with_embedding(row: ClaimRow, embedding: tuple[float, ...]) -> ClaimRow:
    import dataclasses

    return dataclasses.replace(row, fact_embedding=embedding)


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


def _vector_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return tuple(float(x) for x in value)
        except (TypeError, ValueError):
            return None
    return None


def _evidence_tuple(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _reader_extras(props: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in props.items() if k not in CONTRACT_EDGE_KEYS}


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
    }


def _row_from_dict(pot_id: str, raw: Mapping[str, Any]) -> ClaimRow:
    embedding = raw.get("fact_embedding")
    source_refs = raw.get("source_refs") or ()
    truth = _coerce_str(raw.get("truth"))
    return ClaimRow(
        pot_id=pot_id,
        predicate=raw["predicate"],
        subject_key=raw["subject_key"],
        object_key=raw["object_key"],
        valid_at=_coerce_dt(raw.get("valid_at")),
        invalid_at=_coerce_dt(raw.get("invalid_at")),
        evidence_strength=evidence_strength_for_truth(truth),
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
    )


def dump_store(store: InMemoryClaimQueryStore) -> dict[str, Any]:
    """Serialize a whole (multi-pot) claim store for persistence."""
    return {
        "format_version": "2",
        "rows": [{"pot_id": r.pot_id, **_row_to_dict(r)} for r in store.rows],
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
