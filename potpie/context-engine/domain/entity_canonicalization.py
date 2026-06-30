"""Normalize entity keys and merge duplicates in a ``ReconciliationPlan``.

Entities arriving from LLM planners or heterogeneous deterministic planners
can drift on casing / whitespace (``Agents`` vs ``agents``; ``context_resolve``
vs ``context_resolve ``). A single normalization step before upsert collapses
the bulk of the drift without touching the extractor.

Conservative rules:
    * Trim leading/trailing whitespace.
    * Lowercase the full key (including scheme prefixes like ``github:pr:``).
    * Collapse internal whitespace runs to a single ``_``.
    * Apply a small ``SYNONYMS`` table last (kept intentionally empty; extended
      as concrete collisions are identified by operators).

When two upserts resolve to the same canonical key:
    * Labels are unioned (first-seen order, deduped).
    * Properties are merged with *first-seen-wins* for scalars — deterministic
      planners typically emit the richest record first (repo → PR → commit …).
    * Edge endpoints and invalidation targets are rewritten to the canonical
      key; edges that become self-loops after rewrite are dropped.
    * Edge upserts are deduped on ``(edge_type, from, to)`` so the merge cannot
      multiply parallel edges.

Plan warnings record how many merges ran so operators can spot an extractor
regression from the ledger without digging into the diff.
"""

from __future__ import annotations

from typing import Iterable

from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
)
from domain.reconciliation import ReconciliationPlan

SYNONYMS: dict[str, str] = {}
"""Post-normalization synonym collapse. Keys and values are already normalized."""


def normalize_entity_key(key: str) -> str:
    """Deterministic key normalization used by the plan canonicalizer."""
    if not key:
        return key
    stripped = key.strip()
    if not stripped:
        return stripped
    collapsed = "_".join(stripped.lower().split())
    return SYNONYMS.get(collapsed, collapsed)


def canonicalize_reconciliation_plan(plan: ReconciliationPlan) -> int:
    """Normalize keys and merge duplicates in place. Returns merge count."""
    rewrite = _build_rewrite_map(plan.entity_upserts)
    rewrite.update(_extra_rewrites_from_edges(plan, rewrite))

    merged, merge_count = _merge_entities(plan.entity_upserts, rewrite)
    plan.entity_upserts = merged
    plan.edge_upserts = _rewrite_edges(plan.edge_upserts, rewrite)
    plan.edge_deletes = _rewrite_edge_deletes(plan.edge_deletes, rewrite)
    plan.invalidations = _rewrite_invalidations(plan.invalidations, rewrite)

    if merge_count:
        plan.warnings.append(
            f"canonicalized {merge_count} duplicate entity key(s) before apply"
        )
    return merge_count


def _build_rewrite_map(items: Iterable[EntityUpsert]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key = item.entity_key
        if not key:
            continue
        canonical = normalize_entity_key(key)
        if canonical and key != canonical:
            out[key] = canonical
    return out


def _extra_rewrites_from_edges(
    plan: ReconciliationPlan, existing: dict[str, str]
) -> dict[str, str]:
    """Normalize endpoints that appear only on edges / invalidations.

    An LLM plan can reference a key on an edge without re-declaring the entity.
    We still want ``A`` and ``a`` endpoints to collapse in that case.
    """
    extras: dict[str, str] = {}
    seen: set[str] = set(existing.keys())

    def _consider(key: str | None) -> None:
        if not key or key in seen:
            return
        seen.add(key)
        canonical = normalize_entity_key(key)
        if canonical and key != canonical:
            extras[key] = canonical

    for edge in plan.edge_upserts:
        _consider(edge.from_entity_key)
        _consider(edge.to_entity_key)
    for edge in plan.edge_deletes:
        _consider(edge.from_entity_key)
        _consider(edge.to_entity_key)
    for inv in plan.invalidations:
        _consider(inv.target_entity_key)
        _consider(inv.superseded_by_key)
        if inv.target_edge is not None:
            _consider(inv.target_edge[1])
            _consider(inv.target_edge[2])
    return extras


def _canonical(rewrite: dict[str, str], key: str) -> str:
    return rewrite.get(key, key)


def _merge_entities(
    items: list[EntityUpsert], rewrite: dict[str, str]
) -> tuple[list[EntityUpsert], int]:
    by_key: dict[str, EntityUpsert] = {}
    order: list[str] = []
    merges = 0
    for item in items:
        canonical_key = _canonical(rewrite, item.entity_key)
        if canonical_key == item.entity_key and canonical_key not in by_key:
            by_key[canonical_key] = item
            order.append(canonical_key)
            continue

        if canonical_key in by_key:
            by_key[canonical_key] = _merge_pair(by_key[canonical_key], item, canonical_key)
            merges += 1
        else:
            by_key[canonical_key] = EntityUpsert(
                entity_key=canonical_key,
                labels=item.labels,
                properties=dict(item.properties),
            )
            order.append(canonical_key)
    return [by_key[k] for k in order], merges


def _merge_pair(
    prior: EntityUpsert, incoming: EntityUpsert, canonical_key: str
) -> EntityUpsert:
    labels = tuple(dict.fromkeys([*prior.labels, *incoming.labels]))
    merged_props = dict(prior.properties)
    for k, v in incoming.properties.items():
        merged_props.setdefault(k, v)
    return EntityUpsert(entity_key=canonical_key, labels=labels, properties=merged_props)


def _rewrite_edges(
    edges: list[EdgeUpsert], rewrite: dict[str, str]
) -> list[EdgeUpsert]:
    out: dict[tuple[str, str, str], EdgeUpsert] = {}
    for edge in edges:
        src = _canonical(rewrite, edge.from_entity_key)
        dst = _canonical(rewrite, edge.to_entity_key)
        if src == dst:
            continue
        edge.from_entity_key = src
        edge.to_entity_key = dst
        out[(edge.edge_type, src, dst)] = edge
    return list(out.values())


def _rewrite_edge_deletes(
    edges: list[EdgeDelete], rewrite: dict[str, str]
) -> list[EdgeDelete]:
    out: dict[tuple[str, str, str], EdgeDelete] = {}
    for edge in edges:
        src = _canonical(rewrite, edge.from_entity_key)
        dst = _canonical(rewrite, edge.to_entity_key)
        if src == dst:
            continue
        edge.from_entity_key = src
        edge.to_entity_key = dst
        out[(edge.edge_type, src, dst)] = edge
    return list(out.values())


def _rewrite_invalidations(
    invalidations: list[InvalidationOp], rewrite: dict[str, str]
) -> list[InvalidationOp]:
    kept: list[InvalidationOp] = []
    for inv in invalidations:
        if inv.target_entity_key:
            inv.target_entity_key = _canonical(rewrite, inv.target_entity_key)
        if inv.superseded_by_key:
            inv.superseded_by_key = _canonical(rewrite, inv.superseded_by_key)
        if inv.target_edge is not None:
            et, src, dst = inv.target_edge
            src = _canonical(rewrite, src)
            dst = _canonical(rewrite, dst)
            if src == dst:
                continue
            inv.target_edge = (et, src, dst)
        kept.append(inv)
    return kept
