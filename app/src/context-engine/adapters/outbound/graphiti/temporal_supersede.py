"""Post-ingest predicate-family auto-supersede for Graphiti entity edges (Neo4j).

On by default; set ``CONTEXT_ENGINE_AUTO_SUPERSEDE=0`` to disable.
See docs/context-graph-improvements/01-temporal-resolution-in-search.md
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from domain.graph_mutations import EpisodicSupersessionRecord
from domain.graph_quality import temporal_supersession_resolved_issue
from domain.reconciliation_flags import auto_supersede_enabled
from domain.ontology import (
    normalize_graphiti_edge_name,
    object_counterparty_uuid_for_edge,
    predicate_family_for_episodic_supersede,
    temporal_subject_key_for_edge,
)

logger = logging.getLogger(__name__)


def _families_from_env() -> frozenset[str] | None:
    """None means all families in ``PREDICATE_FAMILY_EDGE_NAMES``."""
    raw = os.getenv("CONTEXT_ENGINE_AUTO_SUPERSEDE_FAMILIES", "").strip()
    if not raw:
        return None
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return frozenset(parts)


def _normalize_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        try:
            native = to_native()
            if isinstance(native, datetime):
                return _normalize_dt(native)
        except Exception:
            pass
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


@dataclass(slots=True)
class _EdgeRow:
    uuid: str
    name: str
    source: str
    target: str
    family: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime | None


def _effective_edge_time(edge: _EdgeRow) -> datetime | None:
    """Observation time for ordering; matches conflict detection (``valid_at`` then ``created_at``)."""
    return edge.valid_at or edge.created_at


async def apply_predicate_family_auto_supersede(driver: Any, group_id: str) -> dict[str, Any]:
    """Invalidate older same-subject / same-family edges when the object differs (Neo4j only)."""
    if not auto_supersede_enabled():
        return {"ok": True, "skipped": "disabled"}

    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("graphiti_core not available for auto_supersede: %s", exc)
        return {"ok": False, "error": "graphiti_core_unavailable"}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {"ok": True, "skipped": "unsupported_provider"}

    families_filter = _families_from_env()
    query = """
    MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
    WHERE e.group_id = $gid AND e.invalid_at IS NULL
    RETURN e.uuid AS uuid, e.name AS name,
           a.uuid AS source, b.uuid AS target,
           labels(b) AS target_labels,
           e.valid_at AS valid_at, e.invalid_at AS invalid_at, e.created_at AS created_at
    """

    records, _, _ = await driver.execute_query(query, gid=group_id)

    rows: list[_EdgeRow] = []
    for rec in records:
        name = str(rec.get("name") or "")
        if not normalize_graphiti_edge_name(name):
            continue
        raw_labels = rec.get("target_labels")
        if raw_labels is None:
            target_labels: tuple[str, ...] = ()
        elif isinstance(raw_labels, (list, tuple)):
            target_labels = tuple(str(x) for x in raw_labels)
        else:
            target_labels = (str(raw_labels),)
        fam = predicate_family_for_episodic_supersede(name, target_labels)
        if fam is None:
            continue
        if families_filter is not None and fam not in families_filter:
            continue
        sk = temporal_subject_key_for_edge(
            name, str(rec["source"]), str(rec["target"]), predicate_family=fam
        )
        ok = object_counterparty_uuid_for_edge(
            name, str(rec["source"]), str(rec["target"]), predicate_family=fam
        )
        if sk is None or ok is None:
            continue
        rows.append(
            _EdgeRow(
                uuid=str(rec["uuid"]),
                name=name,
                source=str(rec["source"]),
                target=str(rec["target"]),
                family=fam,
                valid_at=_normalize_dt(rec.get("valid_at")),
                invalid_at=_normalize_dt(rec.get("invalid_at")),
                created_at=_normalize_dt(rec.get("created_at")),
            )
        )

    buckets: dict[tuple[str, str], list[_EdgeRow]] = defaultdict(list)
    for edge in rows:
        sk = temporal_subject_key_for_edge(
            edge.name, edge.source, edge.target, predicate_family=edge.family
        )
        if sk is None:
            continue
        buckets[(edge.family, sk)].append(edge)

    now = datetime.now(timezone.utc)
    invalidated = 0
    audit: list[EpisodicSupersessionRecord] = []
    issues: list[Any] = []

    update_cypher = """
    MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
    WHERE e.uuid = $uuid
    SET e.invalid_at = $invalid_at,
        e.expired_at = coalesce(e.expired_at, $expired_at),
        e.superseded_by_uuid = $superseded_by_uuid
    RETURN e.uuid AS uuid
    """

    for (_fam, _sk), group in buckets.items():
        by_object: dict[str, list[_EdgeRow]] = defaultdict(list)
        for edge in group:
            ok = object_counterparty_uuid_for_edge(
                edge.name, edge.source, edge.target, predicate_family=edge.family
            )
            if ok is None:
                continue
            by_object[ok].append(edge)
        if len(by_object) < 2:
            continue

        times = [_effective_edge_time(e) for e in group if _effective_edge_time(e) is not None]
        if not times:
            # No comparable timestamps — leave edges live for predicate-family conflict detection.
            continue

        max_t = max(times)
        at_max = [e for e in group if _effective_edge_time(e) == max_t]
        objs_at_max: set[str] = set()
        for edge in at_max:
            ok = object_counterparty_uuid_for_edge(
                edge.name, edge.source, edge.target, predicate_family=edge.family
            )
            if ok:
                objs_at_max.add(ok)

        same_time_contradiction = len(objs_at_max) >= 2
        # Stable representative at max_t for ``superseded_by_uuid`` when older edges are invalidated.
        canonical_at_max = min(at_max, key=lambda e: e.uuid).uuid

        async def _invalidate_loser(loser: _EdgeRow, winner: str) -> None:
            nonlocal invalidated
            inv_at = max_t
            await driver.execute_query(
                update_cypher,
                uuid=loser.uuid,
                invalid_at=inv_at,
                expired_at=now,
                superseded_by_uuid=winner,
            )
            invalidated += 1
            fam = loser.family or "unknown"
            audit.append(
                EpisodicSupersessionRecord(
                    group_id=group_id,
                    superseded_edge_uuid=loser.uuid,
                    superseding_edge_uuid=winner,
                    predicate_family=fam,
                )
            )
            issues.append(
                temporal_supersession_resolved_issue(
                    group_id=group_id,
                    superseded_edge_uuid=loser.uuid,
                    superseding_edge_uuid=winner,
                    predicate_family=fam,
                )
            )

        for edge in group:
            if edge.invalid_at is not None:
                continue
            eff = _effective_edge_time(edge)
            if eff is not None and eff < max_t:
                await _invalidate_loser(edge, canonical_at_max)
                continue
            if eff is None and not same_time_contradiction:
                await _invalidate_loser(edge, canonical_at_max)
                continue
            if (
                not same_time_contradiction
                and eff == max_t
                and edge.uuid != canonical_at_max
            ):
                ok = object_counterparty_uuid_for_edge(
                    edge.name, edge.source, edge.target, predicate_family=edge.family
                )
                w_ok = None
                for w in at_max:
                    if w.uuid == canonical_at_max:
                        w_ok = object_counterparty_uuid_for_edge(
                            w.name, w.source, w.target, predicate_family=w.family
                        )
                        break
                if ok and w_ok and ok == w_ok:
                    await _invalidate_loser(edge, canonical_at_max)

    if invalidated:
        logger.info(
            "auto_supersede: invalidated %d edges in group_id=%s", invalidated, group_id
        )

    return {
        "ok": True,
        "group_id": group_id,
        "invalidated_count": invalidated,
        "audit": [asdict(a) for a in audit],
        "quality_issues": [
            {"code": i.code, "message": i.message, "severity": i.severity, "refs": i.refs}
            for i in issues
        ],
    }
