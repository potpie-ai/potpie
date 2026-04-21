"""Post-ingest Neo4j pass: add canonical ontology labels on Entity endpoints of RELATES_TO edges.

See docs/context-graph-improvements/03-canonical-node-labels.md
"""

from __future__ import annotations

import logging
from typing import Any

from domain.reconciliation_flags import infer_canonical_labels_enabled
from domain.ontology import (
    ENTITY_TYPES,
    EDGE_ENDPOINT_INFERRED_LABELS,
    is_canonical_entity_label,
)

logger = logging.getLogger(__name__)


def _safe_label(label: str) -> str:
    if not is_canonical_entity_label(label) or label not in ENTITY_TYPES:
        raise ValueError(f"unsafe or unknown ontology label: {label!r}")
    return label


async def apply_episodic_canonical_labels(
    driver: Any, group_id: str, *, force: bool = False
) -> dict[str, Any]:
    """For each episodic ``RELATES_TO`` edge, add inferred labels on source/target nodes.

    When ``force`` is True (maintenance backfill), ignore ``CONTEXT_ENGINE_INFER_LABELS``.
    """
    if not force and not infer_canonical_labels_enabled():
        return {"ok": True, "skipped": "disabled", "labels_applied": 0}

    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:  # pragma: no cover
        logger.debug("graphiti_core not available: %s", exc)
        return {"ok": False, "error": "graphiti_core_unavailable"}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {"ok": True, "skipped": "unsupported_provider", "labels_applied": 0}

    total = 0
    for (edge_norm, role), labels in EDGE_ENDPOINT_INFERRED_LABELS.items():
        for label in labels:
            try:
                safe = _safe_label(label)
            except ValueError:
                continue
            if role == "target":
                query = f"""
                MATCH (a:Entity {{group_id: $gid}})-[e:RELATES_TO]->(b:Entity {{group_id: $gid}})
                WHERE toUpper(trim(e.name)) = $edge_name
                  AND e.invalid_at IS NULL
                SET b:{safe}
                RETURN count(b) AS cnt
                """
            else:
                query = f"""
                MATCH (a:Entity {{group_id: $gid}})-[e:RELATES_TO]->(b:Entity {{group_id: $gid}})
                WHERE toUpper(trim(e.name)) = $edge_name
                  AND e.invalid_at IS NULL
                SET a:{safe}
                RETURN count(a) AS cnt
                """
            records, _, _ = await driver.execute_query(
                query,
                gid=group_id,
                edge_name=edge_norm,
            )
            cnt = 0
            if records:
                cnt = int(records[0].get("cnt") or 0)
            total += cnt

    hint_total = await _apply_canonical_type_hints(driver, group_id)
    total += hint_total

    return {"ok": True, "group_id": group_id, "labels_applied": total}


async def _apply_canonical_type_hints(driver: Any, group_id: str) -> int:
    """When ``canonical_type`` is stored on an Entity, add that ontology label if valid."""
    distinct_q = """
    MATCH (e:Entity {group_id: $gid})
    WHERE e.canonical_type IS NOT NULL
    RETURN DISTINCT trim(toString(e.canonical_type)) AS ct
    """
    recs, _, _ = await driver.execute_query(distinct_q, gid=group_id)
    total = 0
    seen: set[str] = set()
    for row in recs or []:
        raw = row.get("ct")
        if raw is None:
            continue
        ct = str(raw).strip()
        if not ct or ct in seen:
            continue
        seen.add(ct)
        if ct not in ENTITY_TYPES:
            continue
        safe = _safe_label(ct)
        query = f"""
        MATCH (e:Entity {{group_id: $gid}})
        WHERE trim(toString(e.canonical_type)) = $ct
        SET e:{safe}
        RETURN count(e) AS cnt
        """
        out, _, _ = await driver.execute_query(query, gid=group_id, ct=ct)
        if out:
            total += int(out[0].get("cnt") or 0)
    return total


async def relabel_nodes_from_edges(driver: Any, group_id: str) -> dict[str, Any]:
    """Idempotent maintenance/backfill over existing pots (ignores ``CONTEXT_ENGINE_INFER_LABELS``)."""
    return await apply_episodic_canonical_labels(driver, group_id, force=True)
