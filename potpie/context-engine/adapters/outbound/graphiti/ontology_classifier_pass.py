"""Post-ingest Neo4j pass: pin Graphiti-extracted nodes to the canonical ontology.

For every ``Entity`` node in a pot, collect the signals the classifier needs
(existing labels, properties, names of incoming/outgoing ``RELATES_TO`` edges)
and apply any canonical labels it recommends. This is the single enforcement
point that turns flexible episodic extraction into governed vocabulary.

The pass is idempotent — running it repeatedly converges. It is gated by
``CONTEXT_ENGINE_INFER_LABELS`` on the ingest path, and always runs from the
maintenance backfill entrypoint.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from domain.ontology import ENTITY_TYPES, is_canonical_entity_label
from domain.ontology_classifier import build_signals, classify_entity
from domain.reconciliation_flags import infer_canonical_labels_enabled

logger = logging.getLogger(__name__)


def _safe_label(label: str) -> str:
    if not is_canonical_entity_label(label) or label not in ENTITY_TYPES:
        raise ValueError(f"unsafe or unknown ontology label: {label!r}")
    return label


async def run_ontology_classifier_pass(
    driver: Any, group_id: str, *, force: bool = False
) -> dict[str, Any]:
    """Pin ``Entity`` nodes in one pot to the canonical ontology.

    When ``force`` is True (maintenance backfill), bypass the
    ``CONTEXT_ENGINE_INFER_LABELS`` gate.

    Returns a summary with ``ok``, ``group_id``, ``entities_classified``
    (nodes that gained at least one label), and ``labels_applied`` (sum of
    label-node pairs written).
    """
    if not force and not infer_canonical_labels_enabled():
        return {
            "ok": True,
            "skipped": "disabled",
            "entities_classified": 0,
            "labels_applied": 0,
        }

    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:  # pragma: no cover
        logger.debug("graphiti_core not available: %s", exc)
        return {"ok": False, "error": "graphiti_core_unavailable"}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {
            "ok": True,
            "skipped": "unsupported_provider",
            "entities_classified": 0,
            "labels_applied": 0,
        }

    nodes = await _load_nodes(driver, group_id)
    if not nodes:
        return {
            "ok": True,
            "group_id": group_id,
            "entities_classified": 0,
            "labels_applied": 0,
        }

    outgoing, incoming = await _load_edge_adjacency(driver, group_id)

    label_to_uuids: dict[str, list[str]] = defaultdict(list)
    classified = 0
    for uuid, (labels, properties) in nodes.items():
        signals = build_signals(
            labels=labels,
            properties=properties,
            outgoing_edge_names=outgoing.get(uuid, ()),
            incoming_edge_names=incoming.get(uuid, ()),
        )
        additions = classify_entity(signals)
        if not additions:
            continue
        classified += 1
        for label in additions:
            label_to_uuids[label].append(uuid)

    applied = 0
    for label, uuids in label_to_uuids.items():
        try:
            safe = _safe_label(label)
        except ValueError:
            continue
        applied += await _batch_set_label(driver, group_id, safe, uuids)

    return {
        "ok": True,
        "group_id": group_id,
        "entities_classified": classified,
        "labels_applied": applied,
    }


async def _load_nodes(
    driver: Any, group_id: str
) -> dict[str, tuple[tuple[str, ...], dict[str, Any]]]:
    query = """
    MATCH (n:Entity {group_id: $gid})
    RETURN n.uuid AS uuid, labels(n) AS labels, properties(n) AS props
    """
    records, _, _ = await driver.execute_query(query, gid=group_id)
    out: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {}
    for row in records or []:
        uuid = row.get("uuid")
        if uuid is None:
            continue
        labels = tuple(str(lb) for lb in (row.get("labels") or ()))
        props = dict(row.get("props") or {})
        out[str(uuid)] = (labels, props)
    return out


async def _load_edge_adjacency(
    driver: Any, group_id: str
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return ``(outgoing, incoming)`` maps: uuid → normalized edge names."""
    query = """
    MATCH (a:Entity {group_id: $gid})-[e:RELATES_TO]->(b:Entity {group_id: $gid})
    WHERE e.invalid_at IS NULL
    RETURN a.uuid AS src, b.uuid AS tgt, toUpper(trim(e.name)) AS name
    """
    records, _, _ = await driver.execute_query(query, gid=group_id)
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for row in records or []:
        name = row.get("name")
        src = row.get("src")
        tgt = row.get("tgt")
        if not name or not src or not tgt:
            continue
        n = str(name)
        outgoing[str(src)].add(n)
        incoming[str(tgt)].add(n)
    return outgoing, incoming


async def _batch_set_label(
    driver: Any, group_id: str, safe_label: str, uuids: list[str]
) -> int:
    if not uuids:
        return 0
    query = f"""
    UNWIND $uuids AS uid
    MATCH (n:Entity {{group_id: $gid, uuid: uid}})
    SET n:{safe_label}
    RETURN count(n) AS cnt
    """
    records, _, _ = await driver.execute_query(query, gid=group_id, uuids=uuids)
    if not records:
        return 0
    return int(records[0].get("cnt") or 0)
