"""Post-ingest predicate-family conflict detection (Neo4j + Graphiti episodic edges).

On by default; set ``CONTEXT_ENGINE_CONFLICT_DETECT=0`` to disable.
See docs/context-graph-improvements/06-conflict-surfacing.md
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from domain.graph_quality import (
    EpisodicEdgeConflictInput,
    detect_family_conflicts,
)
from domain.ontology import normalize_graphiti_edge_name
from domain.reconciliation_flags import conflict_detection_enabled

logger = logging.getLogger(__name__)


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


_LIVE_EDGES_QUERY = """
MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
WHERE e.group_id = $gid AND e.invalid_at IS NULL
RETURN e.uuid AS uuid, e.name AS name,
       a.uuid AS source, b.uuid AS target,
       labels(b) AS target_labels,
       e.valid_at AS valid_at, e.created_at AS created_at
"""


async def apply_family_conflict_detection(driver: Any, group_id: str) -> dict[str, Any]:
    """Detect predicate-family conflicts among live episodic edges; upsert ``QualityIssue`` rows."""
    if not conflict_detection_enabled():
        return {"ok": True, "skipped": "disabled"}

    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("graphiti_core not available for conflict detection: %s", exc)
        return {"ok": False, "error": "graphiti_core_unavailable"}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {"ok": True, "skipped": "unsupported_provider"}

    records, _, _ = await driver.execute_query(_LIVE_EDGES_QUERY, gid=group_id)

    inputs: list[EpisodicEdgeConflictInput] = []
    for rec in records:
        name = str(rec.get("name") or "")
        if not normalize_graphiti_edge_name(name):
            continue
        raw_labels = rec.get("target_labels")
        if raw_labels is None:
            tl: tuple[str, ...] = ()
        elif isinstance(raw_labels, (list, tuple)):
            tl = tuple(str(x) for x in raw_labels)
        else:
            tl = (str(raw_labels),)
        inputs.append(
            EpisodicEdgeConflictInput(
                uuid=str(rec["uuid"]),
                name=name,
                source_uuid=str(rec["source"]),
                target_uuid=str(rec["target"]),
                valid_at=_normalize_dt(rec.get("valid_at")),
                created_at=_normalize_dt(rec.get("created_at")),
                target_labels=tl,
            )
        )

    candidates = detect_family_conflicts(inputs)
    created = 0

    check_open = """
    MATCH (qi:Entity:QualityIssue {group_id: $gid, status: 'open'})
    WHERE qi.kind = 'conflict'
      AND (
        (qi.edge_a_uuid = $ea AND qi.edge_b_uuid = $eb)
        OR (qi.edge_a_uuid = $eb AND qi.edge_b_uuid = $ea)
      )
    RETURN qi.uuid AS uuid LIMIT 1
    """

    create_cypher = """
    CREATE (qi:Entity:QualityIssue {
      uuid: $issue_uuid,
      group_id: $gid,
      code: 'predicate_family_conflict',
      kind: 'conflict',
      pair_key: $pair_key,
      severity: $severity,
      status: 'open',
      family: $family,
      subject_uuid: $subject_uuid,
      edge_a_uuid: $edge_a_uuid,
      edge_b_uuid: $edge_b_uuid,
      conflict_type: $conflict_type,
      auto_resolvable: $auto_resolvable,
      suggested_action: $suggested_action,
      detected_at: $detected_at
    })
    RETURN qi.uuid AS uuid
    """

    flag_subject_cypher = """
    MATCH (qi:Entity:QualityIssue {uuid: $issue_uuid, group_id: $gid})
    MATCH (subj:Entity {uuid: $subject_uuid, group_id: $gid})
    MERGE (qi)-[:FLAGS]->(subj)
    RETURN qi.uuid AS uuid
    """

    for cand in candidates:
        ea = str(cand["edge_a_uuid"])
        eb = str(cand["edge_b_uuid"])
        pair_key = f"{min(ea, eb)}|{max(ea, eb)}"
        dup, _, _ = await driver.execute_query(
            check_open,
            gid=group_id,
            ea=ea,
            eb=eb,
        )
        if dup:
            continue
        issue_uuid = str(uuid.uuid4())
        detected_at = datetime.now(timezone.utc).isoformat()
        recs, _, _ = await driver.execute_query(
            create_cypher,
            gid=group_id,
            issue_uuid=issue_uuid,
            pair_key=pair_key,
            severity=str(cand.get("severity") or "warning"),
            family=str(cand.get("family") or ""),
            subject_uuid=str(cand.get("subject_uuid") or ""),
            edge_a_uuid=ea,
            edge_b_uuid=eb,
            conflict_type=str(cand.get("conflict_type") or "overlap"),
            auto_resolvable=bool(cand.get("auto_resolvable")),
            suggested_action=str(cand.get("suggested_action") or "human_review"),
            detected_at=detected_at,
        )
        row = recs[0] if recs else None
        if row and row.get("uuid"):
            created += 1
            await driver.execute_query(
                flag_subject_cypher,
                gid=group_id,
                issue_uuid=str(row["uuid"]),
                subject_uuid=str(cand.get("subject_uuid") or ""),
            )

    return {
        "ok": True,
        "group_id": group_id,
        "candidates": len(candidates),
        "issues_created": created,
    }


_CONFLICT_LIST_QUERY = """
MATCH (qi:Entity:QualityIssue {group_id: $gid, kind: 'conflict', status: 'open'})
WHERE qi.code = 'predicate_family_conflict' OR qi.code IS NULL
RETURN qi.uuid AS uuid,
       qi.family AS family,
       qi.subject_uuid AS subject_uuid,
       qi.edge_a_uuid AS edge_a_uuid,
       qi.edge_b_uuid AS edge_b_uuid,
       qi.conflict_type AS conflict_type,
       qi.auto_resolvable AS auto_resolvable,
       qi.suggested_action AS suggested_action,
       qi.severity AS severity,
       qi.detected_at AS detected_at
"""


async def list_open_conflicts_async(driver: Any, group_id: str) -> list[dict[str, Any]]:
    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception:
        return []

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return []

    recs, _, _ = await driver.execute_query(_CONFLICT_LIST_QUERY, gid=group_id)
    out: list[dict[str, Any]] = []
    for row in recs:
        out.append(
            {
                "uuid": str(row.get("uuid") or ""),
                "family": row.get("family"),
                "subject_uuid": row.get("subject_uuid"),
                "edge_a_uuid": row.get("edge_a_uuid"),
                "edge_b_uuid": row.get("edge_b_uuid"),
                "conflict_type": row.get("conflict_type"),
                "auto_resolvable": row.get("auto_resolvable"),
                "suggested_action": row.get("suggested_action"),
                "severity": row.get("severity"),
                "detected_at": row.get("detected_at"),
            }
        )
    return out


_RESOLVE_SUPERSEDE_CYPHER = """
MATCH (qi:Entity:QualityIssue {uuid: $issue_uuid, group_id: $gid})
WHERE qi.status = 'open'
SET qi.status = 'closed',
    qi.resolved_at = $resolved_at,
    qi.resolution_action = $action
WITH qi
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid = $older_edge AND e.group_id = $gid AND e.invalid_at IS NULL
SET e.invalid_at = $resolved_at,
    e.superseded_by_uuid = $newer_edge,
    e.resolved_conflict_issue_uuid = $issue_uuid
RETURN qi.uuid AS qi_uuid, e.uuid AS edge_uuid
"""


async def resolve_conflict_supersede_older_async(
    driver: Any,
    group_id: str,
    issue_uuid: str,
) -> dict[str, Any]:
    """Close a conflict issue and stamp ``invalid_at`` on the older episodic edge."""
    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {"ok": False, "error": "unsupported_provider"}

    recs, _, _ = await driver.execute_query(
        """
        MATCH (qi:Entity:QualityIssue {uuid: $iid, group_id: $gid})
        WHERE qi.status = 'open'
        RETURN qi.edge_a_uuid AS ea, qi.edge_b_uuid AS eb, qi.detected_at AS da
        """,
        iid=issue_uuid,
        gid=group_id,
    )
    if not recs:
        return {"ok": False, "error": "issue_not_found"}
    ea = str(recs[0].get("ea") or "")
    eb = str(recs[0].get("eb") or "")
    if not ea or not eb:
        return {"ok": False, "error": "issue_missing_edges"}

    edge_rows, _, _ = await driver.execute_query(
        """
        MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
        WHERE e.group_id = $gid AND e.uuid IN $uuids AND e.invalid_at IS NULL
        RETURN e.uuid AS uuid, e.valid_at AS valid_at, e.created_at AS created_at
        """,
        gid=group_id,
        uuids=[ea, eb],
    )
    if len(edge_rows) != 2:
        return {"ok": False, "error": "edges_not_both_live"}

    def eff(r: dict[str, Any]) -> datetime | None:
        return _normalize_dt(r.get("valid_at")) or _normalize_dt(r.get("created_at"))

    r0, r1 = edge_rows[0], edge_rows[1]
    t0, t1 = eff(r0), eff(r1)
    if t0 is not None and t1 is not None:
        older_id = str(r0["uuid"]) if t0 <= t1 else str(r1["uuid"])
        newer_id = str(r1["uuid"]) if t0 <= t1 else str(r0["uuid"])
    else:
        older_id, newer_id = sorted([str(r0["uuid"]), str(r1["uuid"])])

    now = datetime.now(timezone.utc).isoformat()
    out, _, _ = await driver.execute_query(
        _RESOLVE_SUPERSEDE_CYPHER,
        issue_uuid=issue_uuid,
        gid=group_id,
        older_edge=older_id,
        newer_edge=newer_id,
        resolved_at=now,
        action="supersede_older",
    )
    ok = bool(out)
    return {"ok": ok, "older_edge_uuid": older_id, "newer_edge_uuid": newer_id}
