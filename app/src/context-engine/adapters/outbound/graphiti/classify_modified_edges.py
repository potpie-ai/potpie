"""Maintenance job: reclassify vague ``MODIFIED`` episodic edges (Neo4j).

Writes are allowed by default; set ``CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE=0`` or
``CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES=0`` to block non-dry-run updates.

See docs/context-graph-improvements/02-edge-type-collapse.md.
"""

from __future__ import annotations

import logging
from typing import Any

from domain.entity_schema import normalized_episodic_edge_allowlist
from domain.extraction_edges import classify_episodic_edge, normalize_relation_name
from domain.reconciliation_flags import (
    allow_edge_classify_write_enabled,
    classify_modified_edges_enabled,
)

logger = logging.getLogger(__name__)


_FETCH_MODIFIED = """
MATCH (a:Entity)-[e:RELATES_TO]->(b:Entity)
WHERE e.group_id = $gid
  AND toUpper(trim(e.name)) = 'MODIFIED'
RETURN e.uuid AS uuid,
       e.name AS name,
       e.fact AS fact,
       e.lifecycle_status AS lifecycle_status,
       labels(a) AS source_labels,
       labels(b) AS target_labels
"""

_UPDATE_EDGE = """
MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
SET e.name = $name,
    e.lifecycle_status = $lifecycle_status
RETURN e.uuid AS uuid
"""


async def classify_modified_edges_for_group(
    driver: Any,
    group_id: str,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Scan MODIFIED edges for ``group_id`` and apply the same rules as ingest normalization."""
    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:  # pragma: no cover
        logger.debug("graphiti_core not available: %s", exc)
        return {"ok": False, "error": "graphiti_core_unavailable"}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {"ok": True, "skipped": "unsupported_provider"}

    if not dry_run and not allow_edge_classify_write_enabled():
        return {
            "ok": False,
            "error": "write_blocked",
            "hint": "Writes blocked (CONTEXT_ENGINE_ALLOW_EDGE_CLASSIFY_WRITE=0); set to 1 to allow.",
        }

    if not dry_run and not classify_modified_edges_enabled():
        return {
            "ok": False,
            "error": "job_disabled",
            "hint": "Job disabled (CONTEXT_ENGINE_CLASSIFY_MODIFIED_EDGES=0); set to 1 to allow writes.",
        }

    records, _, _ = await driver.execute_query(_FETCH_MODIFIED, gid=group_id)
    allowed = normalized_episodic_edge_allowlist()

    examined = len(records)
    would_update = 0
    updated = 0
    errors: list[str] = []
    samples: list[dict[str, Any]] = []

    for rec in records:
        uuid = str(rec.get("uuid") or "")
        fact = str(rec.get("fact") or "")
        src = tuple(rec.get("source_labels") or ())
        tgt = tuple(rec.get("target_labels") or ())
        old_name = str(rec.get("name") or "")
        old_ls = rec.get("lifecycle_status")
        old_ls_s = str(old_ls) if old_ls not in (None, "") else None

        new_name, new_ls = classify_episodic_edge(
            old_name,
            fact,
            src,
            tgt,
            allowed_normalized_names=allowed,
            existing_lifecycle=old_ls_s,
        )

        name_changed = normalize_relation_name(new_name) != normalize_relation_name(
            old_name
        )
        ls_changed = (new_ls or "") != (old_ls_s or "")

        if name_changed or ls_changed:
            would_update += 1
            if len(samples) < 12:
                samples.append(
                    {
                        "uuid": uuid,
                        "old_name": old_name,
                        "new_name": new_name,
                        "old_lifecycle": old_ls_s,
                        "new_lifecycle": new_ls,
                    }
                )

        if dry_run:
            continue

        try:
            await driver.execute_query(
                _UPDATE_EDGE,
                uuid=uuid,
                name=new_name,
                lifecycle_status=new_ls,
            )
            if name_changed or ls_changed:
                updated += 1
        except Exception as exc:
            errors.append(f"{uuid}: {exc}")
            logger.warning("classify_modified_edges update failed: %s", exc)

    out: dict[str, Any] = {
        "ok": True,
        "group_id": group_id,
        "dry_run": dry_run,
        "examined": examined,
        "would_update": would_update,
        "samples": samples,
    }
    if not dry_run:
        out["updated"] = updated
        out["errors"] = errors
    return out
