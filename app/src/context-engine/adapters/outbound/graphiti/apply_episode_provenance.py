"""Post-write pass: stamp provenance on a Graphiti episode and its extracted edges.

After ``g.add_episode()`` returns, Graphiti has created one ``Episodic`` node
plus one or more ``Entity``→``Entity`` relationships whose ``episodes``
property lists the episode uuid. This pass copies the full
:class:`ProvenanceRef` onto both the Episodic node and those relationships
so search-row readers see the same ``prov_*`` contract that the Neo4j
structural applier already stamps on canonical mutations.
"""

from __future__ import annotations

import logging
from typing import Any

from domain.graph_mutations import ProvenanceRef

logger = logging.getLogger(__name__)


async def apply_episode_provenance(
    driver: Any,
    group_id: str,
    episode_uuid: str,
    provenance: ProvenanceRef | None,
) -> dict[str, Any]:
    """Stamp provenance.to_properties() on the Episodic node and its entity edges."""
    if not episode_uuid or provenance is None:
        return {"ok": True, "skipped": "no_episode_or_provenance", "edges_stamped": 0}

    try:
        from graphiti_core.driver.driver import GraphProvider
    except Exception as exc:  # pragma: no cover
        logger.debug("graphiti_core not available: %s", exc)
        return {"ok": False, "error": "graphiti_core_unavailable"}

    if getattr(driver, "provider", None) != GraphProvider.NEO4J:
        return {"ok": True, "skipped": "unsupported_provider", "edges_stamped": 0}

    props = provenance.to_properties()
    if not props:
        return {"ok": True, "skipped": "empty_provenance", "edges_stamped": 0}

    try:
        await driver.execute_query(
            """
            MATCH (ep:Episodic {uuid: $episode_uuid, group_id: $gid})
            SET ep += $props
            """,
            episode_uuid=episode_uuid,
            gid=group_id,
            props=props,
        )
    except Exception as exc:
        logger.warning("episode provenance node stamp failed: %s", exc)
        return {"ok": False, "error": str(exc), "edges_stamped": 0}

    stamped = 0
    try:
        records, _, _ = await driver.execute_query(
            """
            MATCH (a:Entity {group_id: $gid})-[r]->(b:Entity {group_id: $gid})
            WHERE $episode_uuid IN coalesce(r.episodes, [])
            SET r += $props
            RETURN count(r) AS cnt
            """,
            gid=group_id,
            episode_uuid=episode_uuid,
            props=props,
        )
        if records:
            stamped = int(records[0].get("cnt") or 0)
    except Exception as exc:
        logger.warning("episode provenance edge stamp failed: %s", exc)
        return {"ok": False, "error": str(exc), "edges_stamped": stamped}

    return {"ok": True, "group_id": group_id, "edges_stamped": stamped}
