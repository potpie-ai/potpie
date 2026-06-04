"""Add a raw episode to the Graphiti episodic graph (scoped by pot group_id)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from adapters.outbound.graphiti.port import EpisodicGraphPort
from domain.actor import Actor
from domain.graph_mutations import ProvenanceRef


def _actor_provenance(pot_id: str, actor: Actor | None) -> ProvenanceRef | None:
    """Minimal ProvenanceRef carrying actor fields so they land on the Episodic node."""
    if actor is None:
        return None
    return ProvenanceRef(
        pot_id=pot_id,
        source_event_id="",
        actor_user_id=actor.user_id,
        actor_surface=actor.surface,
        actor_client_name=actor.client_name,
        actor_auth_method=actor.auth_method,
    )


def ingest_episode(
    episodic: EpisodicGraphPort,
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    *,
    actor: Actor | None = None,
) -> dict[str, Any]:
    """Return ``{"episode_uuid": str | None}``."""
    if not episodic.enabled:
        return {"episode_uuid": None}
    provenance = _actor_provenance(pot_id, actor)
    episode_uuid: Optional[str] = episodic.add_episode(
        pot_id=pot_id,
        name=name,
        episode_body=episode_body,
        source_description=source_description,
        reference_time=reference_time,
        provenance=provenance,
    )
    return {"episode_uuid": episode_uuid}
