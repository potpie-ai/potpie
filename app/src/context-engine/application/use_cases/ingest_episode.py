"""Add a raw episode to the Graphiti episodic graph (scoped by pot group_id)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from domain.ports.episodic_graph import EpisodicGraphPort


def ingest_episode(
    episodic: EpisodicGraphPort,
    pot_id: str,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
) -> dict[str, Any]:
    """Return ``{"episode_uuid": str | None}``."""
    if not episodic.enabled:
        return {"episode_uuid": None}
    episode_uuid: Optional[str] = episodic.add_episode(
        pot_id=pot_id,
        name=name,
        episode_body=episode_body,
        source_description=source_description,
        reference_time=reference_time,
    )
    return {"episode_uuid": episode_uuid}
