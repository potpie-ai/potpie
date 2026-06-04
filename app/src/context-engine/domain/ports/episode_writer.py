"""Episodic writes from drafts (port)."""

from __future__ import annotations

from typing import Protocol

from domain.graph_mutations import ProvenanceRef
from domain.reconciliation import EpisodeDraft


class EpisodeWriterPort(Protocol):
    def write_episode_drafts(
        self,
        pot_id: str,
        drafts: list[EpisodeDraft],
        provenance: ProvenanceRef | None,
    ) -> list[str | None]:
        """Write one or more episodes; return Graphiti UUIDs (or None if disabled)."""
