"""``EpisodeWriterPort`` implemented via ``EpisodicGraphPort``."""

from __future__ import annotations

from adapters.outbound.graphiti.port import EpisodicGraphPort
from domain.graph_mutations import ProvenanceRef
from domain.ports.episode_writer import EpisodeWriterPort
from domain.reconciliation import EpisodeDraft


class EpisodicEpisodeWriter(EpisodeWriterPort):
    def __init__(self, episodic: EpisodicGraphPort) -> None:
        self._episodic = episodic

    def write_episode_drafts(
        self,
        pot_id: str,
        drafts: list[EpisodeDraft],
        provenance: ProvenanceRef | None,
    ) -> list[str | None]:
        return self._episodic.write_episode_drafts(pot_id, drafts, provenance)
