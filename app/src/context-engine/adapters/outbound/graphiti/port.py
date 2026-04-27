"""Graphiti episodic graph port (adapter-internal).

This protocol used to live in ``domain/ports/``. It was moved in the P3
cleanup because the only callers are the graphiti adapter itself, the
Neo4j structural counterpart, and intelligence adapters — none of which
belong to the application layer. Application-level code depends on
:class:`domain.ports.context_graph.ContextGraphPort` only.
"""

from datetime import datetime
from typing import Any, Optional, Protocol

from domain.graph_mutations import ProvenanceRef
from domain.reconciliation import EpisodeDraft


class EpisodicGraphPort(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def add_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        provenance: ProvenanceRef | None = None,
    ) -> Optional[str]:
        """Return episode UUID or None if disabled/failed."""

    def write_episode_drafts(
        self,
        pot_id: str,
        drafts: list[EpisodeDraft],
        provenance: ProvenanceRef | None = None,
    ) -> list[Optional[str]]:
        """Write one or more ``EpisodeDraft`` values; return UUIDs in order."""
        ...

    async def add_episode_async(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        provenance: ProvenanceRef | None = None,
    ) -> Optional[str]:
        ...

    def search(
        self,
        pot_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        repo_name: str | None = None,
        source_description: str | None = None,
        *,
        include_invalidated: bool = False,
        as_of: Optional[datetime] = None,
        episode_uuid: str | None = None,
    ) -> list[Any]:
        ...

    async def search_async(
        self,
        pot_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        repo_name: str | None = None,
        source_description: str | None = None,
        *,
        include_invalidated: bool = False,
        as_of: Optional[datetime] = None,
        episode_uuid: str | None = None,
    ) -> list[Any]:
        ...

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        """Delete all Graphiti episodic data for this pot (Neo4j database partition)."""
        ...

    async def reset_pot_async(self, pot_id: str) -> dict[str, Any]:
        ...

    def list_open_conflicts(self, pot_id: str) -> list[dict[str, Any]]:
        """Open ``QualityIssue`` rows for predicate-family conflicts in this pot."""
        ...

    def resolve_open_conflict(
        self, pot_id: str, issue_uuid: str, action: str
    ) -> dict[str, Any]:
        """Resolve a conflict issues (e.g. ``action=supersede_older``)."""
        ...
