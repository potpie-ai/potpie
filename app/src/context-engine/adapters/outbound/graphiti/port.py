"""Graphiti episodic graph port (adapter-internal).

This is the substrate-level write port for the Context Engine. It owns:

- Episodic ingestion via Graphiti (``add_episode``, ``write_episode_drafts``).
- Canonical ontology mutations layered on top of Graphiti's driver
  (``apply_entity_upserts``, ``apply_edge_upserts``, ``apply_edge_deletes``,
  ``apply_invalidations``). Phase 1 of the rebuild consolidated these
  here so there is a single writer.
- Pot reset (``reset_pot``).
- Read-side helpers Graphiti's high-level API does not expose
  (``search``, conflict listing/resolution, classifier replay).

Read-only structural queries (decisions, change history, project map,
debugging memory, …) live behind :class:`StructuralReadPort`.
Application-level code depends on
:class:`domain.ports.context_graph.ContextGraphPort` only.
"""

from datetime import datetime
from typing import Any, Optional, Protocol

from domain.graph_mutations import (
    EdgeDelete,
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceRef,
)
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

    def apply_entity_upserts(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        """MERGE canonical entities by ``(group_id, entity_key)``; return count."""
        ...

    def apply_edge_upserts(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        """MERGE canonical typed edges between two entity_keys; return count."""
        ...

    def apply_edge_deletes(
        self,
        pot_id: str,
        items: list[EdgeDelete],
        provenance: ProvenanceRef,
    ) -> int:
        """Hard-delete canonical edges (audit fields stamped); return count."""
        ...

    def apply_invalidations(
        self,
        pot_id: str,
        items: list[InvalidationOp],
        provenance: ProvenanceRef,
    ) -> int:
        """Stamp ``valid_to`` on entity/edge and optionally add SUPERSEDES."""
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

    async def write_episode_drafts_async(
        self,
        pot_id: str,
        drafts: list[EpisodeDraft],
        provenance: ProvenanceRef | None = None,
    ) -> list[Optional[str]]:
        """Async-native episode batch write."""
        ...

    async def apply_entity_upserts_async(
        self,
        pot_id: str,
        items: list[EntityUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        """Async-native MERGE for canonical entities. Use from async callers."""
        ...

    async def apply_edge_upserts_async(
        self,
        pot_id: str,
        items: list[EdgeUpsert],
        provenance: ProvenanceRef,
    ) -> int:
        """Async-native MERGE for canonical edges. Use from async callers."""
        ...

    async def apply_edge_deletes_async(
        self,
        pot_id: str,
        items: list[EdgeDelete],
        provenance: ProvenanceRef,
    ) -> int:
        """Async-native delete for canonical edges. Use from async callers."""
        ...

    async def apply_invalidations_async(
        self,
        pot_id: str,
        items: list[InvalidationOp],
        provenance: ProvenanceRef,
    ) -> int:
        """Async-native invalidation stamp. Use from async callers."""
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
