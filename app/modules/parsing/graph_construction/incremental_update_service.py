"""Incremental knowledge-graph updates.

This module provides building blocks for updating a project's knowledge
graph in response to a partial change set (a subset of files added,
modified or deleted), rather than wiping the entire graph and
reconstructing it from scratch.

Components:

* :class:`FileSnapshot` — a content-hashed view of the files currently
  represented in the graph. Used as the "before" side of differential
  analysis.
* :func:`detect_changes` — computes the set of added / modified /
  deleted files between two snapshots.
* :class:`IncrementalGraphUpdater` — applies a :class:`ChangeSet` to
  the live neo4j-backed graph: removes nodes (and their relationships)
  belonging to deleted or modified files, then writes new / modified
  file subgraphs back in. Untouched files, their nodes, and their
  relationships are preserved verbatim, as are their already-generated
  inferences (which the global content-hash cache returns as cache
  hits on the next inference run).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from app.modules.parsing.graph_construction.code_graph_service import (
    CodeGraphService,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class FileSnapshot:
    """Content-hashed view of a set of files at a point in time.

    ``files`` maps repo-relative file path → SHA-256 hex digest of the
    file contents. Two snapshots are comparable via :func:`detect_changes`.
    """

    files: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_files(cls, files: Dict[str, str]) -> "FileSnapshot":
        return cls(files=dict(files))

    @staticmethod
    def hash_content(content: str | bytes) -> str:
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ChangeSet:
    """Differential between two :class:`FileSnapshot` instances.

    * ``added`` — files present in the new snapshot but not the old one.
    * ``modified`` — files present in both but with a different content hash.
    * ``deleted`` — files present in the old snapshot but not the new one.

    ``affected_files`` returns the union of files whose graph
    representation needs to be removed before the new subgraph is
    inserted (i.e. modified + deleted). ``upserted_files`` returns the
    files whose new subgraph should be inserted (added + modified).
    """

    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)

    @property
    def affected_files(self) -> List[str]:
        return list(self.modified) + list(self.deleted)

    @property
    def upserted_files(self) -> List[str]:
        return list(self.added) + list(self.modified)

    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted)


def detect_changes(old: FileSnapshot, new: FileSnapshot) -> ChangeSet:
    """Compute the :class:`ChangeSet` between two snapshots."""
    old_files = old.files
    new_files = new.files

    old_keys: Set[str] = set(old_files)
    new_keys: Set[str] = set(new_files)

    added = sorted(new_keys - old_keys)
    deleted = sorted(old_keys - new_keys)
    modified = sorted(
        path for path in (old_keys & new_keys) if old_files[path] != new_files[path]
    )

    return ChangeSet(added=added, modified=modified, deleted=deleted)


class IncrementalGraphUpdater:
    """Apply a :class:`ChangeSet` to an existing knowledge graph.

    Wraps a :class:`CodeGraphService` so the same neo4j driver is
    reused for the partial-cleanup and incremental-store operations.
    """

    def __init__(self, code_graph_service: CodeGraphService):
        self.code_graph_service = code_graph_service

    # ------------------------------------------------------------------
    # Snapshot capture
    # ------------------------------------------------------------------
    def get_existing_file_snapshot(self, project_id: str) -> FileSnapshot:
        """Reconstruct a :class:`FileSnapshot` from the live graph.

        Each ``FILE`` node carries a ``file_path`` and ``text`` property
        (the file's source). We hash the text to produce the snapshot
        entry. Files without a ``text`` property fall back to an empty
        hash sentinel so they still participate in deletion detection.
        """
        snapshot: Dict[str, str] = {}
        with self.code_graph_service.driver.session() as session:
            result = session.run(
                """
                MATCH (n:FILE {repoId: $project_id})
                RETURN n.file_path AS file_path,
                       COALESCE(n.text, '') AS text
                """,
                project_id=project_id,
            )
            for record in result:
                path = record["file_path"]
                if not path:
                    continue
                snapshot[path] = FileSnapshot.hash_content(record["text"] or "")
        return FileSnapshot(files=snapshot)

    # ------------------------------------------------------------------
    # Scoped cleanup
    # ------------------------------------------------------------------
    def cleanup_files(self, project_id: str, file_paths: Iterable[str]) -> int:
        """Remove nodes and relationships for the given file paths.

        Matches every ``NODE`` belonging to ``project_id`` whose
        ``file_path`` is in ``file_paths`` and ``DETACH DELETE``s it.
        ``DETACH`` drops the node's incident relationships, so
        relationship integrity for the remaining (untouched) part of
        the graph is preserved automatically: only relationships with
        an endpoint inside the affected file set are removed.

        Returns the number of nodes deleted.
        """
        paths = [p for p in file_paths if p]
        if not paths:
            return 0
        with self.code_graph_service.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {repoId: $project_id})
                WHERE n.file_path IN $paths
                WITH n, count(*) AS _
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                project_id=project_id,
                paths=paths,
            )
            record = result.single()
            deleted = int(record["deleted"]) if record else 0
        logger.info(
            "[INCREMENTAL] Removed %d nodes across %d files for project %s",
            deleted,
            len(paths),
            project_id,
        )
        return deleted

    # ------------------------------------------------------------------
    # Apply a change set
    # ------------------------------------------------------------------
    def apply_changes(
        self,
        project_id: str,
        user_id: str,
        changeset: ChangeSet,
        new_artifacts: Optional[object] = None,
    ) -> Dict[str, int]:
        """Apply ``changeset`` to the live graph.

        * Deletes nodes whose ``file_path`` is in
          ``changeset.affected_files`` (modified + deleted).
        * If ``new_artifacts`` is supplied, reconstructs a subgraph
          from it (using the same reconstruction path as the full
          pipeline) and writes only the nodes/relationships belonging
          to ``changeset.upserted_files`` into the graph, leaving every
          unaffected node and relationship in place.

        Returns a small stats dict for observability.
        """
        if changeset.is_empty():
            logger.info(
                "[INCREMENTAL] No changes detected for project %s — skipping update",
                project_id,
            )
            return {"deleted_nodes": 0, "inserted_nodes": 0, "inserted_edges": 0}

        deleted_nodes = self.cleanup_files(project_id, changeset.affected_files)

        inserted_nodes = 0
        inserted_edges = 0
        if new_artifacts is not None and changeset.upserted_files:
            inserted_nodes, inserted_edges = self._store_subgraph(
                project_id=project_id,
                user_id=user_id,
                new_artifacts=new_artifacts,
                upserted_files=set(changeset.upserted_files),
            )

        logger.info(
            "[INCREMENTAL] Applied changeset for project %s: "
            "+%d nodes, +%d edges, -%d nodes (added=%d modified=%d deleted=%d)",
            project_id,
            inserted_nodes,
            inserted_edges,
            deleted_nodes,
            len(changeset.added),
            len(changeset.modified),
            len(changeset.deleted),
        )
        return {
            "deleted_nodes": deleted_nodes,
            "inserted_nodes": inserted_nodes,
            "inserted_edges": inserted_edges,
        }

    # ------------------------------------------------------------------
    # Internal: scoped subgraph insertion
    # ------------------------------------------------------------------
    def _store_subgraph(
        self,
        *,
        project_id: str,
        user_id: str,
        new_artifacts: object,
        upserted_files: Set[str],
    ) -> tuple[int, int]:
        """Insert only the portion of ``new_artifacts`` whose nodes
        live in ``upserted_files``.

        Reconstructs an ``nx.MultiDiGraph`` from the parser artifacts
        (same helper used by the full-graph path), filters it down to
        the upserted file set, and feeds the filtered graph through
        :meth:`CodeGraphService._store_graph`.
        """
        # Lazy import — keeps RepoMap / tree_sitter off the import path
        # for callers that only need change detection.
        from app.modules.parsing.graph_construction.parsing_repomap import (
            _reconstruct_graph_from_payload,
        )

        nx_graph = _reconstruct_graph_from_payload(new_artifacts)

        # Drop nodes whose file_path is not in the upserted set. Edges
        # touching dropped nodes are removed by networkx automatically
        # when their endpoints disappear.
        to_remove = [
            node_id
            for node_id, data in nx_graph.nodes(data=True)
            if data.get("file") not in upserted_files
        ]
        nx_graph.remove_nodes_from(to_remove)

        node_count = nx_graph.number_of_nodes()
        edge_count = nx_graph.number_of_edges()
        if node_count == 0:
            return 0, 0

        self.code_graph_service._store_graph(nx_graph, project_id, user_id)
        return node_count, edge_count
