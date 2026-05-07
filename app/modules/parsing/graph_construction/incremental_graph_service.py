"""Incremental Knowledge Graph updates (issue #221).

Given a fresh :class:`ParseArtifacts` from the in-sandbox parser, this
service applies only the *delta* against the project's existing graph
in Neo4j, instead of the full clean-and-rebuild path used by
:class:`CodeGraphService.store_graph_from_artifacts`.

Pipeline
--------

1. Compute a per-file content fingerprint from the incoming artifacts
   (deterministic SHA-1 over the sorted (node_id, line, end_line, text)
   tuples that belong to each file, plus edges originating in that file).
2. Read the previously-persisted fingerprints off the existing FILE
   nodes in Neo4j (``FILE.content_hash``). On a project that has never
   been parsed incrementally, this returns an empty mapping and the
   caller falls back to a full rebuild — the full rebuild also writes
   hashes, so the next run is incremental.
3. Diff the two maps into ``added`` / ``modified`` / ``deleted`` /
   ``unchanged`` buckets.
4. Apply the delta:
   * For every *modified* and *deleted* file, ``DETACH DELETE`` all
     nodes whose ``file_path`` matches that file. This drops every
     edge attached to those nodes (inbound *and* outbound) in one hop.
   * For every *added* and *modified* file, insert the fresh nodes
     for that file from the artifacts.
   * Re-insert every edge from the artifacts whose source *or* target
     file is in the changed set. Edges between two unchanged files
     are left untouched (they were never deleted in step 4a).
5. Update / write ``content_hash`` on each affected FILE node so the
   next run can diff against this build.

Inference is run *only* over the affected node IDs — see
:meth:`InferenceService.run_incremental_inference`.

Why diff at the file level?
---------------------------

* Node-level diffs would need stable cross-version node IDs that
  survive a function being moved within its file; file-level granularity
  sidesteps that and matches how source-control changes actually arrive.
* Edge integrity reduces to a single rule: an edge is dirty iff its
  source-file or target-file is dirty. The parser already emits the
  full edge set on every run, so we just filter rather than re-resolve.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import networkx as nx

from app.modules.parsing.graph_construction.code_graph_service import (
    CodeGraphService,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


# Node types we route into Qdrant; mirrors INDEXABLE_NODE_TYPES in
# qdrant_indexing_service so we can update the Qdrant collection
# incrementally without re-importing the constant in every caller.
_INDEXABLE_NODE_TYPES = {"CLASS", "FUNCTION", "INTERFACE"}

# Node-type → Neo4j label mapping. Mirrors the assembly logic in
# CodeGraphService._store_graph so an incrementally-inserted node
# carries the same label set a full-rebuild node would.
_TYPED_LABELS = {"FILE", "CLASS", "FUNCTION", "INTERFACE"}


@dataclass(slots=True)
class GraphDelta:
    """Result of diffing new artifacts against the persisted graph."""

    added: set[str] = field(default_factory=set)
    modified: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)

    @property
    def dirty(self) -> set[str]:
        """Files whose nodes/edges need to be (re)written."""
        return self.added | self.modified

    @property
    def removed(self) -> set[str]:
        """Files whose nodes need to be deleted with no replacement."""
        return self.deleted

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
        }


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def compute_file_hashes(artifacts) -> dict[str, str]:
    """Compute a per-file SHA-1 fingerprint from the parser artifacts.

    The fingerprint is over a deterministic projection of every node
    attached to the file plus every edge originating in it. We hash
    line numbers and node text rather than re-reading source from disk
    because the artifacts are the canonical view the host process sees
    — anything not represented there can't affect the resulting graph.
    """
    by_file_nodes: dict[str, list[tuple]] = defaultdict(list)
    by_file_edges: dict[str, list[tuple]] = defaultdict(list)

    # Build a node_id -> file lookup so we can attribute edges to the
    # source node's file (edges aren't directly tagged with a file).
    node_to_file: dict[str, str] = {}
    for node in artifacts.nodes:
        file_path = getattr(node, "file", "") or ""
        node_to_file[node.id] = file_path
        by_file_nodes[file_path].append(
            (
                node.id,
                getattr(node, "node_type", "UNKNOWN"),
                getattr(node, "line", -1),
                getattr(node, "end_line", -1),
                getattr(node, "name", "") or "",
                getattr(node, "text", "") or "",
            )
        )

    edges = getattr(artifacts, "relationships", None)
    if edges is None:
        edges = getattr(artifacts, "edges", [])
    for edge in edges:
        source_id = getattr(edge, "source_id")
        target_id = getattr(edge, "target_id")
        rel_type = getattr(edge, "relationship_type", None) or getattr(
            edge, "edge_type", "REFERENCES"
        )
        file_path = node_to_file.get(source_id, "")
        by_file_edges[file_path].append(
            (
                source_id,
                target_id,
                rel_type,
                getattr(edge, "ident", "") or "",
                getattr(edge, "ref_line", -1),
                getattr(edge, "end_ref_line", -1),
            )
        )

    hashes: dict[str, str] = {}
    all_files = set(by_file_nodes) | set(by_file_edges)
    for file_path in all_files:
        if not file_path:
            continue
        h = hashlib.sha1(usedforsecurity=False)  # noqa: S324 — content-change fingerprint, not a security hash
        for tup in sorted(by_file_nodes.get(file_path, [])):
            h.update(repr(tup).encode("utf-8"))
            h.update(b"\x00")
        h.update(b"|edges|")
        for tup in sorted(by_file_edges.get(file_path, [])):
            h.update(repr(tup).encode("utf-8"))
            h.update(b"\x00")
        hashes[file_path] = h.hexdigest()
    return hashes


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class IncrementalGraphService:
    """Apply file-level deltas to the persisted graph.

    Holds a reference to the underlying :class:`CodeGraphService` rather
    than subclassing it: the orchestrator (``ParsingService``) already
    constructs a ``CodeGraphService`` for cleanup / writes and we don't
    want a second Neo4j driver per parse.
    """

    def __init__(self, graph_service: CodeGraphService):
        self.graph_service = graph_service

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def store_graph_from_artifacts_incremental(
        self,
        artifacts,
        project_id: str,
        user_id: str,
    ) -> GraphDelta:
        """Apply ``artifacts`` to the project's existing graph as a delta.

        Returns the :class:`GraphDelta` so the caller can decide what to
        do with downstream pipelines (skip inference if no changes,
        run partial inference on dirty files, etc.).

        If the project has no persisted ``content_hash`` data yet (first
        incremental run on a previously full-rebuilt or empty project),
        we fall through to a full rebuild via
        :meth:`CodeGraphService.store_graph_from_artifacts`. The full
        rebuild now also writes hashes, so subsequent runs are
        incremental.
        """
        existing_hashes = self._read_existing_file_hashes(project_id)
        new_hashes = compute_file_hashes(artifacts)

        # Fall back to a full rebuild when hashes are absent *or* when
        # coverage is partial (some FILE nodes lack content_hash). A
        # partial-seeding gap means files without hashes are treated as
        # "added", so their old nodes are never deleted and
        # apoc.create.node() will duplicate them on the next incremental
        # run. Comparing against the total FILE count in the graph lets
        # us detect this case and force a clean slate.
        total_file_count = self._count_file_nodes(project_id)
        is_unseeded = (
            not existing_hashes
            or len(existing_hashes) != total_file_count
        )
        if is_unseeded:
            logger.info(
                "[INCREMENTAL] Incomplete hash coverage for project %s "
                "(hashed=%d, total_files=%d) — running full rebuild and "
                "seeding hashes",
                project_id,
                len(existing_hashes),
                total_file_count,
            )
            self.graph_service.store_graph_from_artifacts(
                artifacts, project_id, user_id
            )
            self._write_file_hashes(project_id, new_hashes)
            return GraphDelta(added=set(new_hashes))

        delta = self._diff(existing_hashes, new_hashes)
        logger.info(
            "[INCREMENTAL] Delta for project %s: %s",
            project_id,
            delta.summary(),
        )

        if not delta.has_changes:
            logger.info(
                "[INCREMENTAL] No file-level changes — graph untouched",
                project_id=project_id,
            )
            return delta

        start = time.time()
        qdrant_ok = self._apply_delta(
            artifacts=artifacts,
            project_id=project_id,
            user_id=user_id,
            delta=delta,
        )
        if qdrant_ok:
            # Persist hashes for the next run. We update *all* files'
            # hashes — including unchanged ones — so a corrupted hash on
            # any file (e.g. from an aborted previous run) self-heals.
            self._write_file_hashes(project_id, new_hashes)
        else:
            # Qdrant sync failed. Skip writing hashes so the next parse
            # sees the same delta and retries the vector update. The
            # Neo4j graph is already consistent — only the search index
            # is stale until the retry succeeds.
            logger.warning(
                "[INCREMENTAL] Skipping hash write for project %s because "
                "Qdrant sync failed — will retry on next parse",
                project_id,
            )

        logger.info(
            "[INCREMENTAL] Applied delta in %.2fs",
            time.time() - start,
            project_id=project_id,
        )
        return delta

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    @staticmethod
    def _diff(
        old: Mapping[str, str], new: Mapping[str, str]
    ) -> GraphDelta:
        delta = GraphDelta()
        for file_path, h in new.items():
            if file_path not in old:
                delta.added.add(file_path)
            elif old[file_path] != h:
                delta.modified.add(file_path)
            else:
                delta.unchanged.add(file_path)
        for file_path in old:
            if file_path not in new:
                delta.deleted.add(file_path)
        return delta

    # ------------------------------------------------------------------
    # Neo4j operations
    # ------------------------------------------------------------------

    def _count_file_nodes(self, project_id: str) -> int:
        """Return the total number of FILE nodes for *project_id* in Neo4j.

        Used to detect partial hash-seeding: if the count of FILE nodes
        with a ``content_hash`` differs from the total FILE count, some
        files were never hashed and the incremental path must not be used
        (it would treat those files as "added" rather than "unchanged",
        creating duplicates via apoc.create.node()).
        """
        with self.graph_service.driver.session() as session:
            result = session.run(
                """
                MATCH (n:FILE {repoId: $project_id})
                RETURN count(n) AS total
                """,
                project_id=project_id,
            )
            record = result.single()
            return int(record["total"]) if record else 0

    def _read_existing_file_hashes(self, project_id: str) -> dict[str, str]:
        with self.graph_service.driver.session() as session:
            result = session.run(
                """
                MATCH (n:FILE {repoId: $project_id})
                WHERE n.content_hash IS NOT NULL
                RETURN n.file_path AS file_path, n.content_hash AS content_hash
                """,
                project_id=project_id,
            )
            return {
                record["file_path"]: record["content_hash"]
                for record in result
                if record["file_path"]
            }

    def _write_file_hashes(
        self, project_id: str, hashes: Mapping[str, str]
    ) -> None:
        if not hashes:
            return
        rows = [
            {"file_path": fp, "content_hash": h}
            for fp, h in hashes.items()
        ]
        with self.graph_service.driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (n:FILE {repoId: $project_id, file_path: row.file_path})
                SET n.content_hash = row.content_hash
                """,
                project_id=project_id,
                rows=rows,
            )

    def _delete_files(
        self, project_id: str, files: Iterable[str]
    ) -> int:
        """``DETACH DELETE`` every node whose ``file_path`` is in ``files``.

        ``DETACH DELETE`` removes inbound edges as well as outbound, so
        we don't need a separate edge cleanup pass — that's the whole
        reason file-level granularity buys us correctness here.
        """
        files = list(files)
        if not files:
            return 0
        with self.graph_service.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {repoId: $project_id})
                WHERE n.file_path IN $files
                WITH n, count(n) AS _placeholder
                DETACH DELETE n
                RETURN count(_placeholder) AS deleted
                """,
                project_id=project_id,
                files=files,
            )
            record = result.single()
            return int(record["deleted"]) if record else 0

    def _insert_delta_nodes_and_edges(
        self,
        artifacts,
        project_id: str,
        user_id: str,
        dirty_files: set[str],
    ) -> None:
        """Insert the artifacts' nodes/edges that touch ``dirty_files``.

        Reuses the same node-shape and label assembly logic as
        :meth:`CodeGraphService._store_graph` so an incrementally-
        written node is indistinguishable from one written by the full
        rebuild path. Re-implementing rather than calling ``_store_graph``
        because we need the *subset* of the artifacts that intersects
        ``dirty_files``.
        """
        if not dirty_files:
            return

        # Bucket nodes by file so we can filter with one pass through
        # the artifacts.
        nodes_to_insert: list[dict] = []
        node_id_to_file: dict[str, str] = {}
        for node in artifacts.nodes:
            file_path = getattr(node, "file", "") or ""
            node_id_to_file[node.id] = file_path
            if file_path not in dirty_files:
                continue
            node_type = getattr(node, "node_type", "UNKNOWN")
            if node_type == "UNKNOWN":
                continue
            labels = ["NODE"]
            if node_type in _TYPED_LABELS:
                labels.append(node_type)
            processed = {
                "name": getattr(node, "name", None) or node.id,
                "file_path": file_path,
                "start_line": getattr(node, "line", -1),
                "end_line": getattr(node, "end_line", -1),
                "repoId": project_id,
                "node_id": CodeGraphService.generate_node_id(node.id, user_id),
                "entityId": user_id,
                "type": node_type,
                "text": getattr(node, "text", "") or "",
                "labels": labels,
            }
            processed = {k: v for k, v in processed.items() if v is not None}
            nodes_to_insert.append(processed)

        with self.graph_service.driver.session() as session:
            session.run(
                """
                CREATE INDEX node_id_repo_idx IF NOT EXISTS
                FOR (n:NODE) ON (n.node_id, n.repoId)
                """
            )
            if nodes_to_insert:
                session.run(
                    """
                    UNWIND $nodes AS node
                    CALL apoc.create.node(node.labels, node) YIELD node AS n
                    RETURN count(*)
                    """,
                    nodes=nodes_to_insert,
                )

            # Edges: include any edge whose source or target lives in a
            # dirty file. Edges between two unchanged files were never
            # deleted, so we skip them to avoid duplicates.
            edges = getattr(artifacts, "relationships", None)
            if edges is None:
                edges = getattr(artifacts, "edges", [])
            edges_by_type: dict[str, list[dict]] = defaultdict(list)
            for edge in edges:
                source_id = getattr(edge, "source_id")
                target_id = getattr(edge, "target_id")
                source_file = node_id_to_file.get(source_id, "")
                target_file = node_id_to_file.get(target_id, "")
                if (
                    source_file not in dirty_files
                    and target_file not in dirty_files
                ):
                    continue
                rel_type = getattr(edge, "relationship_type", None) or getattr(
                    edge, "edge_type", "REFERENCES"
                )
                edges_by_type[rel_type].append(
                    {
                        "source_id": CodeGraphService.generate_node_id(
                            source_id, user_id
                        ),
                        "target_id": CodeGraphService.generate_node_id(
                            target_id, user_id
                        ),
                        "repoId": project_id,
                    }
                )

            for rel_type, edge_rows in edges_by_type.items():
                # MERGE the edge so we don't double-insert when an
                # unchanged file's outbound edge happens to also be
                # in the artifacts (parser emits the full set every run).
                query = f"""
                    UNWIND $edges AS edge
                    MATCH (source:NODE {{node_id: edge.source_id, repoId: edge.repoId}})
                    MATCH (target:NODE {{node_id: edge.target_id, repoId: edge.repoId}})
                    MERGE (source)-[r:{rel_type} {{repoId: edge.repoId}}]->(target)
                """
                session.run(query, edges=edge_rows)

    def _apply_delta(
        self,
        artifacts,
        project_id: str,
        user_id: str,
        delta: GraphDelta,
    ) -> bool:
        """Apply the delta to Neo4j and Qdrant.

        Returns ``True`` when Qdrant sync completed without error, ``False``
        when it failed (Neo4j is always updated regardless).  The return
        value lets the caller decide whether to advance ``content_hash``.
        """
        # Step 1: drop nodes (and their edges) for deleted + modified files.
        files_to_delete = delta.removed | delta.modified
        deleted = self._delete_files(project_id, files_to_delete)
        logger.info(
            "[INCREMENTAL] Deleted %d nodes for %d files",
            deleted,
            len(files_to_delete),
            project_id=project_id,
        )

        # Step 2: insert nodes / edges for added + modified files. Edges
        # whose other endpoint is unchanged but whose endpoint here is
        # dirty are also (re)inserted so cross-file references hold.
        self._insert_delta_nodes_and_edges(
            artifacts=artifacts,
            project_id=project_id,
            user_id=user_id,
            dirty_files=delta.dirty,
        )

        # Step 3: keep Qdrant in sync with the graph. Drop points for
        # nodes we removed; index the new dirty-file nodes.
        return self._update_qdrant(
            artifacts=artifacts,
            project_id=project_id,
            user_id=user_id,
            delta=delta,
        )

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------

    def _update_qdrant(
        self,
        artifacts,
        project_id: str,
        user_id: str,
        delta: GraphDelta,
    ) -> bool:
        """Apply the delta to the project's Qdrant collection.

        Returns ``True`` on success, ``False`` on failure.  Failures are
        logged but do not roll back Neo4j writes.  The return value is
        threaded back to ``store_graph_from_artifacts_incremental`` so it
        can decide whether to advance ``content_hash``: if we skip the
        hash write on failure, the next parse will see the same delta and
        retry the vector update automatically.
        """
        try:
            from app.modules.parsing.knowledge_graph.qdrant_indexing_service import (  # noqa: E501
                delete_points_by_node_ids,
                index_nodes_to_qdrant,
            )
        except ImportError:
            # The Qdrant helper is not present in this build.  We cannot
            # sync the vector index at all, so we must signal failure
            # (return False) to prevent content_hash from advancing.
            # Advancing the hash would permanently hide the stale search
            # state; returning False ensures the caller skips the hash
            # write and retries on the next parse.
            logger.warning(
                "[INCREMENTAL] qdrant helper missing; "
                "vector sync skipped — hash write deferred",
                project_id=project_id,
            )
            return False

        collection_alias = self.graph_service.get_qdrant_collection_alias(
            project_id
        )
        # After a staged full-rebuild, the alias points at the new
        # staging collection and the original base collection name has
        # been deleted. Always target the alias so incremental writes
        # land in whatever collection currently holds the active
        # index. Qdrant routes alias names transparently for both
        # delete and upsert operations.
        active_collection_name = collection_alias

        # Delete vector points for nodes we removed (deleted + modified
        # files). We re-derive the node_ids from the *old* graph state
        # is impossible at this point — they're already gone from Neo4j.
        # Instead we delete by file_path filter, which Qdrant supports
        # via a payload index. The collection stores `file` payload,
        # added at index time below.
        files_to_drop = delta.removed | delta.modified
        if files_to_drop:
            try:
                delete_points_by_node_ids(
                    self.graph_service.qdrant_client,
                    active_collection_name,
                    file_paths=list(files_to_drop),
                )
            except Exception:
                logger.exception(
                    "[INCREMENTAL] Qdrant delete failed — hash write "
                    "deferred to allow retry on next parse",
                    project_id=project_id,
                )
                return False

        # Index new/modified nodes into Qdrant.
        nodes_to_index = []
        for node in artifacts.nodes:
            file_path = getattr(node, "file", "") or ""
            if file_path not in delta.dirty:
                continue
            if getattr(node, "node_type", "") not in _INDEXABLE_NODE_TYPES:
                continue
            nodes_to_index.append(
                {
                    "node_id": CodeGraphService.generate_node_id(
                        node.id, user_id
                    ),
                    "name": getattr(node, "name", "") or "",
                    "file": file_path,
                    "type": getattr(node, "node_type", ""),
                    "line": getattr(node, "line", -1),
                    "end_line": getattr(node, "end_line", -1),
                    "text": getattr(node, "text", "") or "",
                }
            )
        if not nodes_to_index:
            return True
        try:
            index_nodes_to_qdrant(
                self.graph_service.qdrant_client,
                # Target the alias, not the (possibly deleted) base
                # collection name; same reasoning as the delete path.
                active_collection_name,
                nodes_to_index,
                # Don't recreate — we want to add to the existing
                # collection. The full-rebuild path passes
                # recreate_collection=True.
                recreate_collection=False,
                alias_name=collection_alias,
                # Preserve the persisted BM25 token vocabulary built
                # during the full rebuild instead of overwriting it
                # with one derived from only the dirty nodes. This
                # keeps sparse term indices stable across older,
                # already-indexed points.
                preserve_bm25_vocabulary=True,
            )
        except Exception:
            logger.exception(
                "[INCREMENTAL] Qdrant indexing failed — hash write "
                "deferred to allow retry on next parse",
                project_id=project_id,
            )
            return False
        return True


# ---------------------------------------------------------------------------
# Helpers exposed for the orchestrator
# ---------------------------------------------------------------------------


def affected_node_ids_for_inference(
    artifacts, delta: GraphDelta, user_id: str
) -> list[str]:
    """Return the persisted node IDs that need (re-)inference after
    applying ``delta``.

    Used by :class:`InferenceService` to scope docstring regeneration to
    just the changed components instead of the whole graph.
    """
    affected: list[str] = []
    dirty = delta.dirty
    if not dirty:
        return affected
    for node in artifacts.nodes:
        file_path = getattr(node, "file", "") or ""
        if file_path not in dirty:
            continue
        node_type = getattr(node, "node_type", "UNKNOWN")
        if node_type == "UNKNOWN" or node_type == "FILE":
            continue
        affected.append(CodeGraphService.generate_node_id(node.id, user_id))
    return affected


def reconstruct_subgraph(
    artifacts, dirty_files: set[str]
) -> nx.MultiDiGraph:
    """Build an nx graph over just the dirty files' nodes/edges.

    Used by the inference layer for the partial pass: we want the same
    nx shape the full pipeline consumes, but only over the slice the
    delta covers.

    Re-implements the small ``build_node_attrs`` / ``build_edge_attrs``
    bodies from :mod:`parsing_repomap` rather than importing them, so
    this helper stays light enough to be used from contexts that don't
    want to drag in tree_sitter / grep_ast just to build a subgraph.
    """
    graph = nx.MultiDiGraph()
    node_id_to_file: dict[str, str] = {}
    for node in artifacts.nodes:
        file_path = getattr(node, "file", "") or ""
        node_id_to_file[node.id] = file_path
        if file_path not in dirty_files:
            continue
        node_attrs = {
            "file": file_path,
            "line": getattr(node, "line", -1),
            "end_line": getattr(node, "end_line", -1),
            "type": getattr(node, "node_type", "UNKNOWN"),
            "name": getattr(node, "name", "") or "",
        }
        if node_attrs["type"] != "FILE":
            node_attrs["class_name"] = getattr(node, "class_name", None)
        text = getattr(node, "text", None)
        if text is not None:
            node_attrs["text"] = text
        graph.add_node(node.id, **node_attrs)

    edges = getattr(artifacts, "relationships", None)
    if edges is None:
        edges = getattr(artifacts, "edges", [])
    for edge in edges:
        source_file = node_id_to_file.get(edge.source_id, "")
        target_file = node_id_to_file.get(edge.target_id, "")
        if source_file not in dirty_files and target_file not in dirty_files:
            continue
        edge_type = getattr(edge, "edge_type", None) or getattr(
            edge, "relationship_type", "REFERENCES"
        )
        edge_attrs = {"type": edge_type}
        ident = getattr(edge, "ident", None)
        if ident is not None:
            edge_attrs["ident"] = ident
        ref_line = getattr(edge, "ref_line", None)
        if ref_line is not None:
            edge_attrs["ref_line"] = ref_line
        end_ref_line = getattr(edge, "end_ref_line", None)
        if end_ref_line is not None:
            edge_attrs["end_ref_line"] = end_ref_line
        graph.add_edge(edge.source_id, edge.target_id, **edge_attrs)
    return graph
