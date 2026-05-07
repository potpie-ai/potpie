"""Unit tests for incremental knowledge-graph updates (issue #221).

Covers the three pieces that make the incremental path work:

* ``compute_file_hashes`` — stable per-file fingerprints derived from
  the parser artifacts.
* ``GraphDelta`` / ``IncrementalGraphService._diff`` — bucketing files
  into added / modified / deleted / unchanged.
* ``IncrementalGraphService.store_graph_from_artifacts_incremental`` —
  the orchestration that:
    - falls back to a full rebuild on first run (no prior hashes)
    - is a no-op when nothing changed
    - issues file-scoped deletes for modified/deleted files
    - re-inserts only the dirty subset of nodes/edges
    - re-inserts cross-file edges so inbound references survive

The tests stub the Neo4j session and the Qdrant client so they don't
need infra; the goal is to lock in the contract between the diff and
the DB writes, not to retest neo4j itself.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.modules.parsing.graph_construction.incremental_graph_service import (
    GraphDelta,
    IncrementalGraphService,
    affected_node_ids_for_inference,
    compute_file_hashes,
    reconstruct_subgraph,
)
from sandbox.api.parser_wire import ParseArtifacts


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _node(node_id: str, file: str, node_type: str = "FUNCTION", **kw):
    base = dict(
        id=node_id,
        node_type=node_type,
        file=file,
        line=1,
        end_line=10,
        name=node_id.split(":")[-1],
        class_name=None,
        text=f"# code for {node_id}",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _edge(source_id: str, target_id: str, rel: str = "CALLS", **kw):
    base = dict(
        source_id=source_id,
        target_id=target_id,
        relationship_type=rel,
        ident=None,
        ref_line=None,
        end_ref_line=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _artifacts(nodes, edges) -> ParseArtifacts:
    return ParseArtifacts(nodes=list(nodes), relationships=list(edges))


def _three_file_repo() -> ParseArtifacts:
    """Three files; A.foo calls B.bar; C is standalone."""
    return _artifacts(
        nodes=[
            _node("A.py", "A.py", node_type="FILE"),
            _node("A.py:foo", "A.py"),
            _node("B.py", "B.py", node_type="FILE"),
            _node("B.py:bar", "B.py"),
            _node("C.py", "C.py", node_type="FILE"),
            _node("C.py:baz", "C.py"),
        ],
        edges=[
            _edge("A.py:foo", "B.py:bar"),
            _edge("A.py", "A.py:foo", rel="CONTAINS"),
            _edge("B.py", "B.py:bar", rel="CONTAINS"),
            _edge("C.py", "C.py:baz", rel="CONTAINS"),
        ],
    )


# ---------------------------------------------------------------------------
# Hash + diff building blocks
# ---------------------------------------------------------------------------


class TestComputeFileHashes:
    def test_per_file_hashes_are_stable_and_distinct(self):
        artifacts = _three_file_repo()
        hashes = compute_file_hashes(artifacts)
        assert set(hashes) == {"A.py", "B.py", "C.py"}
        # Re-running should produce identical hashes.
        assert compute_file_hashes(artifacts) == hashes
        # Different files should differ.
        assert hashes["A.py"] != hashes["B.py"] != hashes["C.py"]

    def test_modifying_a_file_changes_only_that_files_hash(self):
        original = _three_file_repo()
        original_hashes = compute_file_hashes(original)

        # Change the body of A.foo (different `text` value).
        modified_nodes = list(original.nodes)
        for i, node in enumerate(modified_nodes):
            if node.id == "A.py:foo":
                modified_nodes[i] = _node(
                    "A.py:foo", "A.py", text="# DIFFERENT body"
                )
        modified = _artifacts(modified_nodes, original.relationships)
        new_hashes = compute_file_hashes(modified)

        assert new_hashes["A.py"] != original_hashes["A.py"]
        assert new_hashes["B.py"] == original_hashes["B.py"]
        assert new_hashes["C.py"] == original_hashes["C.py"]

    def test_edges_originating_in_a_file_count_toward_its_hash(self):
        """If A.foo's call target changes, A's hash changes (A's outbound
        edges contributed to its fingerprint), but B's hash doesn't."""
        before = _three_file_repo()
        before_hashes = compute_file_hashes(before)
        modified_edges = []
        for edge in before.relationships:
            if (
                edge.source_id == "A.py:foo"
                and edge.target_id == "B.py:bar"
            ):
                modified_edges.append(_edge("A.py:foo", "C.py:baz"))
            else:
                modified_edges.append(edge)
        after = _artifacts(before.nodes, modified_edges)
        after_hashes = compute_file_hashes(after)
        assert after_hashes["A.py"] != before_hashes["A.py"]
        assert after_hashes["B.py"] == before_hashes["B.py"]


class TestGraphDeltaDiff:
    def test_diff_buckets(self):
        old = {"A.py": "a1", "B.py": "b1", "C.py": "c1"}
        new = {"A.py": "a1", "B.py": "b2", "D.py": "d1"}
        delta = IncrementalGraphService._diff(old, new)
        assert delta.added == {"D.py"}
        assert delta.modified == {"B.py"}
        assert delta.deleted == {"C.py"}
        assert delta.unchanged == {"A.py"}
        assert delta.has_changes is True
        assert delta.dirty == {"D.py", "B.py"}
        assert delta.removed == {"C.py"}

    def test_no_changes(self):
        delta = IncrementalGraphService._diff(
            {"A.py": "a"}, {"A.py": "a"}
        )
        assert delta.has_changes is False


# ---------------------------------------------------------------------------
# Service orchestration
# ---------------------------------------------------------------------------


class _FakeNeo4jSession:
    """Captures Cypher calls so we can assert on them."""

    def __init__(self, hash_rows=None, total_file_count: int = 0):
        self._hash_rows = hash_rows or []
        self._total_file_count = total_file_count
        self.calls: list[tuple[str, dict]] = []
        # Return value for the read-hashes query — overridden via
        # set_hash_rows when the test wants to simulate prior state.

    def set_hash_rows(self, rows):
        self._hash_rows = rows
        # Keep total file count in sync by default (full coverage).
        self._total_file_count = len(rows)

    def set_total_file_count(self, count: int):
        """Override the FILE node count independently for partial-seeding tests."""
        self._total_file_count = count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeNeo4jResult(self._results_for(query, params))

    def _results_for(self, query, params):
        # The hash-read query is the only one tests inspect the rows of.
        if "RETURN n.file_path AS file_path, n.content_hash AS content_hash" in query:
            return self._hash_rows
        # The total-FILE-count query used by _count_file_nodes.
        if "RETURN count(n) AS total" in query:
            return [{"total": self._total_file_count}]
        return []


class _FakeNeo4jResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _FakeDriver:
    def __init__(self, total_file_count: int = 0):
        self.session_obj = _FakeNeo4jSession(total_file_count=total_file_count)

    def session(self):
        # Return the same session every time so tests can inspect
        # the full call log across the whole orchestration.
        return self.session_obj


def _make_graph_service():
    """Hand-built CodeGraphService stand-in.

    The real class binds Neo4j + Qdrant in __init__. We bypass the
    constructor so we can inject our fakes without spinning up infra.
    """
    from app.modules.parsing.graph_construction.code_graph_service import (
        CodeGraphService,
    )

    svc = CodeGraphService.__new__(CodeGraphService)
    svc.driver = _FakeDriver()
    svc.qdrant_client = MagicMock(name="qdrant_client")
    svc.qdrant_client.collection_exists = MagicMock(return_value=False)
    svc.db = MagicMock(name="db_session")
    return svc


class TestStoreGraphFromArtifactsIncremental:
    def test_first_run_falls_back_to_full_rebuild_and_seeds_hashes(self):
        graph = _make_graph_service()
        graph.store_graph_from_artifacts = MagicMock(name="full_rebuild")
        artifacts = _three_file_repo()

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        # Full rebuild was called exactly once.
        graph.store_graph_from_artifacts.assert_called_once_with(
            artifacts, "p1", "u1"
        )
        # Every file is reported as added (the seed view).
        assert delta.added == {"A.py", "B.py", "C.py"}
        assert delta.modified == set()
        assert delta.deleted == set()
        # Hashes were written: confirm a SET n.content_hash query was issued.
        hash_writes = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "SET n.content_hash" in q
        ]
        assert hash_writes, "expected a content_hash write after full rebuild"

    def test_no_changes_is_a_noop(self):
        graph = _make_graph_service()
        graph.store_graph_from_artifacts = MagicMock(name="full_rebuild")
        artifacts = _three_file_repo()

        # Pre-populate the session with hashes that match what the
        # current artifacts would produce → no changes.
        existing = compute_file_hashes(artifacts)
        graph.driver.session_obj.set_hash_rows(
            [{"file_path": fp, "content_hash": h} for fp, h in existing.items()]
        )

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        assert delta.has_changes is False
        assert delta.unchanged == {"A.py", "B.py", "C.py"}
        # Did NOT trigger a full rebuild and did NOT issue any
        # DETACH DELETE.
        graph.store_graph_from_artifacts.assert_not_called()
        delete_calls = [
            q for q, _ in graph.driver.session_obj.calls if "DETACH DELETE" in q
        ]
        assert not delete_calls

    def test_modified_file_drops_only_that_file(self):
        graph = _make_graph_service()
        graph.store_graph_from_artifacts = MagicMock(name="full_rebuild")

        original = _three_file_repo()
        original_hashes = compute_file_hashes(original)
        graph.driver.session_obj.set_hash_rows(
            [
                {"file_path": fp, "content_hash": h}
                for fp, h in original_hashes.items()
            ]
        )

        # Modify B.py: rewrite B.bar's body.
        modified_nodes = []
        for node in original.nodes:
            if node.id == "B.py:bar":
                modified_nodes.append(
                    _node("B.py:bar", "B.py", text="# rewritten body")
                )
            else:
                modified_nodes.append(node)
        artifacts = _artifacts(modified_nodes, original.relationships)

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        assert delta.modified == {"B.py"}
        assert delta.added == set()
        assert delta.deleted == set()
        assert "A.py" in delta.unchanged
        assert "C.py" in delta.unchanged

        # The DETACH DELETE call should have only B.py in its files list.
        delete_calls = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "DETACH DELETE" in q
        ]
        assert delete_calls, "expected a delete pass for the modified file"
        _, params = delete_calls[0]
        assert set(params["files"]) == {"B.py"}

        # Full rebuild was NOT used.
        graph.store_graph_from_artifacts.assert_not_called()

    def test_added_file_is_inserted_with_no_delete(self):
        graph = _make_graph_service()
        original = _three_file_repo()
        original_hashes = compute_file_hashes(original)
        graph.driver.session_obj.set_hash_rows(
            [
                {"file_path": fp, "content_hash": h}
                for fp, h in original_hashes.items()
            ]
        )

        # Add D.py.
        new_nodes = list(original.nodes) + [
            _node("D.py", "D.py", node_type="FILE"),
            _node("D.py:qux", "D.py"),
        ]
        new_edges = list(original.relationships) + [
            _edge("D.py", "D.py:qux", rel="CONTAINS"),
        ]
        artifacts = _artifacts(new_nodes, new_edges)

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        assert delta.added == {"D.py"}
        assert delta.modified == set()
        assert delta.deleted == set()

        # No DETACH DELETE — additions shouldn't drop existing nodes.
        delete_calls = [
            q for q, _ in graph.driver.session_obj.calls if "DETACH DELETE" in q
        ]
        assert not delete_calls

        # Inserted nodes should include the D.py FILE node.
        insert_calls = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "apoc.create.node" in q
        ]
        assert insert_calls, "expected a node insert for the added file"
        inserted_files = {
            n["file_path"] for q, p in insert_calls for n in p["nodes"]
        }
        assert inserted_files == {"D.py"}

    def test_deleted_file_is_removed_with_no_reinsert(self):
        graph = _make_graph_service()
        original = _three_file_repo()
        original_hashes = compute_file_hashes(original)
        graph.driver.session_obj.set_hash_rows(
            [
                {"file_path": fp, "content_hash": h}
                for fp, h in original_hashes.items()
            ]
        )

        # Drop C.py from the new artifacts.
        kept_nodes = [n for n in original.nodes if n.file != "C.py"]
        kept_edges = [
            e
            for e in original.relationships
            # No edges out of C.py, but the CONTAINS edge from C.py file
            # node → C.py:baz lives in C.py — drop it.
            if not (
                e.source_id.startswith("C.py")
                or e.target_id.startswith("C.py")
            )
        ]
        artifacts = _artifacts(kept_nodes, kept_edges)

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        assert delta.deleted == {"C.py"}
        assert delta.added == set()
        assert delta.modified == set()

        delete_calls = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "DETACH DELETE" in q
        ]
        _, params = delete_calls[0]
        assert "C.py" in params["files"]

        # No insert pass should have included C.py nodes.
        insert_calls = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "apoc.create.node" in q
        ]
        for _, p in insert_calls:
            for node_row in p["nodes"]:
                assert node_row["file_path"] != "C.py"

    def test_cross_file_edge_is_reissued_when_target_file_changes(self):
        """Modify B.py (which contains B.bar). A.foo calls B.bar.

        The incremental path must re-emit the A.foo→B.bar edge so the
        inbound reference survives B.bar being rewritten with a new
        attribute set. Edges between two unchanged files are NOT
        re-emitted (they were never deleted).
        """
        graph = _make_graph_service()
        original = _three_file_repo()
        original_hashes = compute_file_hashes(original)
        graph.driver.session_obj.set_hash_rows(
            [
                {"file_path": fp, "content_hash": h}
                for fp, h in original_hashes.items()
            ]
        )

        modified_nodes = []
        for node in original.nodes:
            if node.id == "B.py:bar":
                modified_nodes.append(
                    _node("B.py:bar", "B.py", line=99, text="# new sig")
                )
            else:
                modified_nodes.append(node)
        artifacts = _artifacts(modified_nodes, original.relationships)

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        # Sanity: only B.py is dirty.
        assert delta.modified == {"B.py"}

        # The MERGE pass must include the cross-file CALLS edge.
        merge_calls = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "MERGE (source)" in q and ":CALLS" in q
        ]
        assert merge_calls, "expected a MERGE for CALLS edges"
        # Every emitted CALLS edge must touch B.py (source or target).
        # Specifically, A.foo → B.bar should be among them.
        from app.modules.parsing.graph_construction.code_graph_service import (
            CodeGraphService,
        )

        expected_source = CodeGraphService.generate_node_id("A.py:foo", "u1")
        expected_target = CodeGraphService.generate_node_id("B.py:bar", "u1")
        all_edge_rows = [
            row for _, p in merge_calls for row in p["edges"]
        ]
        assert any(
            r["source_id"] == expected_source
            and r["target_id"] == expected_target
            for r in all_edge_rows
        ), (
            "A.foo → B.bar should be re-emitted because B.py changed; "
            "got: " + repr(all_edge_rows)
        )

        # CONTAINS edges between two unchanged files should NOT be
        # re-emitted. C.py → C.py:baz was unchanged on both ends.
        contains_calls = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "MERGE (source)" in q and ":CONTAINS" in q
        ]
        c_source = CodeGraphService.generate_node_id("C.py", "u1")
        for _, p in contains_calls:
            for row in p["edges"]:
                assert row["source_id"] != c_source, (
                    "C.py → C.py:baz is between two unchanged files and "
                    "should not be re-emitted"
                )


    def test_partial_seeding_forces_full_rebuild(self):
        """If only some FILE nodes have content_hash, treat project as unseeded.

        Verifies fix for: partial seeding gap where files without hashes
        are treated as "added" → old nodes never deleted → duplicates.
        """
        graph = _make_graph_service()
        graph.store_graph_from_artifacts = MagicMock(name="full_rebuild")
        artifacts = _three_file_repo()

        # Pre-seed only ONE of the three files' hashes.
        existing = compute_file_hashes(artifacts)
        only_a = [{"file_path": "A.py", "content_hash": existing["A.py"]}]
        graph.driver.session_obj.set_hash_rows(only_a)
        # But there are 3 FILE nodes in the graph (partial coverage).
        graph.driver.session_obj.set_total_file_count(3)

        service = IncrementalGraphService(graph)
        delta = service.store_graph_from_artifacts_incremental(
            artifacts, "p1", "u1"
        )

        # Must have fallen back to a full rebuild, not incremental.
        graph.store_graph_from_artifacts.assert_called_once_with(
            artifacts, "p1", "u1"
        )
        # Hashes are seeded for all files.
        hash_writes = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "SET n.content_hash" in q
        ]
        assert hash_writes, "expected a content_hash write after fallback rebuild"

    def test_qdrant_failure_defers_hash_write(self):
        """When Qdrant sync fails, content_hash must NOT be advanced.

        Verifies fix for: Qdrant errors were swallowed but hashes were
        written anyway, so the next parse saw "no changes" and skipped
        the retry.
        """
        graph = _make_graph_service()
        graph.store_graph_from_artifacts = MagicMock(name="full_rebuild")
        original = _three_file_repo()
        original_hashes = compute_file_hashes(original)
        graph.driver.session_obj.set_hash_rows(
            [
                {"file_path": fp, "content_hash": h}
                for fp, h in original_hashes.items()
            ]
        )

        # Modify B.py so there's an actual delta to process.
        modified_nodes = []
        for node in original.nodes:
            if node.id == "B.py:bar":
                modified_nodes.append(
                    _node("B.py:bar", "B.py", text="# changed")
                )
            else:
                modified_nodes.append(node)
        artifacts = _artifacts(modified_nodes, original.relationships)

        service = IncrementalGraphService(graph)
        # Make _update_qdrant fail by having qdrant_client raise on delete.
        graph.qdrant_client.collection_exists = MagicMock(return_value=True)

        with patch(
            "app.modules.parsing.graph_construction.incremental_graph_service"
            ".IncrementalGraphService._update_qdrant",
            return_value=False,
        ):
            delta = service.store_graph_from_artifacts_incremental(
                artifacts, "p1", "u1"
            )

        # Content hash must NOT have been written — retry must be possible.
        hash_writes = [
            (q, p)
            for q, p in graph.driver.session_obj.calls
            if "SET n.content_hash" in q
        ]
        assert not hash_writes, (
            "content_hash must NOT be written when Qdrant sync fails — "
            "next parse must be able to retry the vector update"
        )
        # Delta is still returned so the caller knows what happened.
        assert delta.modified == {"B.py"}


# ---------------------------------------------------------------------------
# Inference-side helper
# ---------------------------------------------------------------------------


class TestInferenceHelpers:
    def test_affected_node_ids_only_returns_dirty_non_file_nodes(self):
        from app.modules.parsing.graph_construction.code_graph_service import (
            CodeGraphService,
        )

        artifacts = _three_file_repo()
        delta = GraphDelta(modified={"B.py"})
        ids = affected_node_ids_for_inference(artifacts, delta, "u1")
        # Only B.py:bar is dirty and non-FILE.
        assert ids == [
            CodeGraphService.generate_node_id("B.py:bar", "u1")
        ]

    def test_affected_node_ids_empty_when_no_changes(self):
        artifacts = _three_file_repo()
        delta = GraphDelta()
        assert affected_node_ids_for_inference(artifacts, delta, "u1") == []

    def test_reconstruct_subgraph_keeps_only_dirty_nodes(self):
        artifacts = _three_file_repo()
        sub = reconstruct_subgraph(artifacts, {"B.py"})
        # Nodes with full attributes only come from B.py.
        nodes_with_type = {
            n for n, attrs in sub.nodes(data=True) if attrs.get("type")
        }
        assert nodes_with_type == {"B.py", "B.py:bar"}
        # Cross-file edge from A.foo → B.bar should be preserved
        # (target file is dirty). nx.add_edge auto-creates the
        # endpoint, so A.foo appears as a placeholder node without
        # attributes — that's fine, the inference layer reads node
        # data only off the attributed nodes.
        assert sub.has_edge("A.py:foo", "B.py:bar")
