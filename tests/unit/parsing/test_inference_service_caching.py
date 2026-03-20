"""
Unit tests for InferenceService hash-based caching of graph nodes.

Covers:
- _lookup_cache_for_nodes: cache hit/miss detection using content hashes
- batch_update_neo4j_with_cached_inference: stores content_hash in Neo4j
- update_neo4j_with_docstrings: stores content_hash via content_hashes parameter
- fetch_graph: returns node_type for proper hash generation
- Cross-branch cache reuse via the global content-hash keyed cache
"""

import pytest
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict, List

from app.modules.parsing.utils.content_hash import generate_content_hash, is_content_cacheable

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Substantial code that passes is_content_cacheable(min_length=100)
LONG_CODE = "\n".join(
    [f"def func_{i}(x, y):\n    '''Return sum.'''\n    return x + y + {i}" for i in range(10)]
)
assert len(LONG_CODE) > 100, "LONG_CODE must be long enough to be cacheable"


def _make_node(node_id: str, text: str, node_type: str = "FUNCTION") -> Dict[str, Any]:
    return {"node_id": node_id, "text": text, "node_type": node_type}


def _make_inference_service():
    """
    Construct a bare InferenceService with all external dependencies mocked.
    Avoids importing Neo4j driver, SentenceTransformer, etc. at construction time.
    """
    with (
        patch("app.modules.parsing.knowledge_graph.inference_service.GraphDatabase"),
        patch("app.modules.parsing.knowledge_graph.inference_service.get_embedding_model"),
        patch("app.modules.parsing.knowledge_graph.inference_service.SearchService"),
        patch("app.modules.parsing.knowledge_graph.inference_service.ProjectService"),
        patch("app.modules.parsing.knowledge_graph.inference_service.ProviderService"),
    ):
        from app.modules.parsing.knowledge_graph.inference_service import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc.db = MagicMock()
        svc.embedding_model = MagicMock()
        svc.embedding_model.encode.return_value = [0.1] * 384
        svc.driver = MagicMock()
        svc.parallel_requests = 5
        svc.project_manager = MagicMock()
        svc.project_manager.get_project_from_db_by_id_sync.return_value = {
            "repo_path": "/some/repo"
        }
        svc.search_service = MagicMock()
        svc.provider_service = MagicMock()
        return svc


# ---------------------------------------------------------------------------
# Tests for _lookup_cache_for_nodes
# ---------------------------------------------------------------------------


class TestLookupCacheForNodes:
    """Test that _lookup_cache_for_nodes correctly marks cache hits and misses."""

    def test_cache_hit_sets_cached_inference(self):
        """A node whose hash exists in the cache gets cached_inference populated."""
        svc = _make_inference_service()
        cache_service = MagicMock()
        cache_service.get_cached_inference.return_value = {
            "docstring": "Does something useful",
            "tags": ["API"],
        }

        nodes = [_make_node("node-1", LONG_CODE)]
        node_dict = {n["node_id"]: n for n in nodes}
        svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj-A")

        assert nodes[0].get("cached_inference") is not None
        assert nodes[0]["cached_inference"]["docstring"] == "Does something useful"
        # should_cache must NOT be set on a cache hit
        assert not nodes[0].get("should_cache")

    def test_cache_miss_sets_should_cache_and_content_hash(self):
        """A node whose hash is absent from cache gets should_cache=True and content_hash set."""
        svc = _make_inference_service()
        cache_service = MagicMock()
        cache_service.get_cached_inference.return_value = None  # miss

        nodes = [_make_node("node-2", LONG_CODE)]
        node_dict = {n["node_id"]: n for n in nodes}
        svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj-A")

        assert nodes[0].get("should_cache") is True
        assert nodes[0].get("content_hash") is not None
        assert len(nodes[0]["content_hash"]) == 64  # SHA-256 hex
        assert nodes[0].get("cached_inference") is None

    def test_node_type_used_in_hash(self):
        """Nodes with different types produce different hashes for the same code."""
        svc = _make_inference_service()

        captured_hashes: List[str] = []

        def capture_lookup(h):
            captured_hashes.append(h)
            return None  # always miss

        cache_service = MagicMock()
        cache_service.get_cached_inference.side_effect = capture_lookup

        func_node = _make_node("n-func", LONG_CODE, "FUNCTION")
        class_node = _make_node("n-class", LONG_CODE, "CLASS")
        nodes = [func_node, class_node]
        node_dict = {n["node_id"]: n for n in nodes}
        svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj-A")

        assert len(captured_hashes) == 2
        assert captured_hashes[0] != captured_hashes[1]

    def test_short_node_text_skipped(self):
        """Nodes with text shorter than the cacheability threshold are not looked up."""
        svc = _make_inference_service()
        cache_service = MagicMock()

        nodes = [_make_node("node-short", "x = 1")]
        node_dict = {n["node_id"]: n for n in nodes}
        svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj-A")

        cache_service.get_cached_inference.assert_not_called()
        assert nodes[0].get("cached_inference") is None
        assert nodes[0].get("should_cache") is None

    def test_no_text_node_skipped(self):
        """Nodes with no text field are silently skipped."""
        svc = _make_inference_service()
        cache_service = MagicMock()

        nodes = [{"node_id": "no-text", "node_type": "FUNCTION"}]
        node_dict = {n["node_id"]: n for n in nodes}
        svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj-A")

        cache_service.get_cached_inference.assert_not_called()

    def test_cross_branch_cache_hit(self):
        """
        The same code appearing in a different branch/project gets a cache hit
        because the cache is keyed globally by content hash.
        """
        svc = _make_inference_service()

        expected_hash = generate_content_hash(LONG_CODE, "FUNCTION")
        cache_service = MagicMock()
        cache_service.get_cached_inference.side_effect = lambda h: (
            {"docstring": "Cached from branch A", "tags": []} if h == expected_hash else None
        )

        # Node from "branch B" — different project_id, same code
        node = _make_node("nodeB", LONG_CODE, "FUNCTION")
        nodes = [node]
        node_dict = {n["node_id"]: n for n in nodes}
        svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj-branch-B")

        assert nodes[0].get("cached_inference") is not None
        assert nodes[0]["cached_inference"]["docstring"] == "Cached from branch A"

    def test_returns_stats_dict(self):
        """_lookup_cache_for_nodes returns a stats dict with hit/miss counts."""
        svc = _make_inference_service()
        cache_service = MagicMock()
        cache_service.get_cached_inference.return_value = None

        nodes = [_make_node(f"n{i}", LONG_CODE) for i in range(3)]
        node_dict = {n["node_id"]: n for n in nodes}
        stats = svc._lookup_cache_for_nodes(nodes, node_dict, cache_service, "proj")

        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 3


# ---------------------------------------------------------------------------
# Tests for batch_update_neo4j_with_cached_inference
# ---------------------------------------------------------------------------


class TestBatchUpdateNeo4jWithCachedInference:
    """Test that batch_update_neo4j_with_cached_inference writes content_hash to Neo4j."""

    def _run_batch_update(self, nodes, repo_id="proj-1"):
        svc = _make_inference_service()

        session_ctx = MagicMock()
        session_ctx.__enter__ = MagicMock(return_value=session_ctx)
        session_ctx.__exit__ = MagicMock(return_value=False)
        svc.driver.session.return_value = session_ctx

        count = svc.batch_update_neo4j_with_cached_inference(nodes, repo_id)
        return count, session_ctx

    def test_stores_content_hash_in_neo4j(self):
        """content_hash from the node dict is included in the Cypher SET clause."""
        content_hash = generate_content_hash(LONG_CODE, "FUNCTION")
        nodes = [
            {
                "node_id": "node-x",
                "content_hash": content_hash,
                "cached_inference": {
                    "docstring": "Does things",
                    "tags": ["API"],
                    "embedding_vector": [0.1] * 384,
                },
            }
        ]
        count, session_ctx = self._run_batch_update(nodes)

        assert count == 1
        call_args = session_ctx.run.call_args
        query = call_args[0][0]
        assert "content_hash" in query

        # Verify content_hash value is passed in batch data
        batch_param = call_args[1]["batch"]
        assert batch_param[0]["content_hash"] == content_hash

    def test_empty_nodes_returns_zero(self):
        """Returns 0 immediately when passed an empty list."""
        count, session_ctx = self._run_batch_update([])
        assert count == 0
        session_ctx.run.assert_not_called()

    def test_node_without_cached_inference_skipped(self):
        """Nodes lacking cached_inference are silently skipped."""
        nodes = [{"node_id": "bare-node", "content_hash": "abc"}]
        count, session_ctx = self._run_batch_update(nodes)
        assert count == 0


# ---------------------------------------------------------------------------
# Tests for update_neo4j_with_docstrings
# ---------------------------------------------------------------------------


class TestUpdateNeo4jWithDocstrings:
    """Test that update_neo4j_with_docstrings passes content_hash to Neo4j."""

    def test_content_hash_included_in_query(self):
        """When content_hashes is provided, content_hash appears in the SET clause."""
        from app.modules.parsing.knowledge_graph.inference_schema import (
            DocstringResponse,
            DocstringNode,
        )

        svc = _make_inference_service()

        session_ctx = MagicMock()
        session_ctx.__enter__ = MagicMock(return_value=session_ctx)
        session_ctx.__exit__ = MagicMock(return_value=False)
        svc.driver.session.return_value = session_ctx

        docstrings = DocstringResponse(
            docstrings=[DocstringNode(node_id="n1", docstring="Useful function", tags=[])]
        )
        content_hashes = {"n1": generate_content_hash(LONG_CODE, "FUNCTION")}

        svc.update_neo4j_with_docstrings(
            "proj-1", docstrings, content_hashes=content_hashes
        )

        call_args = session_ctx.run.call_args
        query = call_args[0][0]
        assert "content_hash" in query

        batch_param = call_args[1]["batch"]
        assert batch_param[0]["content_hash"] == content_hashes["n1"]

    def test_no_content_hashes_still_sets_field(self):
        """Even without content_hashes, the query still has content_hash (set to None)."""
        from app.modules.parsing.knowledge_graph.inference_schema import (
            DocstringResponse,
            DocstringNode,
        )

        svc = _make_inference_service()

        session_ctx = MagicMock()
        session_ctx.__enter__ = MagicMock(return_value=session_ctx)
        session_ctx.__exit__ = MagicMock(return_value=False)
        svc.driver.session.return_value = session_ctx

        docstrings = DocstringResponse(
            docstrings=[DocstringNode(node_id="n2", docstring="Helper", tags=[])]
        )

        svc.update_neo4j_with_docstrings("proj-1", docstrings)

        call_args = session_ctx.run.call_args
        query = call_args[0][0]
        assert "content_hash" in query

        batch_param = call_args[1]["batch"]
        # content_hash should be None (not provided)
        assert batch_param[0]["content_hash"] is None


# ---------------------------------------------------------------------------
# Tests for fetch_graph including node_type
# ---------------------------------------------------------------------------


class TestFetchGraphIncludesNodeType:
    """Verify fetch_graph returns node_type so hashes include the type dimension."""

    def test_fetch_graph_query_requests_node_type(self):
        """The Cypher query in fetch_graph must request n.type AS node_type."""
        svc = _make_inference_service()

        session_ctx = MagicMock()
        session_ctx.__enter__ = MagicMock(return_value=session_ctx)
        session_ctx.__exit__ = MagicMock(return_value=False)
        # Return one record then empty to stop pagination
        session_ctx.run.side_effect = [
            [{"node_id": "n1", "text": LONG_CODE, "file_path": "f.py",
              "start_line": 1, "end_line": 10, "name": "foo",
              "node_type": "FUNCTION", "content_hash": None}],
            [],
        ]
        svc.driver.session.return_value = session_ctx

        nodes = svc.fetch_graph("proj-1")

        # Check first run call includes node_type in the query
        first_call_query = session_ctx.run.call_args_list[0][0][0]
        assert "node_type" in first_call_query

        # Verify the returned node has node_type populated
        assert nodes[0]["node_type"] == "FUNCTION"
