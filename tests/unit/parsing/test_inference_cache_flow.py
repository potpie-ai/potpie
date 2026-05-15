"""Unit coverage for inference cache flow behavior."""

import pytest

from app.modules.parsing.knowledge_graph.inference_schema import (
    DocstringNode,
    DocstringResponse,
)
from app.modules.parsing.knowledge_graph.inference_service import InferenceService


pytestmark = pytest.mark.unit


def _service_without_init(monkeypatch):
    service = object.__new__(InferenceService)
    monkeypatch.setattr(
        service,
        "num_tokens_from_string",
        lambda text, model="gpt-4": len(text or ""),
    )
    monkeypatch.setattr(
        service,
        "_normalize_node_text",
        lambda text, node_dict: text or "",
    )
    return service


def test_uncacheable_nodes_still_enter_llm_batches(monkeypatch):
    service = _service_without_init(monkeypatch)
    nodes = [
        {
            "node_id": "small-node",
            "text": "short",
            "normalized_text": "short",
        },
        {
            "node_id": "cache-hit",
            "text": "def cached(): pass",
            "cached_inference": {"docstring": "cached", "tags": []},
        },
        {
            "node_id": "cache-miss",
            "text": "def uncached():\n    return 'x'",
            "normalized_text": "def uncached():\n    return 'x'",
            "should_cache": True,
            "content_hash": "abc123",
            "node_type": "FUNCTION",
        },
    ]

    batches = service._create_batches_from_nodes(
        nodes,
        max_tokens=1000,
        project_id="project-1",
    )

    requests = [request for batch in batches for request in batch]
    assert [request.node_id for request in requests] == ["small-node", "cache-miss"]
    assert requests[0].metadata["should_cache"] is False
    assert requests[0].metadata["content_hash"] is None
    assert requests[1].metadata["should_cache"] is True
    assert requests[1].metadata["content_hash"] == "abc123"


def test_update_neo4j_persists_content_hash(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def run(self, query, **kwargs):
            self.calls.append((query, kwargs))

    class FakeDriver:
        def __init__(self):
            self.session_obj = FakeSession()

        def session(self):
            return self.session_obj

    class FakeProjectManager:
        def get_project_from_db_by_id_sync(self, repo_id):
            return {"repo_path": "/tmp/repo"}

    service = object.__new__(InferenceService)
    fake_driver = FakeDriver()
    service.driver = fake_driver
    service.project_manager = FakeProjectManager()
    monkeypatch.setattr(service, "generate_embedding", lambda text: [0.0, 1.0])

    service.update_neo4j_with_docstrings(
        "project-1",
        DocstringResponse(
            docstrings=[
                DocstringNode(
                    node_id="node-1",
                    docstring="Explains node one.",
                    tags=["UTILITY"],
                )
            ]
        ),
        precomputed_embeddings={"node-1": [0.1, 0.2]},
        content_hash_by_node_id={"node-1": "hash-1"},
    )

    query, kwargs = fake_driver.session_obj.calls[0]
    assert "n.content_hash = item.content_hash" in query
    assert kwargs["batch"][0]["content_hash"] == "hash-1"
