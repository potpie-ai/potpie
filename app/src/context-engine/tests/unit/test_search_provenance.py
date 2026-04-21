"""Search rows include episodic provenance (source_refs, reference_time, episode_uuid)."""

from __future__ import annotations

from unittest.mock import MagicMock

from application.use_cases.query_context import search_pot_context


def test_search_pot_context_maps_provenance_from_edge_attributes() -> None:
    edge = MagicMock()
    edge.uuid = "edge-uuid-1"
    edge.name = "Decision"
    edge.summary = "Use Postgres"
    edge.fact = "Use Postgres"
    edge.source_node_uuid = "a"
    edge.target_node_uuid = "b"
    edge.attributes = {
        "source_refs": ["adr-0042", "pr-1287"],
        "reference_time": "2025-04-10T12:00:00+00:00",
        "episode_uuid": "df605b8d-aaaa-bbbb-cccc-ddddeeeeffff",
        "_context_similarity_score": 0.42,
    }
    edge.created_at = None
    edge.valid_at = None
    edge.invalid_at = None
    edge.episodes = []

    episodic = MagicMock()
    episodic.enabled = True
    episodic.search.return_value = [edge]

    rows = search_pot_context(episodic, "pot-1", "ledger", limit=5)

    assert len(rows) == 1
    r = rows[0]
    assert r["source_refs"] == ["adr-0042", "pr-1287"]
    assert r["reference_time"] == "2025-04-10T12:00:00+00:00"
    assert r["episode_uuid"] == "df605b8d-aaaa-bbbb-cccc-ddddeeeeffff"
    assert r["score"] == 0.42
    episodic.search.assert_called_once()
    call_kw = episodic.search.call_args.kwargs
    assert call_kw.get("episode_uuid") is None


def test_search_pot_context_annotates_conflict_rows() -> None:
    e1 = MagicMock()
    e1.uuid = "edge-a"
    e1.name = "USES_DATA_STORE"
    e1.summary = "Mongo"
    e1.fact = None
    e1.source_node_uuid = "svc"
    e1.target_node_uuid = "mongo"
    e1.attributes = {}
    e1.created_at = None
    e1.valid_at = None
    e1.invalid_at = None

    e2 = MagicMock()
    e2.uuid = "edge-b"
    e2.name = "USES_DATA_STORE"
    e2.summary = "Postgres"
    e2.fact = None
    e2.source_node_uuid = "svc"
    e2.target_node_uuid = "pg"
    e2.attributes = {}
    e2.created_at = None
    e2.valid_at = None
    e2.invalid_at = None

    episodic = MagicMock()
    episodic.enabled = True
    episodic.search.return_value = [e1, e2]
    episodic.list_open_conflicts.return_value = [
        {
            "uuid": "qi-1",
            "edge_a_uuid": "edge-a",
            "edge_b_uuid": "edge-b",
        }
    ]

    rows = search_pot_context(episodic, "pot-1", "ledger", limit=5)
    assert len(rows) == 2
    assert rows[0]["conflict_ids"] == ["qi-1"]
    assert rows[0]["conflict_with_rows"] == [2]
    assert rows[1]["conflict_with_rows"] == [1]
