from unittest.mock import MagicMock, patch

import networkx as nx


def test_graph_generation_logs_four_numbered_phases():
    """Phase labels are emitted at runtime (not just present as source text)."""
    from app.modules.parsing.graph_construction import code_graph_service

    recorded: list[str] = []

    def capture_info(message, *args, **kwargs):
        recorded.append(str(message))

    mock_logger = MagicMock()
    mock_logger.info.side_effect = capture_info

    # Step 1/4 lives on parse_repository_structure (host-side structure parse).
    fake_graph = nx.MultiDiGraph()
    fake_graph.add_node(
        "a.py",
        type="FILE",
        name="a.py",
        file="a.py",
        line=1,
        end_line=1,
        text="",
    )
    fake_repo_map = MagicMock()
    fake_repo_map.create_graph.return_value = fake_graph

    with patch.object(code_graph_service, "logger", mock_logger), patch(
        "app.modules.parsing.graph_construction.parsing_repomap.RepoMap",
        return_value=fake_repo_map,
    ):
        code_graph_service.parse_repository_structure("/tmp/repo", "proj-1")

    # Steps 2–4/4 are emitted by _store_graph (stubbed Neo4j + Qdrant).
    store_graph = nx.MultiDiGraph()
    store_graph.add_node(
        "mod.py",
        type="FILE",
        name="mod.py",
        file="mod.py",
        line=1,
        end_line=2,
        text="pass",
    )
    store_graph.add_edge("mod.py", "mod.py", type="CONTAINS")

    service = code_graph_service.CodeGraphService.__new__(
        code_graph_service.CodeGraphService
    )
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = False
    service.driver = mock_driver
    service.db = MagicMock()
    service.qdrant_client = MagicMock()

    with patch.object(code_graph_service, "logger", mock_logger), patch(
        "app.modules.parsing.graph_construction.code_graph_service.index_nodes_to_qdrant",
        return_value=(0, 0, 0),
    ):
        service._store_graph(store_graph, "proj-1", "user-1")

    phase_messages = [
        msg
        for msg in recorded
        if "Step 1/4:" in msg
        or "Step 2/4:" in msg
        or "Step 3/4:" in msg
        or "Step 4/4:" in msg
    ]
    assert len(phase_messages) == 4
    assert "Step 1/4: Parsing repository structure" in phase_messages[0]
    assert "Step 2/4: Creating Neo4j indices" in phase_messages[1]
    assert "Step 3/4: Inserting" in phase_messages[2]
    assert "nodes into Neo4j" in phase_messages[2]
    assert "Step 4/4: Inserting" in phase_messages[3]
    assert "relationships into Neo4j" in phase_messages[3]
    assert phase_messages[2] != phase_messages[3]
