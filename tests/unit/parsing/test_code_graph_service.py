from pathlib import Path


def test_graph_generation_logs_four_numbered_phases():
    source_path = (
        Path(__file__).parents[3]
        / "app/modules/parsing/graph_construction/code_graph_service.py"
    )
    source = source_path.read_text()

    assert "Step 1/4: Parsing repository structure" in source
    assert "Step 2/4: Creating Neo4j indices" in source
    assert "Step 3/4: Inserting {node_count} nodes into Neo4j" in source
    assert (
        "Step 4/4: Inserting {relationship_count} relationships into Neo4j"
        in source
    )
