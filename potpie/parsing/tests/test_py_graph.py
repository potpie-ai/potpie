from parsing.py_graph import create_graph


def test_create_graph_produces_non_file_nodes(tmp_path):
    (tmp_path / "sample.py").write_text(
        '"""Sample module for testing."""\n\n\ndef hello(name: str) -> str:\n    return f"Hello, {name}!"\n\n\nclass Greeter:\n    def greet(self) -> str:\n        return hello("world")\n'
    )
    graph = create_graph(str(tmp_path))
    node_types = [data.get("type") for _, data in graph.nodes(data=True)]
    assert "FILE" in node_types
    non_file_types = [t for t in node_types if t != "FILE"]
    assert len(non_file_types) > 0, f"Expected non-FILE nodes, got: {node_types}"
    assert "CLASS" in node_types, f"Expected CLASS, got: {node_types}"
    assert "FUNCTION" in node_types, f"Expected FUNCTION, got: {node_types}"
