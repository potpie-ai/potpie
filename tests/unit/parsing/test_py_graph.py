import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app" / "src" / "parsing" / "parsing"))

import pytest
import py_graph


pytestmark = pytest.mark.unit


class TestPyGraph:
    def test_create_graph_produces_non_file_nodes(self, tmp_path):
        (tmp_path / "sample.py").write_text(
            '''"""Sample module for testing."""


def hello(name: str) -> str:
    return f"Hello, {name}!"


class Greeter:
    def greet(self) -> str:
        return hello("world")
'''
        )

        graph = py_graph.create_graph(str(tmp_path))

        node_types = [data.get("type") for _, data in graph.nodes(data=True)]

        assert "FILE" in node_types

        non_file_types = [t for t in node_types if t != "FILE"]
        assert len(non_file_types) > 0, f"Expected non-FILE nodes, got: {node_types}"

        assert "CLASS" in node_types, f"Expected CLASS, got: {node_types}"
        assert "FUNCTION" in node_types, f"Expected FUNCTION, got: {node_types}"
