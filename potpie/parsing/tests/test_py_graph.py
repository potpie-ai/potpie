import networkx as nx
import pytest

from parsing import py_graph
from parsing.py_graph import create_graph, create_relationship


def test_create_graph_filters_descendants_of_a_hidden_repository(tmp_path, monkeypatch):
    repo = tmp_path / ".repos_local" / "project"
    repo.mkdir(parents=True)
    included = {"main.py", ".github/check.py", ".vscode/settings.py", "src/app.py"}
    excluded = {".git/config", ".cache/cached.py", ".github/.cache/hidden.py"}
    for name in included | excluded:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n")
    # os.walk does not follow directory symlinks by default.
    (repo / "linked").symlink_to(repo / "src", target_is_directory=True)
    monkeypatch.setattr(py_graph, "get_tags", lambda *_args: ())

    graph = create_graph(str(repo))

    assert set(graph.nodes) == included


def test_create_graph_accepts_a_worktree_git_file(tmp_path, monkeypatch):
    (tmp_path / ".git").write_text("gitdir: /repo/.git/worktrees/example\n")
    (tmp_path / "main.py").write_text("pass\n")
    monkeypatch.setattr(py_graph, "get_tags", lambda *_args: ())

    assert set(create_graph(str(tmp_path)).nodes) == {".git", "main.py"}


@pytest.mark.parametrize(
    "source_type", ["FUNCTION", "CLASS", "FILE", "INTERFACE", None]
)
@pytest.mark.parametrize(
    "target_type", ["FUNCTION", "CLASS", "FILE", "INTERFACE", None]
)
@pytest.mark.parametrize("source", ["source", "ExampleImpl"])
def test_reference_direction_policy(source_type, target_type, source):
    graph = nx.MultiDiGraph()
    graph.add_node(source, type=source_type)
    graph.add_node("target", type=target_type)
    seen = set()
    allowed_targets = {
        "FUNCTION": {"FUNCTION", "CLASS", "FILE", "INTERFACE", None},
        "CLASS": {"CLASS"},
        "FILE": {"CLASS", "FUNCTION"},
        "INTERFACE": {"CLASS"},
        None: {"CLASS"},
    }
    allowed = target_type in allowed_targets[source_type]

    assert (
        create_relationship(
            graph, source, "target", "REFERENCES", seen, {"ident": "example"}
        )
        is allowed
    )
    assert graph.number_of_edges() == int(allowed)
    if allowed:
        assert graph[source]["target"][0]["ident"] == "example"
        assert seen == {(source, "target", "REFERENCES")}


@pytest.mark.parametrize("relationship_type", ["REFERENCES", "CONTAINS", "OTHER"])
def test_relationship_rejects_self_edges_and_duplicates(relationship_type):
    graph = nx.MultiDiGraph()
    graph.add_node("source", type="FUNCTION")
    graph.add_node("target", type="FUNCTION")
    assert not create_relationship(graph, "source", "source", relationship_type, set())
    for edge in [
        ("source", "target", relationship_type),
        ("target", "source", relationship_type),
    ]:
        assert not create_relationship(
            graph, "source", "target", relationship_type, {edge}
        )
    if relationship_type != "REFERENCES":
        assert not create_relationship(
            graph, "source", "target", relationship_type, set()
        )
    assert graph.number_of_edges() == 0


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

        graph = create_graph(str(tmp_path))

        node_types = [data.get("type") for _, data in graph.nodes(data=True)]

        assert "FILE" in node_types

        non_file_types = [t for t in node_types if t != "FILE"]
        assert len(non_file_types) > 0, f"Expected non-FILE nodes, got: {node_types}"

        assert "CLASS" in node_types, f"Expected CLASS, got: {node_types}"
        assert "FUNCTION" in node_types, f"Expected FUNCTION, got: {node_types}"
