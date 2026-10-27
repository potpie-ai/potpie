"""Graph CLI status behavior."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _Backend,
    _Graph,
    _GraphByPot,
    _Host,
    _NotReadyGraph,
)
from tests.graph_cli_support import graph_cli_state as graph_cli_state
from tests.graph_cli_workbench_doubles import QualityRecorder, _quality_result

pytestmark = pytest.mark.unit


def test_graph_status_json_uses_workbench_envelope() -> None:
    _common.set_json(True)
    workbench = QualityRecorder(
        quality_result=_quality_result(report="summary", status="watch")
    )
    _common.set_runtime(_Host(_Graph(), backend=_Backend(), graph_workbench=workbench))

    result = CliRunner().invoke(graph.graph_app, ["status"])

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.status")
    assert emitted["subgraph_versions"] == {"_global": 3}
    assert body["pot"]["name"] == "default"
    assert body["pot"]["source_count"] == 0
    assert body["graph_service"]["supported_commands"][0] == "status"
    # Workbench-facing status speaks view vocabulary only; the include
    # families backing them stay on the data-plane surface.
    assert body["graph_service"]["backed_views"] == ["recent_changes.timeline"]
    assert "reader_backed_includes" not in body["graph_service"]
    assert body["backend"]["profile"] == "memory"
    assert body["health_status"] == "watch"
    assert body["quality"]["source"] == "quality_summary"
    assert body["quality"]["quality_counts"]["projection_drift"] == 2
    assert workbench.quality_calls == [
        {
            "pot_id": "p",
            "report": "summary",
            "subgraph": None,
            "limit": 20,
            "confidence_threshold": 0.5,
        }
    ]


def test_graph_status_not_ready_recommends_backend_doctor() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_NotReadyGraph(), backend=_Backend()))

    result = CliRunner().invoke(graph.graph_app, ["status"])

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.status")
    assert body["backend"]["ready"] is False
    assert body["backend"]["detail"] == "mutation store is unavailable"
    assert body["backend"]["readiness_command"] == "potpie backend doctor"
    assert "potpie backend doctor" in emitted["recommended_next_action"]


def test_graph_status_warns_when_active_repo_pot_is_empty(monkeypatch) -> None:
    class Pot:
        def __init__(self, pot_id, name, active=False):
            self.pot_id = pot_id
            self.name = name
            self.active = active

    class Source:
        kind = "repo"
        name = "github.com/acme/shop"
        location = "github.com/acme/shop"

    class Pots:
        def __init__(self):
            self.p1 = Pot("p1", "empty", True)
            self.p2 = Pot("p2", "populated")

        def active_pot(self):
            return self.p1

        def list_pots(self):
            return [self.p1, self.p2]

        def list_sources(self, *, pot_id):
            return [Source()]

    monkeypatch.setattr(
        graph, "_current_git_remote", lambda cwd: "github.com/acme/shop", raising=False
    )
    from potpie.cli.commands import _common

    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    _common.set_json(True)
    host = _Host(_GraphByPot({"p1": {"claims": 0}, "p2": {"claims": 82}}))
    host.pots = Pots()
    _common.set_runtime(host)

    result = CliRunner().invoke(graph.graph_app, ["status"])

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.status", pot_id="p1")
    assert body["pot"]["id"] == "p1"
    assert emitted["warnings"]
    assert "p2" in emitted["warnings"][0]
    assert "82 claims" in emitted["warnings"][0]
