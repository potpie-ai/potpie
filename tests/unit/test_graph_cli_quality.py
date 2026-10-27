"""Graph CLI quality behavior."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import _assert_graph_envelope, _Graph, _Host
from tests.graph_cli_support import graph_cli_state as graph_cli_state
from tests.graph_cli_workbench_doubles import QualityRecorder, _quality_result

pytestmark = pytest.mark.unit


def test_graph_quality_summary_returns_envelope() -> None:
    _common.set_json(True)
    workbench = QualityRecorder(quality_result=_quality_result(report="summary"))
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(graph.graph_app, ["quality", "summary"])

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.quality.summary")
    assert emitted["subgraph_versions"] == {"_global": 3}
    assert body["report"] == "summary"
    assert body["metrics"]["counts"]["claims"] == 3
    assert workbench.quality_calls == [
        {
            "pot_id": "p",
            "report": "summary",
            "subgraph": None,
            "limit": 50,
            "confidence_threshold": 0.5,
        }
    ]


def test_graph_quality_low_confidence_passes_filters() -> None:
    _common.set_json(True)
    workbench = QualityRecorder(
        quality_result=_quality_result(report="low-confidence", status="watch")
    )
    _common.set_runtime(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "quality",
            "low-confidence",
            "--subgraph",
            "infra_topology",
            "--limit",
            "10",
            "--threshold",
            "0.75",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.quality.low-confidence")
    assert body["finding_count"] == 1
    assert body["findings"][0]["kind"] == "low-confidence"
    assert workbench.quality_calls == [
        {
            "pot_id": "p",
            "report": "low-confidence",
            "subgraph": "infra_topology",
            "limit": 10,
            "confidence_threshold": 0.75,
        }
    ]
