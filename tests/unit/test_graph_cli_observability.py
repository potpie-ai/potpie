"""Graph CLI observability behavior."""

from __future__ import annotations

import json

import pytest
from potpie_context_engine.bootstrap import observability_runtime
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import _Graph, _Host, _timeline_env
from tests.graph_cli_telemetry import (
    _bind_graph_product_analytics,
    _RecordingObservability,
)
from tests.graph_cli_workbench_doubles import (
    InboxRecorder,
    ProposalRecorder,
    QualityRecorder,
    _inbox_result,
    _proposal,
    _quality_result,
)

pytestmark = pytest.mark.unit


def test_graph_workbench_commands_emit_v2_observability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _common.set_json(True)
    obs = _RecordingObservability()
    sentry_counts = []
    sentry_distributions = []
    monkeypatch.setattr(observability_runtime, "_OBSERVABILITY", obs)
    monkeypatch.setattr(
        "potpie_context_engine.bootstrap.sentry_metrics_runtime.count",
        lambda name, value=1, *, unit=None, attributes=None: sentry_counts.append(
            (name, value, unit, dict(attributes or {}))
        ),
    )
    monkeypatch.setattr(
        "potpie_context_engine.bootstrap.sentry_metrics_runtime.distribution",
        lambda name, value, *, unit=None, attributes=None: sentry_distributions.append(
            (name, value, unit, dict(attributes or {}))
        ),
    )
    monkeypatch.setattr(
        "potpie_context_engine.bootstrap.sentry_metrics_runtime.flush",
        lambda timeout=2.0: None,
    )
    sink = _bind_graph_product_analytics(monkeypatch)
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    _common.set_runtime(
        _Host(_Graph(), graph_workbench=ProposalRecorder(proposal=_proposal()))
    )
    propose = CliRunner().invoke(
        graph.graph_app,
        ["propose", "--file", str(payload_file)],
    )
    assert propose.exit_code == 0, propose.output

    _common.set_runtime(
        _Host(_Graph(), graph_workbench=InboxRecorder(inbox_result=_inbox_result()))
    )
    inbox = CliRunner().invoke(
        graph.graph_app,
        ["inbox", "add", "--summary", "Possible graph update"],
    )
    assert inbox.exit_code == 0, inbox.output

    _common.set_runtime(
        _Host(
            _Graph(),
            graph_workbench=QualityRecorder(
                quality_result=_quality_result(report="summary")
            ),
        )
    )
    quality = CliRunner().invoke(graph.graph_app, ["quality", "summary"])
    assert quality.exit_code == 0, quality.output

    propose_counter = next(
        item for item in obs.counters if item[0] == "ce.graph.propose_total"
    )
    assert propose_counter[2]["risk"] == "low"
    assert propose_counter[2]["status"] == "validated"
    inbox_counter = next(
        item for item in obs.counters if item[0] == "ce.graph.inbox_total"
    )
    assert inbox_counter[2]["operation"] == "add"
    quality_counter = next(
        item for item in obs.counters if item[0] == "ce.graph.quality_total"
    )
    assert quality_counter[2]["report"] == "summary"
    assert {span[0] for span in obs.spans} >= {
        "graph.propose",
        "graph.inbox",
        "graph.quality",
    }
    graph_sentry_counts = [
        item for item in sentry_counts if item[0].startswith("ce.graph.")
    ]
    assert {item[0] for item in graph_sentry_counts} >= {
        "ce.graph.propose_total",
        "ce.graph.inbox_total",
        "ce.graph.quality_total",
    }
    sentry_propose = next(
        item for item in graph_sentry_counts if item[0] == "ce.graph.propose_total"
    )
    assert sentry_propose[3]["risk"] == "low"
    assert sentry_propose[3]["status"] == "validated"
    assert any(item[0] == "ce.graph.propose_ms" for item in sentry_distributions)
    analytics_events = [
        event for event in sink.events if event.name == "cli_usage_command_succeeded"
    ]
    assert [event.properties["command"] for event in analytics_events] == [
        "graph.propose",
        "graph.inbox.add",
        "graph.quality.summary",
    ]
    assert {event.properties["result_kind"] for event in analytics_events} == {
        "graph_command"
    }
    assert analytics_events[0].properties["risk"] == "low"
    assert analytics_events[1].properties["operation"] == "add"
    assert analytics_events[2].properties["report"] == "summary"
    assert "pot_id" not in analytics_events[0].properties
    assert "request_id" not in analytics_events[0].properties


def test_graph_read_emits_v2_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    _common.set_json(True)
    obs = _RecordingObservability()
    monkeypatch.setattr(observability_runtime, "_OBSERVABILITY", obs)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "recent_changes", "--view", "timeline"],
    )

    assert result.exit_code == 0, result.output
    span = next(item for item in obs.spans if item[0] == "graph.read")
    assert span[1]["command"] == "graph.read"
    assert span[1]["subgraph"] == "recent_changes"
    assert span[1]["view"] == "recent_changes.timeline"
    counter = next(item for item in obs.counters if item[0] == "ce.graph.read_total")
    assert counter[1] == 1
    assert counter[2]["result"] == "ok"
    assert counter[2]["subgraph"] == "recent_changes"
    assert counter[2]["view"] == "recent_changes.timeline"
    assert any(item[0] == "ce.graph.read_ms" for item in obs.histograms)


def test_graph_validation_error_does_not_record_usage_analytics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _common.set_json(True)
    sink = _bind_graph_product_analytics(monkeypatch)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_runtime(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "recent_changes"],
    )

    assert result.exit_code == 1
    assert [
        event for event in sink.events if event.name == "cli_usage_command_succeeded"
    ] == []


from tests.graph_cli_support import _valid_mutation_payload
from tests.graph_cli_support import graph_cli_state as graph_cli_state
