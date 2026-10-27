"""Graph CLI inspection behavior."""

from __future__ import annotations

import json

import pytest
from potpie_context_engine.core.ports.graph.inspection import (
    GraphEdge,
    GraphNode,
    GraphSlice,
)
from typer.testing import CliRunner

from potpie.cli.commands import _common, graph
from tests.graph_cli_support import (
    _assert_graph_envelope,
    _Backend,
    _Graph,
    _Host,
    _Inspection,
)

pytestmark = pytest.mark.unit


def test_graph_neighborhood_returns_inspection_slice() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph(), backend=_Backend()))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "neighborhood",
            "--entity",
            "service:web",
            "--predicate",
            "DEPENDS_ON",
            "--direction",
            "out",
            "--depth",
            "2",
            "--limit",
            "5",
            "--detail",
            "full",
        ],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.neighborhood")
    assert body["entity_key"] == "service:web"
    assert body["predicates"] == ["DEPENDS_ON"]
    assert body["detail"] == "full"
    assert body["edges"][0]["from"] == "service:web"


def test_graph_neighborhood_defaults_to_relation_summary() -> None:
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph(), backend=_Backend()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["neighborhood", "--entity", "service:web", "--predicate", "DEPENDS_ON"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.neighborhood")
    assert body["detail"] == "summary"
    assert body["relation_count"] == 1
    assert "edges" not in body
    assert body["relations"][0]["from_key"] == "service:web"
    assert body["relations"][0]["to_key"] == "service:api"
    assert body["relations"][0]["source_refs"] == ["repo:manifest"]
    assert body["relations"][0]["truth"] == "source_observation"


def test_full_neighborhood_diagnostic_redacts_credential_metadata(monkeypatch) -> None:
    def neighborhood(self, *, pot_id, entity_key, **kwargs):
        return GraphSlice(
            pot_id=pot_id,
            nodes=(
                GraphNode(
                    key=entity_key,
                    labels=("Repository",),
                    properties={
                        "owner": "acme",
                        "source_ref": "test:repository",
                        "temp_clone_token": "fixture-secret",
                        "sessionCookie": "fixture-cookie",
                    },
                ),
            ),
            edges=(),
        )

    monkeypatch.setattr(_Inspection, "neighborhood", neighborhood)
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph(), backend=_Backend()))
    result = CliRunner().invoke(
        graph.graph_app,
        ["neighborhood", "--entity", "repository:shop", "--detail", "full"],
    )

    assert result.exit_code == 0, result.output
    assert "fixture-secret" not in result.output
    assert "fixture-cookie" not in result.output
    body = _assert_graph_envelope(json.loads(result.output), "graph.neighborhood")
    assert body["nodes"][0]["properties"]["owner"] == "acme"
    assert body["nodes"][0]["properties"]["source_ref"] == "test:repository"


def test_neighborhood_reports_a_missing_anchor_and_names_the_lookup(
    monkeypatch,
) -> None:
    def neighborhood(self, *, pot_id, entity_key, **kwargs):
        return GraphSlice(pot_id=pot_id, nodes=(), edges=())

    monkeypatch.setattr(_Inspection, "neighborhood", neighborhood)
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph(), backend=_Backend()))
    result = CliRunner().invoke(
        graph.graph_app, ["neighborhood", "--entity", "service:missing"]
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.neighborhood")
    assert body["identity_status"] == "missing"
    assert "potpie graph search-entities" in emitted["recommended_next_action"]


def test_neighborhood_full_text_shows_bounded_extra_detail_and_hidden_counts() -> None:
    payload = {
        "entity_key": "service:web",
        "identity_status": "exact",
        "detail": "full",
        "node_count": 7,
        "truncated": True,
        "relations": [
            {
                "predicate": "DEPENDS_ON",
                "from": "service:web",
                "to": f"service:{i}",
                "source_refs": [],
                "fact": f"dependency {i}",
            }
            for i in range(25)
        ],
        "nodes": [
            {
                "key": f"service:{i}",
                "labels": ["Service"],
                "properties": {"name": f"Node {i}"},
            }
            for i in range(7)
        ],
        "edges": [
            {
                "from": "service:web",
                "to": f"service:{i}",
                "predicate": "DEPENDS_ON",
                "properties": {"confidence": 0.9},
            }
            for i in range(25)
        ],
    }

    text = graph._neighborhood_human(payload)

    assert "5 relations hidden" in text
    assert "Node 0" in text
    assert '"confidence":0.9' in text
    assert "2 node details hidden" in text
    assert "20 edge details hidden" in text
    assert "completeness is unknown" in text


def test_neighborhood_full_text_spells_out_the_anchor_answer_fields() -> None:
    payload = {
        "entity_key": "fix:cookie",
        "identity_status": "exact",
        "detail": "full",
        "node_count": 1,
        "relations": [],
        "nodes": [
            {
                "key": "fix:cookie",
                "labels": ["Fix"],
                "properties": {
                    "root_cause": "Shared cookie name across daemons",
                    "fix_steps": ["Use a per-daemon cookie suffix"],
                    "verification_status": "passed",
                    "source_status": "current",
                },
            }
        ],
        "edges": [],
    }

    text = graph._neighborhood_human(payload)

    assert "root_cause: Shared cookie name across daemons" in text
    assert 'fix_steps: ["Use a per-daemon cookie suffix"]' in text
    assert "verification_status: passed" in text
    assert "source_status: current" in text


def test_neighborhood_byte_budget_preserves_fix_fields_and_exact_route() -> None:
    payload = {
        "entity_key": "fix:cookie",
        "identity_status": "exact",
        "detail": "full",
        "depth": 2,
        "direction": "both",
        "limit": 20,
        "predicates": ["VERIFIED"],
        "truncated": False,
        "node_count": 2,
        "relations": [
            {
                "predicate": "VERIFIED",
                "from_key": "activity:check",
                "to_key": "fix:cookie",
                "fact": "verified " + "x" * 3_000,
                "source_refs": ["test:verification"],
            }
            for _ in range(20)
        ],
        "nodes": [
            {
                "key": "fix:cookie",
                "labels": ["Fix"],
                "properties": {
                    "root_cause": "Shared cookie name across daemons",
                    "fix_steps": ["Use a per-daemon cookie suffix"],
                    "verification_status": "passed",
                    "unrelated": "x" * 8_000,
                },
            },
            {"key": "activity:check", "labels": ["Activity"], "properties": {}},
        ],
        "edges": [
            {
                "predicate": "VERIFIED",
                "from": "activity:check",
                "to": "fix:cookie",
                "properties": {"fact": "x" * 3_000},
            }
            for _ in range(20)
        ],
    }
    payload["relation_count"] = len(payload["relations"])

    bounded = graph._bound_neighborhood_payload(payload, pot_id="p")

    assert len(json.dumps(bounded, ensure_ascii=False).encode()) <= 32_768
    assert bounded["omitted_relation_count"] > 0
    assert bounded["omitted_field_count"] > 0
    assert (
        bounded["nodes"][0]["properties"]["root_cause"]
        == "Shared cookie name across daemons"
    )
    assert bounded["nodes"][0]["properties"]["fix_steps"] == [
        "Use a per-daemon cookie suffix"
    ]
    assert (
        "--predicate VERIFIED --unbounded --pot p" in bounded["recommended_next_action"]
    )


def test_neighborhood_cli_byte_budget_includes_workbench_envelope(monkeypatch) -> None:
    from potpie_context_engine.core.agent_envelope import DEFAULT_OUTPUT_BUDGET_BYTES

    def neighborhood(self, *, pot_id, entity_key, **kwargs):
        return GraphSlice(
            pot_id=pot_id,
            nodes=(
                GraphNode(
                    key=entity_key,
                    labels=("Fix",),
                    properties={
                        "root_cause": "Shared cookie name",
                        "fix_steps": ["Use a per-daemon suffix"],
                        "diagnostic": "x" * 8_000,
                    },
                ),
            ),
            edges=tuple(
                GraphEdge(
                    predicate="VERIFIED",
                    from_key=entity_key,
                    to_key=f"activity:check-{index}",
                    properties={"fact": "x" * 3_000, "source_refs": ["test:fix"]},
                )
                for index in range(30)
            ),
        )

    monkeypatch.setattr(_Inspection, "neighborhood", neighborhood)
    _common.set_json(True)
    _common.set_runtime(_Host(_Graph(), backend=_Backend()))
    result = CliRunner().invoke(
        graph.graph_app,
        ["neighborhood", "--entity", "fix:cookie", "--detail", "full"],
    )

    assert result.exit_code == 0, result.output
    # One output budget: the whole emitted envelope, not only the body.
    assert len(result.output.encode("utf-8")) <= DEFAULT_OUTPUT_BUDGET_BYTES
    body = _assert_graph_envelope(json.loads(result.output), "graph.neighborhood")
    assert body["nodes"][0]["properties"]["root_cause"] == "Shared cookie name"
    assert body["omitted_relation_count"] > 0


from tests.graph_cli_support import graph_cli_state as graph_cli_state
