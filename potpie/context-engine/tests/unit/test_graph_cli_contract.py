"""CLI contract coverage for Graph Surface Lite."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli.commands import _common, graph
from domain.agent_envelope import AgentEnvelope, CoverageReport, EvidenceItem
from domain.nudge import GraphNudgeResult
from domain.semantic_mutations import (
    SemanticMutationResult,
    SemanticMutationValidationIssue,
)
from domain.ports.services.graph_service import (
    GraphEntityCandidate,
    GraphEntitySearchResult,
)
from domain.ports.graph.analytics import RepairReport
from domain.ports.graph.backend import BackendCapabilities

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_json_mode():
    yield
    _common.set_json(False)


class _Pot:
    pot_id = "p"
    name = "default"


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]


class _Graph:
    def __init__(
        self,
        mutate_result: SemanticMutationResult | None = None,
        read_result: AgentEnvelope | None = None,
    ) -> None:
        self.mutate_result = mutate_result
        self.read_result = read_result
        self.read_called = False
        self.read_request = None

    def mutate(self, _request):
        assert self.mutate_result is not None
        return self.mutate_result

    def read(self, _request):
        self.read_called = True
        self.read_request = _request
        if self.read_result is None:
            raise AssertionError("read should not be called")
        return self.read_result


class _Nudge:
    def __init__(self) -> None:
        self.request = None

    def nudge(self, request):
        self.request = request
        return GraphNudgeResult(
            ok=True,
            silent=True,
            event=request.event.replace("-", "_"),
            pot_id=request.pot_id,
            detail="nothing relevant",
        )


class _Host:
    def __init__(
        self,
        graph_service: _Graph,
        nudge_service: _Nudge | None = None,
        backend=None,
    ) -> None:
        self.graph = graph_service
        self.nudge = nudge_service or _Nudge()
        self.pots = _Pots()
        self.backend = backend


class _Analytics:
    def __init__(self) -> None:
        self.calls = []

    def repair(self, pot_id, *, targets):
        self.calls.append((pot_id, tuple(targets)))
        return RepairReport(
            pot_id=pot_id,
            targets=tuple(targets),
            repaired={"entity_summaries": 2},
            detail="repaired 2 entity summaries",
        )


class _Backend:
    def __init__(self) -> None:
        self.analytics = _Analytics()


class _UnsupportedBackend:
    profile = "neo4j"

    def __init__(self) -> None:
        self.accessed_ports: list[str] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            profile=self.profile, inspection=False, snapshot=False
        )

    @property
    def inspection(self):
        self.accessed_ports.append("inspection")
        raise AssertionError("inspection port should not be reached")

    @property
    def snapshot(self):
        self.accessed_ports.append("snapshot")
        raise AssertionError("snapshot port should not be reached")


def _valid_mutation_payload() -> dict:
    return {
        "operations": [
            {
                "op": "link_entities",
                "subgraph": "infra_topology",
                "subject": {"key": "service:payments-api", "type": "Service"},
                "predicate": "DEPENDS_ON",
                "object": {"key": "service:ledger-api", "type": "Service"},
                "truth": "source_observation",
                "evidence": [{"source_ref": "repo:manifest"}],
                "description": "payments depends on ledger",
            }
        ]
    }


def test_graph_entity_search_result_includes_summary() -> None:
    result = GraphEntitySearchResult(
        entities=(
            GraphEntityCandidate(
                key="service:web",
                labels=("Service",),
                name="web",
                summary="Web frontend service.",
                description="Web frontend service for browser clients.",
            ),
        ),
        match_mode="lexical",
        graph_contract_version="v1.5",
        ontology_version="test",
    )

    payload = result.to_dict()

    assert payload["entities"][0]["summary"] == "Web frontend service."


def test_graph_repair_accepts_entity_summaries_target() -> None:
    backend = _Backend()
    _common.set_host(_Host(_Graph(), backend=backend))

    result = CliRunner().invoke(graph.graph_app, ["repair", "--entity-summaries"])

    assert result.exit_code == 0, result.output
    assert backend.analytics.calls == [("p", ("entity_summaries",))]
    assert "repaired 2 entity summaries" in result.output


@pytest.mark.parametrize(
    ("args", "capability", "method"),
    [
        (["inspect", "service:web"], "inspection", "neighborhood"),
        (["export", "out.json"], "snapshot", "export"),
        (["import", "in.json"], "snapshot", "import_"),
    ],
)
def test_graph_capability_commands_precheck_backend_capabilities(
    args: list[str],
    capability: str,
    method: str,
) -> None:
    _common.set_json(True)
    backend = _UnsupportedBackend()
    _common.set_host(_Host(_Graph(), backend=backend))

    result = CliRunner().invoke(graph.graph_app, args)

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    emitted = json.loads(result.output)
    assert emitted["code"] == "not_implemented"
    assert f"graph.neo4j.{capability}.{method}" in emitted["message"]
    assert emitted["recommended_next_action"]
    assert backend.accessed_ports == []


def test_graph_mutate_rejection_emits_result_and_exits_nonzero(tmp_path) -> None:
    _common.set_json(True)
    result_payload = SemanticMutationResult(
        ok=False,
        status="rejected",
        risk="low",
        pot_id="p",
        issues=(
            SemanticMutationValidationIssue(
                code="invalid_endpoints",
                message="invalid endpoint pair",
            ),
        ),
    )
    graph_service = _Graph(mutate_result=result_payload)
    _common.set_host(_Host(graph_service))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        ["mutate", "--file", str(payload_file)],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    assert emitted["ok"] is False
    assert emitted["status"] == "rejected"
    assert emitted["issues"][0]["code"] == "invalid_endpoints"


@pytest.mark.parametrize("scope", ["service", "service:"])
def test_graph_read_rejects_malformed_scope_before_service_call(scope: str) -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--view", "bugs.prior_occurrences", "--scope", scope],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is False
    emitted = json.loads(result.output)
    assert emitted["code"] == "validation_error"
    assert "invalid --scope entry" in emitted["message"]


def _timeline_env() -> AgentEnvelope:
    return AgentEnvelope(
        pot_id="p",
        intent="unknown",
        overall_confidence="high",
        coverage=(CoverageReport(include="timeline", status="complete"),),
        metadata={
            "view": "recent_changes.timeline",
            "subgraph": "recent_changes",
            "backed": True,
            "read_shape": "entity_relations",
        },
        items=(
            EvidenceItem(
                include="timeline",
                candidate_key="activity:github:pr-2",
                score=0.9,
                coverage_status="complete",
                payload={
                    "entity": {"key": "activity:github:pr-2"},
                    "relations": [
                        {
                            "predicate": "TOUCHED",
                            "from_key": "activity:github:pr-2",
                            "to_key": "repo:github.com/acme/widgets",
                            "fact": 'PR #2 "newer" was merged into acme/widgets on 2026-06-08 by Bob.',
                            "source_refs": ["github:pr:2"],
                            "truth": "timeline_event",
                        },
                        {
                            "predicate": "PERFORMED",
                            "from_key": "person:bob",
                            "to_key": "activity:github:pr-2",
                            "fact": 'PR #2 "newer" was merged into acme/widgets on 2026-06-08 by Bob.',
                            "source_refs": ["github:pr:2"],
                            "truth": "timeline_event",
                        },
                    ],
                },
            ),
            EvidenceItem(
                include="timeline",
                candidate_key="activity:github:pr-1",
                score=1.0,
                coverage_status="complete",
                payload={
                    "entity": {"key": "activity:github:pr-1"},
                    "relations": [
                        {
                            "predicate": "TOUCHED",
                            "from_key": "activity:github:pr-1",
                            "to_key": "repo:github.com/acme/widgets",
                            "fact": 'PR #1 "older" was merged into acme/widgets on 2026-06-01 by Alice.',
                            "source_refs": ["github:pr:1"],
                            "truth": "timeline_event",
                        }
                    ],
                },
            ),
        ),
    )


def test_graph_read_timeline_defaults_to_deduped_event_json() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--view", "recent_changes.timeline", "--limit", "1"],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.limit == 40
    emitted = json.loads(result.output)
    assert emitted["read_shape"] == "events"
    assert emitted["event_count"] == 1
    assert emitted["events"][0]["source_refs"] == ["github:pr:2"]
    assert emitted["freshness"]["local_worktree_included"] is False


def test_graph_nudge_accepts_dash_event_alias() -> None:
    _common.set_json(True)
    nudge_service = _Nudge()
    _common.set_host(_Host(_Graph(), nudge_service=nudge_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["nudge", "--event", "pre-edit", "--session", "sess-1", "--path", "src/app.py"],
    )

    assert result.exit_code == 0
    assert nudge_service.request.event == "pre-edit"
    emitted = json.loads(result.output)
    assert emitted["ok"] is True
    assert emitted["event"] == "pre_edit"


def test_timeline_recent_passes_project_scope_and_time_window() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.timeline_app,
        ["--time-window", "7d", "--limit", "2"],
    )

    assert result.exit_code == 0
    req = graph_service.read_request
    assert req.view == "recent_changes.timeline"
    assert req.scope == {}
    assert req.limit == 40
    assert req.since is not None
    emitted = json.loads(result.output)
    assert emitted["event_count"] == 2
