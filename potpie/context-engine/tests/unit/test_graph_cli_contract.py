"""CLI contract coverage for Graph Surface Lite."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli.commands import _common, graph
from domain.graph_plans import (
    GraphMutationCommitResult,
    GraphMutationDiff,
    GraphMutationProposal,
)
from domain.nudge import GraphNudgeResult
from domain.graph_views import views_for_catalog
from domain.ports.services.graph_service import (
    DataPlaneStatus,
    GraphCatalogResult,
    GraphEntityCandidate,
    GraphEntitySearchResult,
    GraphReadResult,
)
from domain.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice
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
        read_result: GraphReadResult | None = None,
        read_error: Exception | None = None,
        catalog_error: Exception | None = None,
    ) -> None:
        self.read_result = read_result
        self.read_error = read_error
        self.catalog_error = catalog_error
        self.read_called = False
        self.read_request = None

    def catalog(self, _request):
        if self.catalog_error is not None:
            raise self.catalog_error
        return GraphCatalogResult(
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
            commands=("catalog", "read", "search-entities", "mutate"),
            truth_classes=("agent_claim",),
            mutation_operations=("link_entities",),
            review_required_operations=("supersede_claim",),
            deferred_operations=("patch_entity",),
            views=tuple(views_for_catalog()),
            entity_types=(),
            predicates=(),
            match_mode="lexical",
        )

    def data_plane_status(self, pot_id):
        return DataPlaneStatus(
            pot_id=pot_id,
            backend_profile="memory",
            backend_ready=True,
            reader_backed_includes=("timeline",),
            counts={"claims": 3},
            freshness={},
            quality={},
            match_mode="lexical",
        )

    def read(self, _request):
        self.read_called = True
        self.read_request = _request
        if self.read_error is not None:
            raise self.read_error
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


class _Workbench:
    def __init__(
        self,
        *,
        proposal: GraphMutationProposal | None = None,
        commit_result: GraphMutationCommitResult | None = None,
    ) -> None:
        self.proposal = proposal
        self.commit_result = commit_result
        self.propose_calls = []
        self.commit_calls = []

    def propose(self, payload, *, pot_id, ttl_seconds=None):
        self.propose_calls.append((payload, pot_id, ttl_seconds))
        if self.proposal is None:
            raise AssertionError("propose should not be called")
        return self.proposal

    def commit(self, plan_id, *, pot_id, approved_by=None):
        self.commit_calls.append((plan_id, pot_id, approved_by))
        if self.commit_result is None:
            raise AssertionError("commit should not be called")
        return self.commit_result


class _Host:
    def __init__(
        self,
        graph_service: _Graph,
        nudge_service: _Nudge | None = None,
        backend=None,
        graph_workbench: _Workbench | None = None,
    ) -> None:
        self.graph = graph_service
        self.graph_workbench = graph_workbench or _Workbench()
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

    profile = "memory"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(profile=self.profile, inspection=True)

    @property
    def inspection(self):
        return _Inspection()


class _Inspection:
    def neighborhood(
        self,
        *,
        pot_id,
        entity_key,
        depth=1,
        direction="both",
        predicates=(),
        limit=None,
    ):
        return GraphSlice(
            pot_id=pot_id,
            nodes=(
                GraphNode(key=entity_key, labels=("Service",), properties={"name": "web"}),
                GraphNode(key="service:api", labels=("Service",)),
            ),
            edges=(
                GraphEdge(
                    predicate="DEPENDS_ON",
                    from_key=entity_key,
                    to_key="service:api",
                    properties={"environment": "staging"},
                ),
            ),
        )


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


def _proposal(
    *,
    ok: bool = True,
    status: str = "validated",
    risk: str = "low",
    plan_id: str = "mutation-plan:test",
    issues: tuple[dict, ...] = (),
) -> GraphMutationProposal:
    return GraphMutationProposal(
        ok=ok,
        plan_id=plan_id,
        status=status,
        risk=risk,
        pot_id="p",
        auto_applicable=status == "validated" and risk == "low",
        expires_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
        expected_subgraph_versions={"_global": 0},
        current_subgraph_versions={"_global": 0},
        diff=GraphMutationDiff(edge_upserts=1, claim_keys=("claim:test",)),
        issues=issues,
        rejected_operations=issues,
        claim_keys=("claim:test",),
    )


def _commit_result(
    *,
    ok: bool = True,
    status: str = "committed",
    plan_id: str = "mutation-plan:test",
) -> GraphMutationCommitResult:
    return GraphMutationCommitResult(
        ok=ok,
        plan_id=plan_id,
        status=status,
        risk="low",
        pot_id="p",
        mutation_id="mutation-1" if ok else None,
        applied_at=datetime(2026, 6, 15, tzinfo=timezone.utc) if ok else None,
        expected_subgraph_versions={"_global": 0},
        current_subgraph_versions={"_global": 0},
        new_subgraph_versions={"_global": 1} if ok else {"_global": 0},
        diff=GraphMutationDiff(edge_upserts=1, claim_keys=("claim:test",)),
        claim_keys=("claim:test",),
    )


def _assert_graph_envelope(payload: dict, command: str, *, ok: bool = True) -> dict:
    assert payload["ok"] is ok
    assert payload["command"] == command
    assert payload["request_id"].startswith("req:")
    assert payload["pot_id"] in {"p", None}
    assert payload["graph_contract_version"] == "v2"
    assert payload["ontology_version"] == "2026-06-graph"
    assert "subgraph_versions" in payload
    assert "warnings" in payload
    assert "unsupported" in payload
    assert "recommended_next_action" in payload
    return payload["result"] or {}


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
        (["neighborhood", "--entity", "service:web"], "inspection", "neighborhood"),
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
    _assert_graph_envelope(emitted, f"graph.{args[0]}", ok=False)
    assert emitted["error"]["code"] == "not_implemented"
    assert f"graph.neo4j.{capability}.{method}" in emitted["error"]["message"]
    assert emitted["recommended_next_action"]
    assert backend.accessed_ports == []


def test_graph_mutate_rejection_emits_result_and_exits_nonzero(tmp_path) -> None:
    _common.set_json(True)
    proposal = _proposal(
        ok=False,
        status="invalid",
        issues=(
            {
                "code": "invalid_endpoints",
                "message": "invalid endpoint pair",
                "severity": "error",
                "op_index": 0,
            },
        ),
    )
    workbench = _Workbench(proposal=proposal)
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        ["mutate", "--file", str(payload_file)],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    detail = _assert_graph_envelope(emitted, "graph.mutate", ok=False)
    assert detail == {}
    assert emitted["error"]["code"] == "invalid_endpoints"
    assert emitted["error"]["detail"]["status"] == "invalid"
    assert emitted["error"]["detail"]["issues"][0]["code"] == "invalid_endpoints"
    assert "legacy transition command" in emitted["warnings"][0]
    assert workbench.commit_calls == []


def test_graph_propose_returns_persisted_plan_envelope(tmp_path) -> None:
    _common.set_json(True)
    workbench = _Workbench(proposal=_proposal())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        ["propose", "--file", str(payload_file), "--ttl", "30m"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.propose")
    assert body["plan_id"] == "mutation-plan:test"
    assert body["status"] == "validated"
    assert workbench.propose_calls[0][1:] == ("p", 1800)


def test_graph_commit_applies_plan_id_only() -> None:
    _common.set_json(True)
    workbench = _Workbench(commit_result=_commit_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert body["mutation_id"] == "mutation-1"
    assert workbench.commit_calls == [("mutation-plan:test", "p", None)]


def test_graph_commit_rejects_raw_payload_option() -> None:
    _common.set_json(True)
    workbench = _Workbench(commit_result=_commit_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--file", "mutation.json"],
    )

    assert result.exit_code != 0
    assert workbench.commit_calls == []


@pytest.mark.parametrize("scope", ["service", "service:"])
def test_graph_read_rejects_malformed_scope_before_service_call(scope: str) -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "debugging",
            "--view",
            "prior_occurrences",
            "--scope",
            scope,
        ],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is False
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "invalid --scope entry" in emitted["error"]["message"]


def test_graph_read_unknown_view_uses_error_envelope() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_error=ValueError("unknown graph view 'missing.view'"))
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "missing", "--view", "view"],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is True
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph view" in emitted["error"]["message"]


def test_graph_read_rejects_fully_qualified_view_before_service_call() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "debugging", "--view", "debugging.prior_occurrences"],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is False
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "--subgraph <name> --view <view>" in emitted["error"]["message"]


def _timeline_env() -> GraphReadResult:
    return GraphReadResult(
        graph_contract_version="v1.5",
        ontology_version="2026-06-graph",
        view="recent_changes.timeline",
        subgraph="recent_changes",
        read_shape="entity_relations",
        coverage=({"include": "timeline", "status": "complete"},),
        freshness={"local_worktree_included": False},
        quality={"status": "ok"},
        items=(
            {
                "entity_key": "activity:github:pr-2",
                "entity_type": "Activity",
                "score": 0.9,
                "summary": 'PR #2 "newer" was merged into acme/widgets.',
                "source_refs": ["github:pr:2"],
                "truth": "timeline_event",
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
            {
                "entity_key": "activity:github:pr-1",
                "entity_type": "Activity",
                "score": 1.0,
                "summary": 'PR #1 "older" was merged into acme/widgets.',
                "source_refs": ["github:pr:1"],
                "truth": "timeline_event",
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
    )


def test_graph_read_timeline_defaults_to_deduped_event_json() -> None:
    _common.set_json(True)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "read",
            "--subgraph",
            "recent_changes",
            "--view",
            "timeline",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.limit == 40
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert "subgraph_versions" not in body
    assert "unsupported" not in body
    assert body["read_shape"] == "events"
    assert body["event_count"] == 1
    assert body["events"][0]["source_refs"] == ["github:pr:2"]
    assert body["freshness"]["local_worktree_included"] is False


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
    body = _assert_graph_envelope(emitted, "graph.nudge")
    assert body["event"] == "pre_edit"
    assert "legacy transition command" in emitted["warnings"][0]


def test_graph_catalog_json_advertises_v2_workbench_commands() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(graph.graph_app, ["catalog"])

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.catalog")
    assert "propose" in body["commands"]
    assert "commit" in body["commands"]
    assert "mutate" not in body["commands"]
    assert "mutate" in body["legacy_commands"]
    assert body["data_plane_graph_contract_version"] == "v1.5"


def test_graph_catalog_task_ranks_relevant_views() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["catalog", "--task", "debug staging timeout after deployment"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.catalog")
    ranking = body["task_ranking"]
    ranked_subgraphs = [entry["subgraph"] for entry in ranking]
    assert ranked_subgraphs.index("debugging") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("recent_changes") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("infra_topology") < ranked_subgraphs.index("features")
    assert ranked_subgraphs.index("decisions") < ranked_subgraphs.index("features")
    assert body["views"][0]["name"] == ranking[0]["view"]


def test_graph_catalog_unknown_subgraph_uses_error_envelope() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph(catalog_error=ValueError("unknown graph subgraph"))))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--subgraph", "missing"])

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.catalog", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph subgraph" in emitted["error"]["message"]


def test_graph_status_json_uses_workbench_envelope() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph(), backend=_Backend()))

    result = CliRunner().invoke(graph.graph_app, ["status"])

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.status")
    assert emitted["subgraph_versions"] == {"_global": 3}
    assert body["graph_service"]["supported_commands"][0] == "status"
    assert body["backend"]["profile"] == "memory"


def test_graph_neighborhood_returns_inspection_slice() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph(), backend=_Backend()))

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
        ],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.neighborhood")
    assert body["entity_key"] == "service:web"
    assert body["predicates"] == ["DEPENDS_ON"]
    assert body["edges"][0]["from"] == "service:web"


def test_graph_describe_returns_executable_view_contract() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["describe", "debugging", "--view", "prior_occurrences", "--examples"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.describe")
    assert body["contract_kind"] == "graph_workbench_ontology"
    assert body["view"]["name"] == "debugging.prior_occurrences"
    assert body["view"]["result_shape"] == "entity_relations"
    assert "REPRODUCES" in body["view"]["inline_relations"]
    assert body["view"]["examples"][0]["command"].startswith("potpie graph read")


def test_graph_describe_unknown_view_uses_error_envelope() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["describe", "debugging", "--view", "missing"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.describe", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph view" in emitted["error"]["message"]


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
    assert req.subgraph == "recent_changes"
    assert req.view == "timeline"
    assert req.scope == {}
    assert req.limit == 40
    assert req.since is not None
    emitted = json.loads(result.output)
    assert emitted["event_count"] == 2
