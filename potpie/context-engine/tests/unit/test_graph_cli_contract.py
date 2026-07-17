"""CLI contract coverage for Graph Surface Lite."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
import re

import pytest
from typer.testing import CliRunner

from potpie_context_engine.bootstrap import observability_runtime
from potpie_context_engine.adapters.inbound.cli.commands import _common, graph
from potpie_context_engine.adapters.inbound.cli.telemetry import product_analytics
from potpie_context_engine.adapters.inbound.cli.telemetry.context import TelemetryContext
from potpie_context_engine.domain.graph_plans import (
    GraphIngestionVerificationResult,
    GraphMutationCommitResult,
    GraphMutationDiff,
    GraphMutationProposal,
)
from potpie_context_engine.domain.graph_history import GraphHistoryEntry, GraphHistoryResult
from potpie_context_engine.domain.graph_inbox import GraphInboxItem, GraphInboxResult
from potpie_context_engine.domain.graph_quality import GraphQualityFinding, GraphQualityResult
from potpie_context_engine.domain.nudge import GraphNudgeResult
from potpie_context_engine.domain.graph_views import views_for_catalog
from potpie_context_engine.domain.ports.services.graph_service import (
    DataPlaneStatus,
    GraphCatalogResult,
    GraphEntityCandidate,
    GraphEntitySearchResult,
    GraphReadResult,
)
from potpie_context_engine.domain.ports.graph.inspection import GraphEdge, GraphNode, GraphSlice
from potpie_context_engine.domain.ports.graph.analytics import RepairReport
from potpie_context_engine.domain.ports.graph.backend import BackendCapabilities

pytestmark = pytest.mark.unit

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_cli_output(output: str) -> str:
    return _ANSI_RE.sub("", output)


@pytest.fixture(autouse=True)
def _reset_json_mode():
    yield
    _common.set_json(False)


class _Pot:
    pot_id = "p"
    name = "default"
    active = True


class _Pots:
    def active_pot(self):
        return _Pot()

    def list_pots(self):
        return [_Pot()]

    def list_sources(self, *, pot_id):
        return []


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
        self.search_request = None
        self.describe_request = None

    def describe(self, request):
        # Delegate to the real domain contract so payload assertions stay
        # meaningful; the stub only stands in for the transport.
        from potpie_context_engine.domain.graph_workbench_ontology import describe_contract

        self.describe_request = request
        return describe_contract(
            subgraph=request.subgraph,
            view=request.view,
            include_examples=request.include_examples,
        )

    def catalog(self, _request):
        if self.catalog_error is not None:
            raise self.catalog_error
        return GraphCatalogResult(
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
            commands=("catalog", "read", "search-entities", "mutate"),
            truth_classes=("agent_claim",),
            mutation_operations=(
                "link_entities",
                "patch_entity",
                "transition_state",
                "supersede_claim",
                "merge_duplicate_entities",
            ),
            review_required_operations=(),
            deferred_operations=(),
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
        return replace(
            self.read_result,
            detail=_request.detail,
            relations=_request.relations,
        )

    def search_entities(self, request):
        self.search_request = request
        supporting_claims = (
            (
                {
                    "claim_key": "claim:widgets-payments",
                    "predicate": "PROVIDES",
                },
            )
            if request.supporting_claims
            else ()
        )
        return GraphEntitySearchResult(
            entities=(
                GraphEntityCandidate(
                    key="feature:payments",
                    labels=("Feature",),
                    summary="Payments capability",
                    score=0.9,
                    supporting_claims=supporting_claims,
                ),
            ),
            match_mode="lexical",
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
        )


class _NotReadyGraph(_Graph):
    def data_plane_status(self, pot_id):
        return DataPlaneStatus(
            pot_id=pot_id,
            backend_profile="memory",
            backend_ready=False,
            reader_backed_includes=("timeline",),
            counts={"claims": 0},
            freshness={},
            quality={"status": "unavailable"},
            match_mode="lexical",
            detail="mutation store is unavailable",
        )


class _GraphByPot(_Graph):
    def __init__(self, counts_by_pot):
        super().__init__()
        self.counts_by_pot = counts_by_pot

    def data_plane_status(self, pot_id):
        return DataPlaneStatus(
            pot_id=pot_id,
            backend_profile="memory",
            backend_ready=True,
            reader_backed_includes=("timeline",),
            counts=self.counts_by_pot.get(pot_id, {"claims": 0, "entities": 0}),
            freshness={},
            quality={},
            match_mode="lexical",
        )


class _RecordedSpan:
    def __init__(self, attrs: dict) -> None:
        self.attrs = attrs
        self.error = None

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_attributes(self, attributes):
        self.attrs.update(dict(attributes))

    def add_event(self, *_args, **_kwargs):
        pass

    def record_exception(self, exc):
        self.attrs["exception"] = repr(exc)

    def set_error(self, message=None):
        self.error = message or True


class _RecordingObservability:
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict, _RecordedSpan]] = []
        self.counters: list[tuple[str, int, dict]] = []
        self.histograms: list[tuple[str, float, dict]] = []

    @contextmanager
    def span(self, name, *, kind="internal", attributes=None, links=None):
        del links
        attrs = {"kind": kind, **dict(attributes or {})}
        span = _RecordedSpan(attrs)
        self.spans.append((name, attrs, span))
        yield span

    def current_traceparent(self):
        return None

    @contextmanager
    def baggage(self, **_items):
        yield

    def counter(self, name, value=1, *, attributes=None):
        self.counters.append((name, value, dict(attributes or {})))

    def histogram(self, name, value, *, attributes=None):
        self.histograms.append((name, value, dict(attributes or {})))

    def gauge(self, name, value, *, attributes=None):
        del name, value, attributes


class _ProductAnalyticsSink:
    def __init__(self) -> None:
        self.events = []

    def capture(self, event) -> None:
        self.events.append(event)


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
        history_result: GraphHistoryResult | None = None,
        inbox_result: GraphInboxResult | None = None,
        quality_result: GraphQualityResult | None = None,
    ) -> None:
        self.proposal = proposal
        self.commit_result = commit_result
        self.history_result = history_result
        self.inbox_result = inbox_result
        self.quality_result = quality_result
        self.propose_calls = []
        self.commit_calls = []
        self.history_calls = []
        self.inbox_calls = []
        self.quality_calls = []

    def propose(self, payload, *, pot_id, ttl_seconds=None):
        self.propose_calls.append((payload, pot_id, ttl_seconds))
        if self.proposal is None:
            raise AssertionError("propose should not be called")
        return self.proposal

    def commit(self, plan_id, *, pot_id, approved_by=None, verify=False):
        self.commit_calls.append((plan_id, pot_id, approved_by, verify))
        if self.commit_result is None:
            raise AssertionError("commit should not be called")
        return self.commit_result

    def history(self, **kwargs):
        self.history_calls.append(kwargs)
        if self.history_result is None:
            raise AssertionError("history should not be called")
        return self.history_result

    def inbox_add(self, **kwargs):
        self.inbox_calls.append(("add", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_add should not be called")
        return self.inbox_result

    def inbox_list(self, **kwargs):
        self.inbox_calls.append(("list", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_list should not be called")
        return self.inbox_result

    def inbox_show(self, **kwargs):
        self.inbox_calls.append(("show", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_show should not be called")
        return self.inbox_result

    def inbox_claim(self, **kwargs):
        self.inbox_calls.append(("claim", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_claim should not be called")
        return self.inbox_result

    def inbox_mark_applied(self, **kwargs):
        self.inbox_calls.append(("mark-applied", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_mark_applied should not be called")
        return self.inbox_result

    def inbox_mark_rejected(self, **kwargs):
        self.inbox_calls.append(("mark-rejected", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_mark_rejected should not be called")
        return self.inbox_result

    def inbox_close(self, **kwargs):
        self.inbox_calls.append(("close", kwargs))
        if self.inbox_result is None:
            raise AssertionError("inbox_close should not be called")
        return self.inbox_result

    def quality(self, **kwargs):
        self.quality_calls.append(kwargs)
        if self.quality_result is None:
            raise AssertionError("quality should not be called")
        return self.quality_result


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
                GraphNode(
                    key=entity_key, labels=("Service",), properties={"name": "web"}
                ),
                GraphNode(key="service:api", labels=("Service",)),
            ),
            edges=(
                GraphEdge(
                    predicate="DEPENDS_ON",
                    from_key=entity_key,
                    to_key="service:api",
                    properties={
                        "fact": "web depends on api",
                        "source_refs": ["repo:manifest"],
                        "truth": "source_observation",
                        "environment": "staging",
                    },
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


def _bind_graph_product_analytics(
    monkeypatch: pytest.MonkeyPatch,
) -> _ProductAnalyticsSink:
    sink = _ProductAnalyticsSink()
    monkeypatch.setattr(product_analytics, "_sink", sink)
    monkeypatch.setattr(
        product_analytics,
        "current_telemetry_context",
        lambda: TelemetryContext(
            anonymous_install_id="install_graph",
            invocation_id="invoke_graph",
            daemon_session_id="daemon_graph",
            environment="test",
            command="graph",
            subcommand=None,
            output_mode="json",
            cli_version="0.1.0",
            python_version="3.13.0",
            os="darwin",
            arch="arm64",
        ),
    )
    return sink


def _bulk_mutation_payload(count: int = 3) -> dict:
    base = _valid_mutation_payload()["operations"][0]
    return {
        "idempotency_key": "bulk:test",
        "created_by": {"surface": "cli", "harness": "test"},
        "operations": [
            {
                **base,
                "object": {"key": f"service:ledger-api-{index}", "type": "Service"},
                "description": f"payments depends on ledger {index}",
            }
            for index in range(count)
        ],
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
    verification: GraphIngestionVerificationResult | None = None,
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
        verification=verification,
    )


def _history_result() -> GraphHistoryResult:
    return GraphHistoryResult(
        ok=True,
        pot_id="p",
        filters={"plan": "mutation-plan:test", "limit": 50},
        entries=(
            GraphHistoryEntry(
                kind="plan",
                id="mutation-plan:test",
                status="committed",
                plan_id="mutation-plan:test",
                mutation_id="mutation-1",
                occurred_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
                source_refs=("repo:manifest",),
                summary="mutation plan committed",
            ),
        ),
        subgraph_versions={"_global": 1},
    )


def _inbox_result(
    *,
    action: str = "add",
    status: str = "pending",
    ok: bool = True,
) -> GraphInboxResult:
    item = GraphInboxItem(
        item_id="graph-inbox:test",
        pot_id="p",
        status=status,
        summary="Possible graph update",
        evidence=("github:pr:955",),
        source_refs=("github:pr:955",),
        suspected_subgraphs=("debugging",),
        created_by={"surface": "cli", "actor": "codex"},
        created_at=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )
    return GraphInboxResult(
        ok=ok,
        pot_id="p",
        action=action,
        item=item if action != "list" else None,
        items=(item,) if action == "list" else (),
        filters={"limit": 50} if action == "list" else {},
        detail=None if ok else "inbox item could not be changed",
    )


def _quality_result(
    *,
    report: str = "summary",
    status: str = "ok",
) -> GraphQualityResult:
    finding = GraphQualityFinding(
        finding_id="quality:low-confidence:test",
        kind="low-confidence",
        severity="warning",
        summary="claim needs stronger evidence",
        claim_keys=("claim:test",),
        entity_keys=("service:web", "repo:github.com/acme/web"),
        predicates=("DEFINED_IN",),
        source_refs=("repo:manifest",),
    )
    return GraphQualityResult(
        ok=True,
        pot_id="p",
        report=report,
        status=status,
        findings=() if report == "summary" else (finding,),
        metrics=(
            {
                "counts": {"claims": 3},
                "quality_counts": {
                    "duplicate_candidates": 0,
                    "stale_facts": 0,
                    "conflicting_claims": 1,
                    "orphan_entities": 0,
                    "low_confidence": 0,
                    "projection_drift": 2,
                },
                "quality_reports": {
                    "conflicting-claims": {
                        "status": status,
                        "finding_count": 1,
                        "severity_counts": {"warning": 1},
                    },
                    "projection-drift": {
                        "status": status,
                        "finding_count": 2,
                        "severity_counts": {"error": 2},
                    },
                },
                "total_findings": 3,
            }
            if report == "summary"
            else {"scanned_claims": 3}
        ),
        filters={"report": report, "limit": 50},
        subgraph_versions={"_global": 3},
    )


def _assert_graph_envelope(
    payload: dict, command: str, *, ok: bool = True, pot_id: str | None = "p"
) -> dict:
    assert payload["ok"] is ok
    assert payload["command"] == command
    assert payload["request_id"].startswith("req:")
    assert payload["pot_id"] in {pot_id, None}
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
    ("command", "required_marker", "optional_marker", "missing_message"),
    [
        (
            "neighborhood",
            "--entity",
            None,
            "Missing option '--entity'",
        ),
        (
            "inspect",
            "ENTITY_KEY",
            "[ENTITY_KEY]",
            "Missing argument 'ENTITY_KEY'",
        ),
        (
            "export",
            "FILE",
            "[FILE]",
            "Missing argument 'FILE'",
        ),
        (
            "import",
            "FILE",
            "[FILE]",
            "Missing argument 'FILE'",
        ),
    ],
)
def test_graph_required_inputs_are_declared_in_help(
    command: str,
    required_marker: str,
    optional_marker: str | None,
    missing_message: str,
) -> None:
    help_result = CliRunner().invoke(graph.graph_app, [command, "--help"])

    assert help_result.exit_code == 0, help_result.output
    help_output = _plain_cli_output(help_result.output)
    assert required_marker in help_output
    assert "[required]" in help_output
    if optional_marker is not None:
        assert optional_marker not in help_output

    missing_result = CliRunner().invoke(graph.graph_app, [command])
    missing_output = _plain_cli_output(missing_result.output)

    assert missing_result.exit_code == 2
    assert missing_message in missing_output


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
        "potpie_context_engine.bootstrap.sentry_metrics_runtime.flush", lambda timeout=2.0: None
    )
    sink = _bind_graph_product_analytics(monkeypatch)
    payload_file = tmp_path / "mutation.json"
    payload_file.write_text(json.dumps(_valid_mutation_payload()), encoding="utf-8")

    _common.set_host(_Host(_Graph(), graph_workbench=_Workbench(proposal=_proposal())))
    propose = CliRunner().invoke(
        graph.graph_app,
        ["propose", "--file", str(payload_file)],
    )
    assert propose.exit_code == 0, propose.output

    _common.set_host(
        _Host(_Graph(), graph_workbench=_Workbench(inbox_result=_inbox_result()))
    )
    inbox = CliRunner().invoke(
        graph.graph_app,
        ["inbox", "add", "--summary", "Possible graph update"],
    )
    assert inbox.exit_code == 0, inbox.output

    _common.set_host(
        _Host(
            _Graph(),
            graph_workbench=_Workbench(
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
    assert workbench.commit_calls == [("mutation-plan:test", "p", None, False)]


def test_graph_commit_verify_passes_hard_gate_flag() -> None:
    _common.set_json(True)
    workbench = _Workbench(commit_result=_commit_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--verify"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert workbench.commit_calls == [("mutation-plan:test", "p", None, True)]


def test_graph_commit_verify_exits_nonzero_when_gate_fails() -> None:
    _common.set_json(True)
    verification = GraphIngestionVerificationResult(
        ok=False,
        status="degraded",
        plan_id="mutation-plan:test",
        pot_id="p",
        claim_keys=("claim:test",),
        missing_claim_keys=("claim:test",),
        quality_status="ok",
        detail="committed plan did not read back all expected claim keys",
        recommended_next_action="Inspect graph history for the plan.",
    )
    workbench = _Workbench(commit_result=_commit_result(verification=verification))
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["commit", "mutation-plan:test", "--verify"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.commit")
    assert body["status"] == "committed"
    assert body["verification"]["ok"] is False
    assert body["verification"]["missing_claim_keys"] == ["claim:test"]


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


def test_graph_bulk_apply_chunks_and_commits(tmp_path) -> None:
    _common.set_json(True)
    workbench = _Workbench(proposal=_proposal(), commit_result=_commit_result())
    graph_service = _Graph()
    _common.set_host(_Host(graph_service, graph_workbench=workbench))
    payload_file = tmp_path / "bulk.json"
    manifest_file = tmp_path / "manifest.json"
    payload_file.write_text(json.dumps(_bulk_mutation_payload(3)), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "bulk",
            "apply",
            "--file",
            str(payload_file),
            "--chunk-size",
            "2",
            "--manifest",
            str(manifest_file),
            "--verify",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.bulk.apply")
    assert body["status"] == "committed"
    assert body["chunks_total"] == 2
    assert body["chunks_attempted"] == 2
    assert body["chunks_committed"] == 2
    assert body["operations_committed"] == 3
    assert body["verification"]["counts"] == {"claims": 3}
    assert len(workbench.propose_calls) == 2
    assert len(workbench.commit_calls) == 2
    assert len(workbench.propose_calls[0][0]["operations"]) == 2
    assert len(workbench.propose_calls[1][0]["operations"]) == 1
    assert workbench.propose_calls[0][0]["idempotency_key"] == "bulk:test:chunk-0001"
    assert workbench.propose_calls[1][0]["idempotency_key"] == "bulk:test:chunk-0002"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["status"] == "committed"
    assert manifest["chunks_committed"] == 2


def test_graph_bulk_apply_dry_run_does_not_commit(tmp_path) -> None:
    _common.set_json(True)
    workbench = _Workbench(proposal=_proposal(), commit_result=_commit_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "bulk.ndjson"
    lines = [json.dumps(op) for op in _bulk_mutation_payload(2)["operations"]]
    payload_file.write_text("\n".join(lines), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "bulk",
            "apply",
            "--file",
            str(payload_file),
            "--chunk-size",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.bulk.apply")
    assert body["status"] == "validated"
    assert body["chunks_validated"] == 2
    assert body["operations_validated"] == 2
    assert workbench.commit_calls == []


def test_graph_bulk_apply_stops_on_failed_proposal(tmp_path) -> None:
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
    workbench = _Workbench(proposal=proposal, commit_result=_commit_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))
    payload_file = tmp_path / "bulk.json"
    payload_file.write_text(json.dumps(_bulk_mutation_payload(3)), encoding="utf-8")

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "bulk",
            "apply",
            "--file",
            str(payload_file),
            "--chunk-size",
            "1",
        ],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.bulk.apply", ok=False)
    assert body == {}
    assert emitted["error"]["code"] == "invalid_endpoints"
    assert emitted["error"]["detail"]["status"] == "failed"
    assert emitted["error"]["detail"]["chunks_attempted"] == 1
    assert workbench.commit_calls == []


def test_graph_history_plan_returns_envelope() -> None:
    _common.set_json(True)
    workbench = _Workbench(history_result=_history_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["history", "--plan", "mutation-plan:test"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.history")
    assert emitted["subgraph_versions"] == {"_global": 1}
    assert body["entry_count"] == 1
    assert body["entries"][0]["plan_id"] == "mutation-plan:test"
    assert workbench.history_calls == [
        {
            "pot_id": "p",
            "entity_key": None,
            "claim_key": None,
            "subgraph": None,
            "plan_id": "mutation-plan:test",
            "mutation_id": None,
            "since": None,
            "until": None,
            "limit": 50,
        }
    ]


def test_graph_quality_summary_returns_envelope() -> None:
    _common.set_json(True)
    workbench = _Workbench(quality_result=_quality_result(report="summary"))
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

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
    workbench = _Workbench(
        quality_result=_quality_result(report="low-confidence", status="watch")
    )
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

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


def test_graph_inbox_add_returns_pending_item_envelope() -> None:
    _common.set_json(True)
    workbench = _Workbench(inbox_result=_inbox_result())
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "add",
            "--summary",
            "Possible graph update",
            "--evidence",
            "github:pr:955",
            "--source-ref",
            "github:pr:955",
            "--subgraph",
            "debugging",
            "--created-by",
            "codex",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.add")
    assert body["action"] == "add"
    assert body["item"]["item_id"] == "graph-inbox:test"
    assert body["item"]["status"] == "pending"
    assert workbench.inbox_calls == [
        (
            "add",
            {
                "pot_id": "p",
                "summary": "Possible graph update",
                "details": None,
                "evidence": ("github:pr:955",),
                "source_refs": ("github:pr:955",),
                "suspected_subgraphs": ("debugging",),
                "created_by": {"surface": "cli", "actor": "codex"},
            },
        )
    ]


def test_graph_inbox_list_passes_filters() -> None:
    _common.set_json(True)
    workbench = _Workbench(inbox_result=_inbox_result(action="list"))
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "list",
            "--status",
            "pending,claimed",
            "--claimed-by",
            "user:alice",
            "--subgraph",
            "debugging",
            "--source-ref",
            "github:pr:955",
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.list")
    assert body["item_count"] == 1
    assert workbench.inbox_calls[0] == (
        "list",
        {
            "pot_id": "p",
            "status": ("pending,claimed",),
            "claimed_by": "user:alice",
            "suspected_subgraph": "debugging",
            "source_ref": "github:pr:955",
            "since": None,
            "until": None,
            "limit": 10,
        },
    )


def test_graph_inbox_claim_passes_actor() -> None:
    _common.set_json(True)
    workbench = _Workbench(inbox_result=_inbox_result(action="claim", status="claimed"))
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        ["inbox", "claim", "graph-inbox:test", "--by", "user:alice"],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.claim")
    assert body["item"]["status"] == "claimed"
    assert workbench.inbox_calls == [
        (
            "claim",
            {
                "pot_id": "p",
                "item_id": "graph-inbox:test",
                "claimed_by": "user:alice",
            },
        )
    ]


def test_graph_inbox_mark_applied_passes_plan_and_mutation() -> None:
    _common.set_json(True)
    workbench = _Workbench(
        inbox_result=_inbox_result(action="mark-applied", status="applied")
    )
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "mark-applied",
            "graph-inbox:test",
            "--plan",
            "mutation-plan:test",
            "--mutation",
            "mutation-1",
            "--by",
            "user:alice",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.inbox.mark-applied")
    assert body["item"]["status"] == "applied"
    assert workbench.inbox_calls == [
        (
            "mark-applied",
            {
                "pot_id": "p",
                "item_id": "graph-inbox:test",
                "closed_by": "user:alice",
                "linked_plan_id": "mutation-plan:test",
                "linked_mutation_id": "mutation-1",
            },
        )
    ]


def test_graph_inbox_mark_rejected_and_close_record_reasons() -> None:
    _common.set_json(True)
    workbench = _Workbench(
        inbox_result=_inbox_result(action="mark-rejected", status="rejected")
    )
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))

    rejected = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "mark-rejected",
            "graph-inbox:test",
            "--reason",
            "not enough evidence",
            "--by",
            "user:alice",
        ],
    )

    assert rejected.exit_code == 0, rejected.output
    emitted = json.loads(rejected.output)
    _assert_graph_envelope(emitted, "graph.inbox.mark-rejected")
    assert workbench.inbox_calls == [
        (
            "mark-rejected",
            {
                "pot_id": "p",
                "item_id": "graph-inbox:test",
                "closed_by": "user:alice",
                "rejection_reason": "not enough evidence",
            },
        )
    ]

    workbench = _Workbench(inbox_result=_inbox_result(action="close", status="closed"))
    _common.set_host(_Host(_Graph(), graph_workbench=workbench))
    closed = CliRunner().invoke(
        graph.graph_app,
        [
            "inbox",
            "close",
            "graph-inbox:test",
            "--reason",
            "superseded",
            "--by",
            "user:alice",
        ],
    )

    assert closed.exit_code == 0, closed.output
    emitted = json.loads(closed.output)
    _assert_graph_envelope(emitted, "graph.inbox.close")
    assert workbench.inbox_calls[0][1]["rejection_reason"] == "superseded"


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


def test_graph_read_missing_required_scope_result_is_error_envelope() -> None:
    _common.set_json(True)
    graph_service = _Graph(
        read_result=GraphReadResult(
            graph_contract_version="v1.5",
            ontology_version="2026-06-graph",
            view="features.feature_context",
            subgraph="features",
            ok=False,
            status="missing_required_scope",
            message=(
                "graph read view 'features.feature_context' requires one of "
                "scope, service, repo, anchor_entity_key, query"
            ),
            coverage=(
                {
                    "view": "features.feature_context",
                    "status": "unsupported",
                    "candidate_pool": 0,
                },
            ),
            quality={"status": "unsupported", "reason": "missing_required_scope"},
            unsupported=(
                {
                    "name": "features.feature_context",
                    "reason": "missing_required_scope",
                },
            ),
        )
    )
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "features", "--view", "feature_context"],
    )

    assert result.exit_code == 1
    assert graph_service.read_called is True
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "missing_required_scope"
    assert emitted["unsupported"][0]["reason"] == "missing_required_scope"
    assert emitted["error"]["detail"]["quality"]["reason"] == "missing_required_scope"


def test_graph_read_include_guess_error_carries_did_you_mean() -> None:
    # Audit item 17: a failed include-family guess returns machine-readable
    # migration guidance in the error envelope (never accepted as input).
    from potpie_context_engine.domain.graph_views import UnknownGraphViewError, include_guess_guidance

    _common.set_json(True)
    guidance = include_guess_guidance("docs", "relevant")
    graph_service = _Graph(
        read_error=UnknownGraphViewError(
            "unknown graph view 'docs.relevant'",
            did_you_mean=guidance,
            recommended_next_action=guidance["read_command"],
        )
    )
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "docs", "--view", "relevant"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.read", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    did_you_mean = emitted["error"]["detail"]["did_you_mean"]
    assert did_you_mean["view"] == "knowledge.document_context"
    assert did_you_mean["matched_include"] == "docs"
    assert emitted["recommended_next_action"] == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


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
        coverage=({"view": "recent_changes.timeline", "status": "complete"},),
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
    assert "items" not in body


def test_graph_read_threads_source_ref_filter() -> None:
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
            "--source-ref",
            "github:pr:2",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.source_refs == ("github:pr:2",)


def test_graph_read_raw_json_defaults_to_compact_relations() -> None:
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
            "--format",
            "raw",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.detail == "compact"
    assert graph_service.read_request.relations == "summary"
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert body["detail"] == "compact"
    assert body["relations_detail"] == "summary"
    assert "relations" not in body["items"][0]
    assert body["items"][0]["relation_count"] == 2


def test_graph_read_full_detail_preserves_relation_payload() -> None:
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
            "--format",
            "raw",
            "--detail",
            "full",
            "--relations",
            "full",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.read_request.detail == "full"
    assert graph_service.read_request.relations == "full"
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.read")
    assert body["detail"] == "full"
    assert body["items"][0]["relations"][0]["predicate"] == "TOUCHED"


def test_graph_search_entities_omits_supporting_claims_by_default() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["search-entities", "payments"],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.supporting_claims == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.search-entities")
    assert body["entities"][0]["supporting_claims"] == []


def test_graph_search_entities_supporting_claims_is_opt_in() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["search-entities", "payments", "--supporting-claims", "1"],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.supporting_claims == 1
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.search-entities")
    assert body["entities"][0]["supporting_claims"][0]["predicate"] == "PROVIDES"


def test_graph_search_entities_threads_source_ref_filter() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["search-entities", "payments", "--source-ref", "github:pr:955"],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.source_refs == ("github:pr:955",)


def test_graph_search_entities_threads_source_facets() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "search-entities",
            "payments",
            "--source-system",
            "github",
            "--source-family",
            "github",
        ],
    )

    assert result.exit_code == 0
    assert graph_service.search_request.source_system == "github"
    assert graph_service.search_request.source_family == "github"


def test_graph_read_emits_v2_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    _common.set_json(True)
    obs = _RecordingObservability()
    monkeypatch.setattr(observability_runtime, "_OBSERVABILITY", obs)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(graph_service))

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
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "recent_changes"],
    )

    assert result.exit_code == 1
    assert [
        event for event in sink.events if event.name == "cli_usage_command_succeeded"
    ] == []


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
    assert "reason" in ranking[0]


def test_graph_catalog_read_profile_returns_compact_contract() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["catalog", "--task", "debug staging timeout", "--profile", "read"],
    )

    assert result.exit_code == 0
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.catalog")
    assert body["profile"] == "read"
    assert "read" in body["commands"]
    assert "commit" not in body["commands"]
    assert "mutation_operations" not in body
    assert "entity_types" not in body
    assert body["views"]
    assert set(body["views"][0]) <= {
        "name",
        "subgraph",
        "view",
        "backed",
        "description",
        "result_shape",
        "required_scope",
        "required_any_scope",
        "supported_filters",
        "next_read",
    }
    assert body["task_ranking"][0]["rank"] == 1
    assert "reason" in body["task_ranking"][0]
    assert "matched_terms" in body["task_ranking"][0]


def test_graph_catalog_table_format_uses_compact_human_output() -> None:
    _common.set_json(False)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        ["catalog", "--profile", "read", "--format", "table"],
    )

    assert result.exit_code == 0
    assert "graph catalog profile=read" in result.output
    assert "view | backed | filters" in result.output


def test_graph_catalog_table_format_shows_task_ranking_context() -> None:
    _common.set_json(False)
    _common.set_host(_Host(_Graph()))

    result = CliRunner().invoke(
        graph.graph_app,
        [
            "catalog",
            "--task",
            "debug staging timeout",
            "--profile",
            "read",
            "--format",
            "table",
        ],
    )

    assert result.exit_code == 0
    assert "rank | score | view | reason" in result.output
    assert "Matched" in result.output


def test_graph_catalog_unknown_subgraph_uses_error_envelope() -> None:
    _common.set_json(True)
    _common.set_host(_Host(_Graph(catalog_error=ValueError("unknown graph subgraph"))))

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--subgraph", "missing"])

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.catalog", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    assert "unknown graph subgraph" in emitted["error"]["message"]


def test_graph_catalog_include_guess_error_carries_did_you_mean() -> None:
    # Audit item 17 first-contact path: an include family typed where a
    # subgraph is expected gets the same migration guidance as read.
    from potpie_context_engine.domain.graph_views import UnknownGraphViewError, include_guess_guidance

    _common.set_json(True)
    guidance = include_guess_guidance("docs", None)
    _common.set_host(
        _Host(
            _Graph(
                catalog_error=UnknownGraphViewError(
                    "unknown graph subgraph 'docs'",
                    did_you_mean=guidance,
                    recommended_next_action=guidance["read_command"],
                )
            )
        )
    )

    result = CliRunner().invoke(graph.graph_app, ["catalog", "--subgraph", "docs"])

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    _assert_graph_envelope(emitted, "graph.catalog", ok=False)
    assert emitted["error"]["code"] == "validation_error"
    did_you_mean = emitted["error"]["detail"]["did_you_mean"]
    assert did_you_mean["view"] == "knowledge.document_context"
    assert did_you_mean["matched_include"] == "docs"
    assert emitted["recommended_next_action"] == (
        "potpie graph read --subgraph knowledge --view document_context"
    )


def test_graph_status_json_uses_workbench_envelope() -> None:
    _common.set_json(True)
    workbench = _Workbench(
        quality_result=_quality_result(report="summary", status="watch")
    )
    _common.set_host(_Host(_Graph(), backend=_Backend(), graph_workbench=workbench))

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
    _common.set_host(_Host(_NotReadyGraph(), backend=_Backend()))

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
    from potpie_context_engine.adapters.inbound.cli.commands import _common

    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    _common.set_json(True)
    host = _Host(_GraphByPot({"p1": {"claims": 0}, "p2": {"claims": 82}}))
    host.pots = Pots()
    _common.set_host(host)

    result = CliRunner().invoke(graph.graph_app, ["status"])

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    body = _assert_graph_envelope(emitted, "graph.status", pot_id="p1")
    assert body["pot"]["id"] == "p1"
    assert emitted["warnings"]
    assert "p2" in emitted["warnings"][0]
    assert "82 claims" in emitted["warnings"][0]


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
    _common.set_host(_Host(_Graph(), backend=_Backend()))

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


def test_graph_describe_returns_executable_view_contract() -> None:
    _common.set_json(True)
    graph_service = _Graph()
    _common.set_host(_Host(graph_service))

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
    assert (
        emitted["recommended_next_action"]
        == "Use `potpie graph read --subgraph debugging --view prior_occurrences --json` after choosing a scope."
    )
    # The CLI is a thin client: the contract must be answered through the
    # service request, never a CLI-local domain call.
    assert graph_service.describe_request is not None
    assert graph_service.describe_request.subgraph == "debugging"
    assert graph_service.describe_request.view == "prior_occurrences"
    assert graph_service.describe_request.include_examples is True


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


def _invoke_timeline_read_text(*args: str) -> str:
    _common.set_json(False)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(graph_service))
    result = CliRunner().invoke(
        graph.graph_app,
        ["read", "--subgraph", "recent_changes", "--view", "timeline", *args],
    )
    assert result.exit_code == 0
    return _plain_cli_output(result.output)


def test_graph_read_table_format_renders_pipe_table() -> None:
    output = _invoke_timeline_read_text("--format", "table", "--limit", "2")
    assert "occurred_at |" in output
    assert "--- | ---" in output
    data_lines = [
        line
        for line in output.splitlines()
        if line.strip()
        and not line.startswith("view=")
        and not line.startswith("scope=")
        and not line.startswith("occurred_at |")
        and not line.startswith("--- |")
    ]
    assert data_lines
    assert not any(line.lstrip().startswith("•") for line in data_lines)


def test_graph_read_events_format_uses_bullets_not_table() -> None:
    output = _invoke_timeline_read_text("--format", "events", "--limit", "2")
    assert "  • " in output
    assert "--- | ---" not in output


def test_graph_read_table_vs_events_produce_different_output() -> None:
    events_output = _invoke_timeline_read_text("--format", "events", "--limit", "2")
    table_output = _invoke_timeline_read_text("--format", "table", "--limit", "2")
    assert events_output != table_output


def test_graph_read_text_detail_full_shows_claim_or_breakdown() -> None:
    compact_output = _invoke_timeline_read_text(
        "--format", "events", "--detail", "compact", "--limit", "1"
    )
    full_output = _invoke_timeline_read_text(
        "--format", "events", "--detail", "full", "--limit", "1"
    )
    assert "truth=" in full_output
    assert "truth=" not in compact_output


def test_graph_read_text_relations_summary_shows_counts() -> None:
    output = _invoke_timeline_read_text(
        "--format", "events", "--relations", "summary", "--limit", "1"
    )
    assert "relations:" in output
    assert "TOUCHED" in output


def test_graph_read_text_relations_full_lists_edges() -> None:
    output = _invoke_timeline_read_text(
        "--format",
        "events",
        "--relations",
        "full",
        "--detail",
        "full",
        "--limit",
        "1",
    )
    assert "↳ TOUCHED" in output
    assert "↳ PERFORMED" in output


def test_timeline_recent_table_format() -> None:
    _common.set_json(False)
    graph_service = _Graph(read_result=_timeline_env())
    _common.set_host(_Host(graph_service))

    result = CliRunner().invoke(
        graph.timeline_app,
        ["--format", "table", "--limit", "2"],
    )

    assert result.exit_code == 0
    output = _plain_cli_output(result.output)
    assert "occurred_at |" in output
    assert "--- | ---" in output
