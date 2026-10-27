"""Shared graph CLI transport doubles and response examples."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from potpie_context_engine.core.graph_views import views_for_catalog
from potpie_context_engine.core.ports.graph.analytics import RepairReport
from potpie_context_engine.core.ports.graph.backend import BackendCapabilities
from potpie_context_engine.core.ports.graph.inspection import (
    GraphEdge,
    GraphNode,
    GraphSlice,
)
from potpie_context_engine.core.ports.graph_service import (
    DataPlaneStatus,
    GraphCatalogResult,
    GraphEntityCandidate,
    GraphEntitySearchResult,
    GraphReadResult,
)
from potpie_context_engine.domain.nudge import GraphNudgeResult

from potpie.cli.commands import _common

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@pytest.fixture(autouse=True)
def graph_cli_state():
    """Isolate the injected runtime and output mode for graph CLI scenarios."""
    _common.set_runtime(None)
    _common.set_json(False)
    yield
    _common.set_runtime(None)
    _common.set_json(False)


def _plain_cli_output(output: str) -> str:
    return _ANSI_RE.sub("", output)


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


class _NoSelectedPots:
    def active_pot(self):
        return None

    def list_pots(self):
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
        from potpie_context_engine.core.graph_workbench_ontology import (
            describe_contract,
        )

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
        graph_workbench=None,
    ) -> None:
        self.graph = graph_service
        self.graph_workbench = graph_workbench
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
        raise AssertionError("unsupported snapshot port should not be reached")


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


def _non_timeline_env() -> GraphReadResult:
    return GraphReadResult(
        graph_contract_version="v1.5",
        ontology_version="2026-06-graph",
        view="infra_topology.service_neighborhood",
        subgraph="infra_topology",
        read_shape="entity_relations",
        coverage=(
            {"view": "infra_topology.service_neighborhood", "status": "complete"},
        ),
        quality={"status": "ok"},
        items=(
            {
                "entity_key": "service:payments-api",
                "entity_type": "Service",
                "score": 0.8,
                "summary": "payments API",
                "source_refs": ["repo:manifest"],
                "relations": [
                    {
                        "predicate": "DEPENDS_ON",
                        "from_key": "service:payments-api",
                        "to_key": "service:ledger-api",
                        "related_key": "service:ledger-api",
                        "fact": "payments depends on ledger",
                    }
                ],
            },
        ),
    )


def _unsupported_query_read_result() -> GraphReadResult:
    """What the real service returns for ``admin.inspection_slice --query QME``."""
    message = (
        "graph read view 'admin.inspection_slice' does not support filter "
        "query; supported filters: source_ref"
    )
    return GraphReadResult(
        graph_contract_version="v1.5",
        ontology_version="2026-06-graph",
        view="admin.inspection_slice",
        subgraph="admin",
        ok=False,
        status="unsupported_filter",
        message=message,
        read_shape="raw_graph",
        coverage=(
            {
                "view": "admin.inspection_slice",
                "status": "unsupported",
                "candidate_pool": 0,
            },
        ),
        quality={"status": "unsupported", "reason": "unsupported_filter"},
        unsupported=(
            {
                "name": "query",
                "reason": "unsupported_filter",
                "detail": {
                    "view": "admin.inspection_slice",
                    "supported_filters": ["source_ref"],
                },
            },
        ),
    )


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
