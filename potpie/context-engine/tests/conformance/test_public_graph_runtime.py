from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from potpie_context_core.api import (
    DEFAULT_GRAPH_DEFINITION,
    ClaimQueryFilter,
    EdgeTypeSpec,
    EntityTypeSpec,
    GraphCatalogRequest,
    GraphDescribeRequest,
    GraphExtension,
    GraphMutationPlanStatus,
    GraphReadRequest,
    GraphReaderSpec,
    GraphViewSpec,
    IdentityClass,
    ReconciliationConfig,
    SemanticMutationRequest,
    build_graph_runtime,
)
from potpie_context_core.graph_mutations import ProvenanceContext
from potpie_context_core.ports.claim_query import (
    AsyncClaimQueryPort,
    ClaimQueryPort,
)
from potpie_context_core.ports.graph.backend import GraphBackend
from potpie_context_core.ports.graph.inbox_store import (
    AsyncGraphInboxStorePort,
    GraphInboxStorePort,
)
from potpie_context_core.ports.graph.mutation import GraphMutationPort
from potpie_context_core.ports.graph.plan_store import (
    AsyncGraphPlanStorePort,
    GraphPlanStorePort,
)
from potpie_context_core.reconciliation import MutationBatch, MutationResult
from potpie_context_core.reconciliation_config import current_reconciliation_config
from potpie_context_engine.api import Candidate, RankedItem, ReadResponse
from potpie_context_engine.testing import (
    InMemoryGraphBackend,
    InMemoryGraphInboxStore,
    InMemoryGraphPlanStore,
    build_test_graph_runtime,
)
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)


class _WidgetReader:
    def __init__(self, *, claim_query, **_) -> None:
        self.claim_query = claim_query

    def read(self, request) -> ReadResponse:
        rows = self.claim_query.find_claims(
            ClaimQueryFilter(
                pot_id=request.pot_id,
                predicate_in=("CONNECTS_WIDGET",),
            )
        )
        items = tuple(
            RankedItem(
                candidate=Candidate(
                    candidate_key=row.claim_key or row.subject_key,
                    payload={
                        "subject_key": row.subject_key,
                        "object_key": row.object_key,
                        "predicate": row.predicate,
                        "claim_key": row.claim_key,
                        "fact": row.fact,
                        "truth": row.truth,
                    },
                ),
                score=1.0,
                breakdown={"strength": 1.0},
            )
            for row in rows
        )
        return ReadResponse(
            family="widgets",
            items=items,
            coverage_status="complete" if items else "empty",
            meta={"candidate_pool": len(items)},
        )


def _definition(*, singleton: bool = False):
    extension = GraphExtension(
        name="widgets",
        version="1.0",
        entity_types={
            "Widget": EntityTypeSpec(
                label="Widget",
                category="topology",
                description="Widget.",
                identity_class=IdentityClass.SLUG_ALIAS,
                key_prefix="widget",
                identity_policy="widget:<slug>",
            )
        },
        edge_types={
            "CONNECTS_WIDGET": EdgeTypeSpec(
                edge_type="CONNECTS_WIDGET",
                description="Connects widgets.",
                allowed_pairs=(("Widget", "Widget"),),
                category="topology",
                singleton=singleton,
            )
        },
        views={
            "widgets.network": GraphViewSpec(
                name="widgets.network",
                subgraph="widgets",
                view="network",
                v1_include="widgets",
                description="Widget network.",
                backed=True,
            )
        },
        readers={
            "widgets": GraphReaderSpec(
                name="widgets",
                include="widgets",
                factory=_WidgetReader,
            )
        },
    )
    return DEFAULT_GRAPH_DEFINITION.extend(extension)


def _request() -> SemanticMutationRequest:
    return SemanticMutationRequest.parse(
        {
            "pot_id": "pot:widgets",
            "operations": [
                {
                    "op": "assert_claim",
                    "subject": {"key": "widget:a", "type": "Widget"},
                    "predicate": "CONNECTS_WIDGET",
                    "object": {"key": "widget:b", "type": "Widget"},
                    "truth": "agent_claim",
                    "description": "Widget A connects to widget B.",
                }
            ],
        }
    )


def _retract_request() -> SemanticMutationRequest:
    return SemanticMutationRequest.parse(
        {
            "pot_id": "pot:widgets",
            "operations": [
                {
                    "op": "retract_claim",
                    "subgraph": "widgets",
                    "subject": {"key": "widget:a", "type": "Widget"},
                    "predicate": "CONNECTS_WIDGET",
                    "object": {"key": "widget:b", "type": "Widget"},
                    "reason": "Widgets were disconnected.",
                }
            ],
        },
        allow_review_required=True,
        approved_by="user:test",
    )


class _FailingObserver:
    def observe(self, event, fields) -> None:
        del event, fields
        raise RuntimeError("observer unavailable")


def _runtime_with_failing_observer():
    return build_graph_runtime(
        InMemoryGraphBackend(),
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        _definition(),
        observability=_FailingObserver(),
    )


def test_public_runtime_extension_round_trip_and_status() -> None:
    runtime = build_test_graph_runtime(definition=_definition())

    mutation = runtime.mutate(_request())
    catalog = runtime.catalog(GraphCatalogRequest(pot_id="pot:widgets"))
    read = runtime.read(
        GraphReadRequest(
            pot_id="pot:widgets",
            subgraph="widgets",
            view="network",
        )
    )
    status = runtime.status("pot:widgets")

    assert mutation.ok
    assert any(view["name"] == "widgets.network" for view in catalog.views)
    assert read.items[0]["entity_type"] == "Widget"
    assert status["definition"]["extensions"] == {"widgets": "1.0"}


def test_runtime_bridges_satisfy_advertised_runtime_protocols() -> None:
    runtime = build_test_graph_runtime(definition=_definition())

    assert isinstance(runtime.backend, GraphBackend)
    assert isinstance(runtime.backend.mutation, GraphMutationPort)
    assert isinstance(runtime.backend.claim_query, ClaimQueryPort)
    assert isinstance(runtime.backend.claim_query, AsyncClaimQueryPort)
    assert isinstance(runtime.plan_store, GraphPlanStorePort)
    assert isinstance(runtime.plan_store, AsyncGraphPlanStorePort)
    assert isinstance(runtime.inbox_store, GraphInboxStorePort)
    assert isinstance(runtime.inbox_store, AsyncGraphInboxStorePort)


def test_public_async_runtime_recovers_a_stale_commit_attempt() -> None:
    async def journey() -> None:
        runtime = build_test_graph_runtime(definition=_definition())
        proposal = runtime.propose(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {"key": "widget:recovered", "type": "Widget"},
                    }
                ]
            },
            pot_id="pot:widgets",
        )
        record = runtime.plan_store.get(
            pot_id="pot:widgets",
            plan_id=proposal.plan_id,
        )
        assert record is not None
        crashed = replace(
            record,
            status=GraphMutationPlanStatus.committing.value,
            mutation_id="mutation:runtime-recovery",
            commit_attempt_id="attempt:crashed-runtime",
            commit_attempt_started_at=datetime.now(timezone.utc)
            - timedelta(minutes=10),
        )
        assert runtime.plan_store.compare_and_set(
            expected=record,
            replacement=crashed,
        )

        recovered = await runtime.commit_async(
            proposal.plan_id,
            pot_id="pot:widgets",
            recover_stale=True,
        )

        assert recovered.ok
        assert recovered.mutation_id == "mutation:runtime-recovery"

    asyncio.run(journey())


def test_runtime_snapshots_reconciliation_environment_per_instance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", "0")
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")
    first = build_test_graph_runtime()

    monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", "1")
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "1")
    second = build_test_graph_runtime()

    monkeypatch.setenv("CONTEXT_ENGINE_RECONCILIATION_ENABLED", "0")
    monkeypatch.setenv("CONTEXT_ENGINE_INFER_LABELS", "0")

    assert first.reconciliation_config.enabled is False
    assert first.reconciliation_config.infer_canonical_labels is False
    assert second.reconciliation_config.enabled is True
    assert second.reconciliation_config.infer_canonical_labels is True


def test_observer_failure_does_not_hide_sync_runtime_results() -> None:
    runtime = _runtime_with_failing_observer()

    mutation = runtime.mutate(_request())
    read = runtime.read(
        GraphReadRequest(
            pot_id="pot:widgets",
            subgraph="widgets",
            view="network",
        )
    )
    proposal = runtime.propose(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "widget:c", "type": "Widget"},
                }
            ]
        },
        pot_id="pot:widgets",
    )
    commit = runtime.commit(proposal.plan_id, pot_id="pot:widgets")

    assert mutation.ok
    assert read.items
    assert proposal.ok
    assert commit.ok


def test_public_runtime_retracts_extension_predicate() -> None:
    runtime = build_test_graph_runtime(definition=_definition())

    assert runtime.mutate(_request()).ok
    retracted = runtime.mutate(_retract_request())

    assert retracted.ok
    assert (
        runtime.backend.claim_query.find_claims(ClaimQueryFilter(pot_id="pot:widgets"))
        == []
    )
    history = runtime.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="pot:widgets", include_invalidated=True)
    )
    assert len(history) == 1
    assert history[0].invalid_at is not None


def test_quality_uses_extension_singleton_predicates() -> None:
    runtime = build_test_graph_runtime(definition=_definition(singleton=True))
    mutation = SemanticMutationRequest.parse(
        {
            "pot_id": "pot:widgets",
            "operations": [
                {
                    "op": "assert_claim",
                    "subject": {"key": "widget:a", "type": "Widget"},
                    "predicate": "CONNECTS_WIDGET",
                    "object": {"key": "widget:b", "type": "Widget"},
                    "truth": "agent_claim",
                    "description": "Widget A connects to Widget B.",
                },
                {
                    "op": "assert_claim",
                    "subject": {"key": "widget:a", "type": "Widget"},
                    "predicate": "CONNECTS_WIDGET",
                    "object": {"key": "widget:c", "type": "Widget"},
                    "truth": "agent_claim",
                    "description": "Widget A connects to Widget C.",
                },
            ],
        }
    )

    assert runtime.mutate(mutation).ok
    quality = runtime.quality(
        pot_id="pot:widgets",
        report="conflicting-claims",
    )

    assert len(quality.findings) == 1
    assert quality.findings[0].predicates == ("CONNECTS_WIDGET",)


def test_persistent_backend_uses_the_runtime_definition(tmp_path) -> None:
    backend = EmbeddedGraphBackend(home=tmp_path)
    runtime = build_graph_runtime(
        backend,
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        _definition(),
    )

    mutation = runtime.mutate(_request())
    rows = backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id="pot:widgets", include_invalidated=True)
    )
    read = runtime.read(
        GraphReadRequest(
            pot_id="pot:widgets",
            subgraph="widgets",
            view="network",
        )
    )

    assert mutation.ok
    assert {row.predicate for row in rows} == {"CONNECTS_WIDGET"}
    assert backend.claim_query.entity_labels(
        pot_id="pot:widgets", entity_keys=["widget:a", "widget:b"]
    ) == {
        "widget:a": ("Widget",),
        "widget:b": ("Widget",),
    }
    assert read.items


def test_describe_uses_extension_definition() -> None:
    runtime = build_test_graph_runtime(definition=_definition())

    described = runtime.describe(GraphDescribeRequest(subgraph="widgets"))

    assert described["ontology_version"] == _definition().ontology_version
    assert described["extensions"] == {"widgets": "1.0"}
    assert [view["name"] for view in described["subgraph"]["views"]] == [
        "widgets.network"
    ]


def test_two_differently_extended_runtimes_do_not_leak() -> None:
    backend = InMemoryGraphBackend()
    extended = build_graph_runtime(
        backend,
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        _definition(),
    )
    base = build_graph_runtime(
        backend,
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
    )

    extended.mutate(_request())

    assert any(
        view["name"] == "widgets.network"
        for view in extended.catalog(GraphCatalogRequest(pot_id="pot:widgets")).views
    )
    assert all(
        view["name"] != "widgets.network"
        for view in base.catalog(GraphCatalogRequest(pot_id="pot:base")).views
    )
    assert not base.mutate(_request()).ok


def test_async_runtime_runs_inside_an_active_event_loop() -> None:
    async def journey() -> None:
        runtime = build_test_graph_runtime(definition=_definition())
        mutation = await runtime.mutate_async(_request())
        read = await runtime.read_async(
            GraphReadRequest(
                pot_id="pot:widgets",
                subgraph="widgets",
                view="network",
            )
        )
        proposal = await runtime.propose_async(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {"key": "widget:c", "type": "Widget"},
                    }
                ]
            },
            pot_id="pot:widgets",
        )
        commit = await runtime.commit_async(proposal.plan_id, pot_id="pot:widgets")

        assert mutation.ok
        assert read.items
        assert commit.ok

    asyncio.run(journey())


def test_observer_failure_does_not_hide_async_runtime_results() -> None:
    async def journey() -> None:
        runtime = _runtime_with_failing_observer()

        mutation = await runtime.mutate_async(_request())
        read = await runtime.read_async(
            GraphReadRequest(
                pot_id="pot:widgets",
                subgraph="widgets",
                view="network",
            )
        )
        proposal = await runtime.propose_async(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {"key": "widget:c", "type": "Widget"},
                    }
                ]
            },
            pot_id="pot:widgets",
        )
        commit = await runtime.commit_async(proposal.plan_id, pot_id="pot:widgets")

        assert mutation.ok
        assert read.items
        assert proposal.ok
        assert commit.ok

    asyncio.run(journey())


class _AsyncMutation:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen_configs: list[ReconciliationConfig | None] = []

    async def apply_async(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        self.seen_configs.append(current_reconciliation_config())
        return await self.inner.apply_async(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    async def invalidate_async(self, **kwargs):
        return self.inner.invalidate(**kwargs)

    async def reset_pot_async(self, pot_id):
        return self.inner.reset_pot(pot_id)

    async def readiness_async(self, pot_id):
        return self.inner.readiness(pot_id)


class _AsyncClaims:
    def __init__(self, inner) -> None:
        self.inner = inner

    @property
    def match_mode(self):
        return self.inner.match_mode

    async def find_claims_async(self, filter_):
        return self.inner.find_claims(filter_)

    async def entity_labels_async(self, **kwargs):
        return self.inner.entity_labels(**kwargs)

    async def entity_properties_async(self, **kwargs):
        return self.inner.entity_properties(**kwargs)


class _AsyncStore:
    def __init__(self, inner) -> None:
        self.inner = inner

    async def save_async(self, item):
        self.inner.save(item)

    async def get_async(self, **kwargs):
        return self.inner.get(**kwargs)

    async def compare_and_set_async(self, **kwargs):
        return self.inner.compare_and_set(**kwargs)

    async def list_async(self, **kwargs):
        return self.inner.list(**kwargs)


class _AsyncOnlyBackend:
    def __init__(self) -> None:
        self.inner = InMemoryGraphBackend()
        self.mutation = _AsyncMutation(self.inner.mutation)
        self.claim_query = _AsyncClaims(self.inner.claim_query)
        self.semantic = self.inner.semantic
        self.inspection = self.inner.inspection
        self.analytics = self.inner.analytics
        self.snapshot = self.inner.snapshot

    @property
    def profile(self):
        return "async_only_test"

    def capabilities(self):
        return self.inner.capabilities()

    def bind_definition(self, definition):
        self.inner = self.inner.bind_definition(definition)
        self.mutation = _AsyncMutation(self.inner.mutation)
        self.claim_query = _AsyncClaims(self.inner.claim_query)
        self.semantic = self.inner.semantic
        self.inspection = self.inner.inspection
        self.analytics = self.inner.analytics
        self.snapshot = self.inner.snapshot
        return self


def test_async_only_ports_use_the_serving_loop_bridge() -> None:
    async def journey() -> None:
        backend = _AsyncOnlyBackend()
        config = ReconciliationConfig(infer_canonical_labels=False)
        runtime = build_graph_runtime(
            backend,
            _AsyncStore(InMemoryGraphPlanStore()),
            _AsyncStore(InMemoryGraphInboxStore()),
            _definition(),
            reconciliation_config=config,
        )
        mutation = await runtime.mutate_async(_request())
        read = await runtime.read_async(
            GraphReadRequest(
                pot_id="pot:widgets",
                subgraph="widgets",
                view="network",
            )
        )
        proposal = await runtime.propose_async(
            {
                "operations": [
                    {
                        "op": "upsert_entity",
                        "subject": {"key": "widget:c", "type": "Widget"},
                    }
                ]
            },
            pot_id="pot:widgets",
        )
        commit = await runtime.commit_async(proposal.plan_id, pot_id="pot:widgets")

        assert mutation.ok
        assert read.items
        assert commit.ok
        assert backend.mutation.seen_configs == [config, config]
        assert current_reconciliation_config() is None

    asyncio.run(journey())


class _SyncMutation:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen_configs: list[ReconciliationConfig | None] = []

    def apply(
        self,
        plan: MutationBatch,
        *,
        expected_pot_id: str,
        provenance_context: ProvenanceContext | None = None,
    ) -> MutationResult:
        self.seen_configs.append(current_reconciliation_config())
        return self.inner.apply(
            plan,
            expected_pot_id=expected_pot_id,
            provenance_context=provenance_context,
        )

    def invalidate(self, **kwargs):
        return self.inner.invalidate(**kwargs)

    def reset_pot(self, pot_id):
        return self.inner.reset_pot(pot_id)

    def readiness(self, pot_id):
        return self.inner.readiness(pot_id)


class _SyncOnlyBackend:
    def __init__(self) -> None:
        self._replace_inner(InMemoryGraphBackend())

    def _replace_inner(self, inner) -> None:
        self.inner = inner
        self.mutation = _SyncMutation(inner.mutation)
        self.claim_query = inner.claim_query
        self.semantic = inner.semantic
        self.inspection = inner.inspection
        self.analytics = inner.analytics
        self.snapshot = inner.snapshot

    @property
    def profile(self):
        return "sync_only_test"

    def capabilities(self):
        return self.inner.capabilities()

    def bind_definition(self, definition):
        self._replace_inner(self.inner.bind_definition(definition))
        return self


def test_sync_only_mutation_uses_non_blocking_async_bridge() -> None:
    async def journey() -> None:
        backend = _SyncOnlyBackend()
        config = ReconciliationConfig(infer_canonical_labels=False)
        runtime = build_graph_runtime(
            backend,
            InMemoryGraphPlanStore(),
            InMemoryGraphInboxStore(),
            _definition(),
            reconciliation_config=config,
        )

        mutation = await runtime.mutate_async(_request())

        assert mutation.ok
        assert backend.mutation.seen_configs == [config]
        assert current_reconciliation_config() is None

    asyncio.run(journey())


def test_old_sync_mutation_signature_supports_runtime_mutation_and_commit() -> None:
    backend = _SyncOnlyBackend()
    config = ReconciliationConfig(infer_canonical_labels=False)
    runtime = build_graph_runtime(
        backend,
        InMemoryGraphPlanStore(),
        InMemoryGraphInboxStore(),
        _definition(),
        reconciliation_config=config,
    )

    mutation = runtime.mutate(_request())
    proposal = runtime.propose(
        {
            "operations": [
                {
                    "op": "upsert_entity",
                    "subject": {"key": "widget:legacy-port", "type": "Widget"},
                }
            ]
        },
        pot_id="pot:widgets",
    )
    commit = runtime.commit(proposal.plan_id, pot_id="pot:widgets")

    assert mutation.ok
    assert commit.ok
    assert backend.mutation.seen_configs == [config, config]
    assert current_reconciliation_config() is None
