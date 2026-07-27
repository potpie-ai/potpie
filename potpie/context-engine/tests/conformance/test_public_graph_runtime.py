from __future__ import annotations

import asyncio

from potpie_context_core.api import (
    DEFAULT_GRAPH_DEFINITION,
    ClaimQueryFilter,
    EdgeTypeSpec,
    EntityTypeSpec,
    GraphCatalogRequest,
    GraphDescribeRequest,
    GraphExtension,
    GraphReadRequest,
    GraphReaderSpec,
    GraphViewSpec,
    IdentityClass,
    SemanticMutationRequest,
    build_graph_runtime,
)
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


class _AsyncMutation:
    def __init__(self, inner) -> None:
        self.inner = inner

    async def apply_async(self, *args, **kwargs):
        return await self.inner.apply_async(*args, **kwargs)

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
        runtime = build_graph_runtime(
            _AsyncOnlyBackend(),
            _AsyncStore(InMemoryGraphPlanStore()),
            _AsyncStore(InMemoryGraphInboxStore()),
            _definition(),
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

    asyncio.run(journey())


class _SyncMutation:
    def __init__(self, inner) -> None:
        self.inner = inner

    def apply(self, *args, **kwargs):
        return self.inner.apply(*args, **kwargs)

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
        runtime = build_graph_runtime(
            _SyncOnlyBackend(),
            InMemoryGraphPlanStore(),
            InMemoryGraphInboxStore(),
            _definition(),
        )

        mutation = await runtime.mutate_async(_request())

        assert mutation.ok

    asyncio.run(journey())
