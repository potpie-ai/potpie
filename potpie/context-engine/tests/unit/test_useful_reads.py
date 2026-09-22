"""Agent audit section 2: useful, scoped reads never masquerade as exhaustive."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from potpie_context_core.graph_workbench_ontology import ontology_contract
from potpie_context_core.ports.claim_query import ClaimRow
from potpie_context_core.ports.graph_service import (
    GraphCatalogRequest,
    GraphReadRequest,
    GraphEntitySearchRequest,
)
from potpie_context_core.ports.resource_store import (
    ResourceBatchResult,
    format_resource_id,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services.resource_facade import ResourceFacade
from potpie_context_engine.testing import InMemoryResourceStore, write_import_directory

NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


def row(predicate, subject, obj, *, fact="timeout", pot="p"):
    return ClaimRow(
        pot_id=pot,
        predicate=predicate,
        subject_key=subject,
        object_key=obj,
        fact=fact,
        valid_at=NOW,
        evidence_strength="attested",
        source_system="agent",
        source_ref="test:source",
        properties={},
    )


def service_with(*rows):
    backend = InMemoryGraphBackend()
    backend.claim_query.add_many(rows)
    return DefaultGraphService(backend=backend)


def test_feature_overview_needs_no_anchor_and_discloses_both_cuts():
    service = service_with(
        *[row("PROVIDES", "repo:a", f"feature:{i}") for i in range(5)]
    )
    result = service.read(
        GraphReadRequest(
            pot_id="p", subgraph="features", view="feature_context", limit=1
        )
    )
    assert result.ok and len(result.items) == 1
    coverage = result.coverage[0]
    assert coverage["completeness"] == "truncated"
    assert coverage["page_status"] == "complete"
    assert coverage["candidate_pool_unit"] == "claims"
    assert coverage["metadata"]["ranking_omitted"] == 4
    assert coverage["metadata"]["projection_omitted"] == 1
    assert result.effective_request["scope"] == {}
    assert result.effective_request["pot_id"] == "p"


def test_feature_browse_is_pot_scoped_and_repo_is_explicit():
    service = service_with(
        row("PROVIDES", "repo:a", "feature:a"),
        row("PROVIDES", "repo:b", "feature:b"),
        row("PROVIDES", "repo:c", "feature:c", pot="other"),
    )
    request = GraphReadRequest(
        pot_id="p", subgraph="features", view="feature_context", limit=12
    )
    overview = service.read(request)
    assert {item["entity_key"] for item in overview.items} == {
        "repo:a",
        "repo:b",
        "feature:a",
        "feature:b",
    }
    scoped = service.read(replace(request, scope={"repo": "repo:a"}))
    assert {item["entity_key"] for item in scoped.items} == {"repo:a", "feature:a"}
    assert scoped.coverage[0]["completeness"] == "complete"


def test_full_candidate_budget_cannot_claim_exhaustive_features():
    service = service_with(
        *[row("PROVIDES", "repo:a", f"feature:{i}") for i in range(100)]
    )
    result = service.read(
        GraphReadRequest(
            pot_id="p", subgraph="features", view="feature_context", limit=2
        )
    )
    assert result.coverage[0]["metadata"]["candidate_budget_reached"] is True
    assert result.coverage[0]["completeness"] != "complete"


@pytest.mark.parametrize(
    "view,subgraph,predicate",
    [
        ("prior_occurrences", "debugging", "REPRODUCES"),
        ("document_context", "knowledge", "DOCUMENTS"),
    ],
)
def test_unsupported_threshold_has_one_separate_bounded_fallback(
    view, subgraph, predicate
):
    service = service_with(
        row(
            predicate,
            "bug_pattern:timeout" if subgraph == "debugging" else "document:timeout",
            "service:api",
        )
    )
    if subgraph == "knowledge":
        service.backend.claim_query.set_entity_label(
            pot_id="p", entity_key="document:timeout", labels=("Document",)
        )
    request = GraphReadRequest(
        pot_id="p",
        subgraph=subgraph,
        view=view,
        query="timeout",
        limit=2,
        query_threshold=1.0,
    )
    result = service.read(request)
    assert not result.ok and result.status == "partial" and result.items == ()
    assert result.unsupported[0]["name"] == "query_threshold"
    assert result.fallback_context["items"]
    assert len(result.fallback_context["items"]) <= 2
    assert result.fallback_context["effective_request"]["query_threshold"] is None
    assert result.effective_request["query_threshold"] == 1.0
    assert not result.effective_request["executed"]


def test_debug_time_window_never_claims_old_context_meets_future_bounds():
    service = service_with(
        row("REPRODUCES", "bug_pattern:timeout", "service:api"),
        row("RESOLVED", "fix:old", "bug_pattern:timeout"),
    )
    request = GraphReadRequest(
        pot_id="p",
        subgraph="debugging",
        view="prior_occurrences",
        query="timeout",
        since=datetime(2100, 1, 1, tzinfo=timezone.utc),
        query_threshold=0.9,
    )
    result = service.read(request)
    assert result.status == "partial" and not result.items
    assert {item["name"] for item in result.unsupported} == {"since", "query_threshold"}
    assert result.fallback_context["items"]
    assert result.fallback_context["effective_request"]["since"] is None
    assert "occurrence timestamps" in result.message


def test_unsupported_threshold_without_query_does_not_read():
    service = service_with()
    result = service.read(
        GraphReadRequest(
            pot_id="p", subgraph="features", view="feature_context", query_threshold=0.5
        )
    )
    assert not result.ok and result.items == () and not result.fallback_context


def test_catalog_and_describe_advertise_threshold_and_occurrence_capabilities():
    service = service_with()
    views = {
        view["name"]: view
        for view in service.catalog(GraphCatalogRequest(pot_id="p")).views
    }
    for name in ("debugging.prior_occurrences", "knowledge.document_context"):
        assert views[name]["extra"]["query_threshold"]["supported"] is False
    assert views["knowledge.document_passages"]["extra"]["query_threshold"][
        "requires_calibrated_index"
    ]
    assert views["features.feature_context"]["required_any_scope"] == []
    contract = ontology_contract().view("debugging.prior_occurrences")
    assert "since" not in contract.supported_filters
    assert contract.extra["time_window"]["semantics"] == "occurrence_time"


def test_exact_identity_is_not_labelled_fuzzy_and_isolated_anchor_is_retained():
    service = service_with()
    service.backend.claim_query.set_entity_label(
        pot_id="p", entity_key="service:alone", labels=("Service",)
    )
    result = service.search_entities(
        GraphEntitySearchRequest(pot_id="p", query="service:alone")
    )
    assert result.match_status == "exact_match"
    present = service.backend.inspection.neighborhood(
        pot_id="p", entity_key="service:alone"
    )
    missing = service.backend.inspection.neighborhood(
        pot_id="p", entity_key="service:missing"
    )
    assert [node.key for node in present.nodes] == [
        "service:alone"
    ] and not present.edges
    assert not missing.nodes


@pytest.fixture(params=["memory", "local"])
def store(request, tmp_path):
    return (
        InMemoryResourceStore()
        if request.param == "memory"
        else LocalResourceStore(home=tmp_path / "home")
    )


def seed(store, tmp_path, *, version=1):
    source = write_import_directory(
        tmp_path / f"source-{version}",
        [
            {
                "slug": "body",
                "summary": "A useful document",
                "chunks": [
                    {
                        "seq": i,
                        "label": f"chunk {i}",
                        "text": f"revision {version} chunk {i}",
                    }
                    for i in range(3)
                ],
            },
        ],
    )
    return store.import_dir(pot_id="p", slug="doc", source_dir=source)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "missing",
    ["potpie://res/doc/body/9999", "malformed", "potpie://res/missing/body/0000"],
)
def test_mixed_resource_batch_keeps_success_and_request_order(
    store, tmp_path, reverse, missing
):
    seed(store, tmp_path)
    valid = format_resource_id("doc", "body", 1, revision=1)
    ids = (missing, valid) if reverse else (valid, missing)
    result = ResourceFacade(store=store).get(pot_id="p", resource_ids=ids)
    assert isinstance(result, ResourceBatchResult) and result.status == "partial"
    assert [outcome.resource_id for outcome in result.outcomes] == list(ids)
    assert len(result.chunks) == 1 and result.chunks[0].text == "revision 1 chunk 1"
    assert sum(outcome.status == "error" for outcome in result.outcomes) == 1


def test_neighbors_deduplicate_and_never_mix_revision_manifests(store, tmp_path):
    seed(store, tmp_path, version=1)
    seed(store, tmp_path, version=2)
    roots = tuple(
        format_resource_id("doc", "body", i, revision=revision)
        for revision, i in [(1, 0), (1, 1), (2, 1)]
    )
    missing = format_resource_id("doc", "no-section", 0, revision=1)
    result = ResourceFacade(store=store).get(
        pot_id="p", resource_ids=(*roots, missing), with_neighbors=True
    )
    assert result.status == "partial" and len(result.chunks) == 6
    assert len({chunk.resource_id for chunk in result.chunks}) == 6
    for outcome in result.outcomes[:3]:
        revision = 1 if outcome.resource_id in roots[:2] else 2
        assert all(value.endswith(f"@rev{revision}") for value in outcome.chunk_ids)
    assert all(f"revision {chunk.revision}" in chunk.text for chunk in result.chunks)
    candidates = result.outcomes[-1].errors[0]["detail"]["candidates"]
    assert 0 < len(candidates) <= 5
    assert all(
        candidate["revision"] == 1 and "--pot p" in candidate["fetch_command"]
        for candidate in candidates
    )


def test_ambiguous_revision_returns_choices_without_substitution(store, tmp_path):
    seed(store, tmp_path, version=1)
    seed(store, tmp_path, version=2)
    result = ResourceFacade(store=store).get(
        pot_id="p", resource_ids=(format_resource_id("doc", "body", 0),)
    )
    assert result.status == "error" and not result.chunks
    error = result.outcomes[0].errors[0]
    assert error["code"] == "resource_revision_ambiguous"
    assert error["detail"]["revision_substituted"] is False
    assert error["detail"]["candidates"]


def test_all_success_facade_shape_remains_a_tuple(store, tmp_path):
    seed(store, tmp_path)
    result = ResourceFacade(store=store).get(
        pot_id="p",
        resource_ids=(format_resource_id("doc", "body", 1),),
        with_neighbors=True,
    )
    assert isinstance(result, tuple) and [chunk.seq for chunk in result] == [0, 1, 2]
    assert all("@rev" not in chunk.resource_id for chunk in result)


def test_explicit_similarity_floor_cannot_be_bypassed_by_literal_text():
    from potpie_context_engine.application.readers._common import row_matches_query

    matching = row("POLICY_APPLIES_TO", "preference:testing", "repo:a", fact="testing")
    low = replace(matching, properties={"semantic_similarity": 0.1})
    high = replace(matching, properties={"semantic_similarity": 0.95})
    assert row_matches_query(low, "testing")
    assert not row_matches_query(low, "testing", threshold=0.9, semantic_only=True)
    assert not row_matches_query(matching, "testing", threshold=0.9, semantic_only=True)
    assert row_matches_query(high, "testing", threshold=0.9, semantic_only=True)


def test_filtered_out_exact_entity_is_not_reported_as_exact_match():
    service = service_with()
    service.backend.claim_query.set_entity_label(
        pot_id="p", entity_key="service:alone", labels=("Service",)
    )
    result = service.search_entities(
        GraphEntitySearchRequest(pot_id="p", query="service:alone", type="Repository")
    )
    assert not result.entities and result.match_status != "exact_match"


def test_missing_query_threshold_reports_one_constraint():
    service = service_with()
    result = service.read(
        GraphReadRequest(
            pot_id="p",
            subgraph="decisions",
            view="preferences_for_scope",
            scope={"repo": "repo:a"},
            query_threshold=0.9,
        )
    )
    assert not result.ok
    assert [item["name"] for item in result.unsupported] == ["query_threshold"]


def test_legacy_resource_recovery_has_a_separate_backend_work_budget(tmp_path):
    store = InMemoryResourceStore()
    seed(store, tmp_path)

    class LegacyStore:
        calls = 0

        def get_many(self, **kwargs):
            return store.get_many(**kwargs)

        def get(self, **kwargs):
            self.calls += 1
            return store.get(**kwargs)

    legacy = LegacyStore()
    ids = (
        format_resource_id("doc", "body", 0, revision=1),
        *(format_resource_id("doc", "body", i, revision=1) for i in range(3, 80)),
    )
    result = ResourceFacade(store=legacy).get(pot_id="p", resource_ids=ids)
    assert result.status == "partial" and len(result.outcomes) == len(ids)
    assert legacy.calls == 64
    assert result.outcomes[-1].errors[0]["code"] == "resource_read_budget_exceeded"


def test_resource_batch_cardinality_is_bounded_before_store_work():
    from potpie_context_core.ports.resource_store import (
        RESOURCE_GET_MAX_IDS,
        ResourceStoreError,
    )

    class NoStoreCalls:
        def get_batch(self, **kwargs):
            raise AssertionError("oversized requests must not reach the store")

    with pytest.raises(ResourceStoreError, match="No chunks were read") as error:
        ResourceFacade(store=NoStoreCalls()).get(
            pot_id="p", resource_ids=("invalid",) * (RESOURCE_GET_MAX_IDS + 1)
        )
    assert error.value.code == "resource_batch_too_large"


def test_page_full_uses_returned_entities_after_projection():
    service = service_with(row("PROVIDES", "repo:a", "feature:a"))
    result = service.read(
        GraphReadRequest(
            pot_id="p", subgraph="features", view="feature_context", limit=2
        )
    )
    assert len(result.items) == 2
    assert result.coverage[0]["metadata"]["page_full"] is True
    assert result.coverage[0]["completeness"] == "complete"
