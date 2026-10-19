"""Resource source intent survives retries across its store/graph boundary."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from potpie_context_engine.api import ResourceFacade
from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.adapters.outbound.resources.local_resource_store import (
    LocalResourceStore,
)
from potpie_context_engine.application.services.graph_service import DefaultGraphService
from potpie_context_engine.application.services import resource_graph_mutations
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.resource_store import ResourceStoreError
from potpie_context_engine.core.runtime import build_graph_runtime
from potpie_context_engine.core.semantic_mutations import (
    SemanticMutationRequest,
    SemanticMutationResult,
)
from potpie_context_engine.core.workbench_service import GraphWorkbenchService
from potpie_context_engine.testing import (
    InMemoryGraphPlanStore,
    InMemoryResourceStore,
    write_import_directory,
)

pytestmark = pytest.mark.unit

POT = "p"
DOC = "guide"


@dataclass
class _Host:
    home: Path
    backend: Any
    plans: Any
    store: Any
    runtime: Any = None
    facade: ResourceFacade | None = None

    def compose(self, *, reopen: bool = False) -> ResourceFacade:
        if reopen:
            if isinstance(self.backend, EmbeddedGraphBackend):
                self.backend = EmbeddedGraphBackend(home=self.home)
                self.plans = LocalJsonGraphPlanStore(home=self.home)
            if isinstance(self.store, LocalResourceStore):
                self.store = LocalResourceStore(home=self.home)
        self.runtime = build_graph_runtime(
            self.backend, self.plans, resource_store=self.store
        )
        self.facade = ResourceFacade.from_runtime(self.runtime, store=self.store)
        return self.facade

    def record(self, result):
        record = self.plans.get(pot_id=POT, plan_id=result.graph.plan_id)
        assert record is not None
        return record

    def sections(self) -> set[str]:
        return {
            row.subject_key
            for row in self.backend.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=POT,
                    predicate_in=("SECTION_OF",),
                    object_key_in=(f"document:{DOC}",),
                )
            )
        }


@pytest.fixture(params=["memory", "embedded"])
def graph_kind(request):
    return request.param


@pytest.fixture(params=["memory", "local"])
def host(tmp_path, request, graph_kind):
    backend = (
        InMemoryGraphBackend()
        if graph_kind == "memory"
        else EmbeddedGraphBackend(home=tmp_path)
    )
    plans = (
        InMemoryGraphPlanStore()
        if graph_kind == "memory"
        else LocalJsonGraphPlanStore(home=tmp_path)
    )
    store = (
        InMemoryResourceStore()
        if request.param == "memory"
        else LocalResourceStore(home=tmp_path)
    )
    result = _Host(tmp_path, backend, plans, store)
    result.compose()
    return result


def _source(host: _Host, section: str, *, version: int = 1) -> Path:
    return write_import_directory(
        host.home / f"source-{section}-{version}",
        [
            {
                "slug": section,
                "title": section.title(),
                "summary": f"Summary of {section}",
                "ordinal": 0,
                "content_hash": f"{section}-v{version}",
                "chunks": [{"label": "body", "text": f"{section} revision {version}"}],
            }
        ],
        source_ref="file:///guide.md",
        source_kind="markdown",
    )


def _import(host: _Host, source: Path):
    return host.facade.import_dir(pot_id=POT, slug=DOC, source_dir=source)


def _assert_bytes(host: _Host, section: str, revision: int, version: int = 1):
    manifest = host.store.current_manifest(pot_id=POT, slug=DOC)
    assert manifest.revision == revision
    assert tuple(row.slug for row in manifest.sections) == (section,)
    chunk = host.store.get(
        pot_id=POT, resource_id=f"potpie://res/{DOC}/{section}/0000@rev{revision}"
    )
    assert chunk.text == f"{section} revision {version}"


def _documents_claim(host: _Host):
    result = host.runtime.graph.mutate(
        SemanticMutationRequest.parse(
            {
                "idempotency_key": "section-scope",
                "operations": [
                    {
                        "op": "assert_claim",
                        "subgraph": "knowledge",
                        "predicate": "DOCUMENTS",
                        "subject": {
                            "key": f"docsection:{DOC}:one",
                            "type": "DocumentSection",
                        },
                        "object": {"key": "service:reader", "type": "Service"},
                        "description": "The section documents the reader service",
                    }
                ],
            },
            pot_id=POT,
        )
    )
    assert result.ok and result.status == "applied", result


def _reject_dependency_lookup(*_args, **_kwargs):
    raise AssertionError("a retry must use its frozen dependency selection")


def test_replaced_import_retry_retains_exact_plan_and_execution(host, monkeypatch):
    first = _import(host, _source(host, "one"))
    replacement_source = _source(host, "two")
    replacement = _import(host, replacement_source)
    assert first.graph_written and replacement.graph_written
    assert replacement.manifest.sections_removed == ("one",)
    assert host.sections() == {f"docsection:{DOC}:two"}
    record = host.record(replacement)
    assert record.prepared is not None and record.execution is not None
    assert record.execution.request_fingerprint == record.prepared.request_fingerprint
    assert all(
        operation["op"] != "retract_claim"
        for operation in record.original_payload["operations"]
    )
    version = host.backend.mutation.current_version(POT)
    plan_count = len(host.plans.list(pot_id=POT))

    host.compose(reopen=True)
    with monkeypatch.context() as patch:
        patch.setattr(ResourceFacade, "_dependent_claims", _reject_dependency_lookup)
        retried = _import(host, replacement_source)

    assert retried.graph_written
    assert retried.manifest.sections_removed == ()
    assert retried.graph.plan_id == replacement.graph.plan_id
    assert retried.graph.mutation_id == replacement.graph.mutation_id
    recovered = host.record(retried)
    assert recovered.prepared == record.prepared
    assert recovered.lowered_batch == record.lowered_batch
    assert recovered.execution == record.execution
    assert len(host.plans.list(pot_id=POT)) == plan_count
    assert host.backend.mutation.current_version(POT) == version
    assert host.sections() == {f"docsection:{DOC}:two"}


def test_crash_after_resource_publication_still_reconciles_old_live_section(
    host, monkeypatch
):
    assert _import(host, _source(host, "one")).graph_written
    replacement_source = _source(host, "two")
    before_plans = len(host.plans.list(pot_id=POT))

    def crash_before_preparation(*_args):
        raise SystemExit("resource published before graph preparation")

    with monkeypatch.context() as patch:
        patch.setattr(
            resource_graph_mutations, "import_graph", crash_before_preparation
        )
        with pytest.raises(SystemExit, match="before graph preparation"):
            _import(host, replacement_source)

    _assert_bytes(host, "two", revision=2)
    assert len(host.plans.list(pot_id=POT)) == before_plans
    assert host.sections() == {f"docsection:{DOC}:one"}
    host.compose(reopen=True)

    recovered = _import(host, replacement_source)

    assert recovered.graph_written
    assert recovered.manifest.sections_removed == ()
    assert host.sections() == {f"docsection:{DOC}:two"}
    assert any(
        operation["op"] == "retract_claim"
        for operation in host.record(recovered).accepted_ops
    )


def test_reimport_after_delete_gets_a_distinct_delete_identity(host):
    source = _source(host, "one")
    first = _import(host, source)
    first_delete = host.facade.delete(pot_id=POT, slug=DOC)
    assert first.graph_written and first_delete.removed and first_delete.graph_retracted
    assert host.sections() == set()
    first_record = host.record(first_delete)
    host.compose(reopen=True)

    second = _import(host, source)
    assert second.manifest.revision > first.manifest.revision
    assert second.graph_written
    second_delete = host.facade.delete(pot_id=POT, slug=DOC)

    assert second_delete.removed and second_delete.graph_retracted
    second_record = host.record(second_delete)
    assert second_record.original_payload["idempotency_key"] != (
        first_record.original_payload["idempotency_key"]
    )
    assert second_record.plan_id != first_record.plan_id
    assert second_record.mutation_id != first_record.mutation_id
    assert second_record.execution != first_record.execution
    assert host.sections() == set()
    with pytest.raises(ResourceStoreError) as absent:
        host.store.current_manifest(pot_id=POT, slug=DOC)
    assert absent.value.code == "resource_not_found"


@pytest.mark.parametrize("action", ["replace", "delete"])
def test_response_loss_recovers_receipt_without_reselecting_dependencies(
    host, monkeypatch, action
):
    assert _import(host, _source(host, "one")).graph_written
    _documents_claim(host)
    conclusion = host.runtime.graph.mutate(
        SemanticMutationRequest.parse(
            {
                "operations": [
                    {
                        "op": "assert_claim",
                        "subgraph": "infra_topology",
                        "predicate": "DEPENDS_ON",
                        "subject": {"key": "service:consumer", "type": "Service"},
                        "object": {"key": "service:provider", "type": "Service"},
                        "truth": "source_observation",
                        "description": "Consumer depends on provider",
                        "evidence": [
                            {"source_ref": f"potpie://res/{DOC}/one/0000@rev1"}
                        ],
                    }
                ]
            },
            pot_id=POT,
        )
    )
    assert conclusion.ok
    revision_before = host.backend.mutation.current_version(POT)
    replacement_source = _source(host, "two")
    mutation_type = type(host.backend.mutation)
    apply = mutation_type.compare_and_apply
    receipts = []

    def lose_response(self, *args, **kwargs):
        result = apply(self, *args, **kwargs)
        receipts.append(result)
        raise ConnectionError("graph committed but response was lost")

    with monkeypatch.context() as patch:
        patch.setattr(mutation_type, "compare_and_apply", lose_response)
        interrupted = (
            _import(host, replacement_source)
            if action == "replace"
            else host.facade.delete(pot_id=POT, slug=DOC)
        )

    assert len(receipts) == 1 and receipts[0].ok
    assert not interrupted.graph.ok
    assert host.backend.mutation.current_version(POT) == revision_before + 1
    assert host.store.current_manifest(pot_id=POT, slug=DOC).pending_review_refs
    if action == "delete":
        assert not interrupted.removed
        _assert_bytes(host, "one", revision=1)
    frozen = host.record(interrupted)
    assert frozen.prepared is not None
    assert any(
        edge.metadata_review_record_id for edge in frozen.lowered_batch.edge_upserts
    )
    assert frozen.lowered_batch.invalidations
    marked = host.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, claim_key_in=conclusion.claim_keys)
    )
    assert len(marked) == 1 and marked[0].properties["evidence_review_required"] is True
    assert marked[0].invalid_at is None  # evidence-only conclusions survive cleanup
    assert (
        host.backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, predicate_in=("DOCUMENTS",))
        )
        == []
    )
    version = host.backend.mutation.current_version(POT)
    plan_count = len(host.plans.list(pot_id=POT))
    host.compose(reopen=True)

    with monkeypatch.context() as patch:
        patch.setattr(ResourceFacade, "_dependent_claims", _reject_dependency_lookup)
        patch.setattr(
            resource_graph_mutations, "review_operations", _reject_dependency_lookup
        )
        recovered = (
            _import(host, replacement_source)
            if action == "replace"
            else host.facade.delete(pot_id=POT, slug=DOC)
        )

    assert recovered.graph.ok and recovered.graph.status == "applied"
    assert recovered.review_required_claim_keys == conclusion.claim_keys
    assert recovered.graph.plan_id == frozen.plan_id
    assert recovered.graph.mutation_id == frozen.mutation_id
    completed = host.record(recovered)
    assert completed.prepared == frozen.prepared
    assert completed.lowered_batch == frozen.lowered_batch
    assert completed.execution == receipts[0].execution
    assert len(host.plans.list(pot_id=POT)) == plan_count
    assert host.backend.mutation.current_version(POT) == version
    if action == "delete":
        assert recovered.removed and host.sections() == set()
    else:
        assert recovered.graph_written
        assert not host.store.current_manifest(pot_id=POT, slug=DOC).pending_review_refs
        assert host.sections() == {f"docsection:{DOC}:two"}


@pytest.mark.parametrize("status", ["rejected", "review_required"])
def test_delete_preserves_bytes_until_the_graph_result_is_applied(
    host, monkeypatch, status
):
    assert _import(host, _source(host, "one")).graph_written
    version = host.backend.mutation.current_version(POT)

    def refuse(_self, request, **_kwargs):
        return SemanticMutationResult(
            ok=status == "review_required", status=status, risk="medium", pot_id=POT
        )

    with monkeypatch.context() as patch:
        patch.setattr(DefaultGraphService, "mutate", refuse)
        deleted = host.facade.delete(pot_id=POT, slug=DOC)

    assert not deleted.removed and not deleted.graph_retracted
    assert deleted.graph.status == status
    _assert_bytes(host, "one", revision=1)
    assert host.sections() == {f"docsection:{DOC}:one"}
    assert host.backend.mutation.current_version(POT) == version


def test_late_old_delete_cannot_remove_a_newer_resource_revision(host, monkeypatch):
    assert _import(host, _source(host, "one")).graph_written
    later_source = _source(host, "two")
    reached_bytes = Event()
    release_delete = Event()
    store_type = type(host.store)
    delete = store_type.delete

    def pause_old_delete(self, *args, **kwargs):
        if self is host.store:
            assert kwargs["expected_revision"] == 1
            reached_bytes.set()
            assert release_delete.wait(timeout=10)
        return delete(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store_type, "delete", pause_old_delete)
        with ThreadPoolExecutor(max_workers=1) as pool:
            old_delete = pool.submit(host.facade.delete, pot_id=POT, slug=DOC)
            try:
                assert reached_bytes.wait(timeout=10)
                assert host.sections() == set()
                later = _import(host, later_source)
                assert later.graph_written and later.manifest.revision == 2
            finally:
                release_delete.set()
            with pytest.raises(ResourceStoreError) as stale:
                old_delete.result(timeout=10)

    assert stale.value.code == "resource_revision_changed"
    _assert_bytes(host, "two", revision=2)
    assert host.sections() == {f"docsection:{DOC}:two"}


def test_terminal_preparation_conflict_never_rebases_on_an_identical_retry(
    host, monkeypatch
):
    assert _import(host, _source(host, "one")).graph_written
    source = _source(host, "two")
    commit = GraphWorkbenchService.commit
    interfered = False

    def commit_after_concurrent_write(self, plan_id, **kwargs):
        nonlocal interfered
        if not interfered:
            interfered = True
            concurrent = host.runtime.graph.mutate(
                SemanticMutationRequest.parse(
                    {
                        "idempotency_key": "concurrent-write",
                        "operations": [
                            {
                                "op": "upsert_entity",
                                "subject": {"key": "service:later", "type": "Service"},
                            }
                        ],
                    },
                    pot_id=POT,
                )
            )
            assert concurrent.ok
        return commit(self, plan_id, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(GraphWorkbenchService, "commit", commit_after_concurrent_write)
        conflicted = _import(host, source)

    assert not conflicted.graph.ok and conflicted.graph.status == "conflict"
    frozen = host.record(conflicted)
    assert frozen.status == "conflict" and frozen.execution is None
    plan_count = len(host.plans.list(pot_id=POT))
    version = host.backend.mutation.current_version(POT)
    host.compose(reopen=True)

    with monkeypatch.context() as patch:
        patch.setattr(ResourceFacade, "_dependent_claims", _reject_dependency_lookup)
        retried = _import(host, source)

    assert not retried.graph.ok
    assert retried.graph.plan_id == frozen.plan_id
    assert host.record(retried).prepared == frozen.prepared
    assert host.record(retried).lowered_batch == frozen.lowered_batch
    assert len(host.plans.list(pot_id=POT)) == plan_count
    assert host.backend.mutation.current_version(POT) == version
    assert host.sections() == {f"docsection:{DOC}:one"}
    _assert_bytes(host, "two", revision=2)


def test_dependency_selection_is_inside_the_first_preparation_revision_bracket(
    host, monkeypatch
):
    assert _import(host, _source(host, "one")).graph_written
    _documents_claim(host)
    source = _source(host, "two")
    dependencies = ResourceFacade._dependent_claims
    selected = []
    before_version = host.backend.mutation.current_version(POT)

    def select_then_change_graph(self, **kwargs):
        selection = dependencies(self, **kwargs)
        selected.append(selection)
        concurrent = host.runtime.graph.mutate(
            SemanticMutationRequest.parse(
                {
                    "idempotency_key": "during-resource-selection",
                    "operations": [
                        {
                            "op": "upsert_entity",
                            "subject": {"key": "service:later", "type": "Service"},
                        }
                    ],
                },
                pot_id=POT,
            )
        )
        assert concurrent.ok
        return selection

    with monkeypatch.context() as patch:
        patch.setattr(ResourceFacade, "_dependent_claims", select_then_change_graph)
        conflicted = _import(host, source)

    assert len(selected) == 1
    assert [claim.predicate for claim in selected[0]] == ["DOCUMENTS"]
    assert not conflicted.graph.ok
    frozen = host.record(conflicted)
    assert frozen.status == "conflict" and frozen.execution is None
    assert frozen.prepared.expected_version == before_version
    assert ("DOCUMENTS", f"docsection:{DOC}:one", "service:reader") in {
        operation.target_edge for operation in frozen.lowered_batch.invalidations
    }
    assert host.backend.mutation.current_version(POT) == before_version + 1
    assert host.sections() == {f"docsection:{DOC}:one"}
    assert (
        len(
            host.backend.claim_query.find_claims(
                ClaimQueryFilter(pot_id=POT, predicate_in=("DOCUMENTS",))
            )
        )
        == 1
    )
    plan_count = len(host.plans.list(pot_id=POT))
    host.compose(reopen=True)

    with monkeypatch.context() as patch:
        patch.setattr(ResourceFacade, "_dependent_claims", _reject_dependency_lookup)
        retried = _import(host, source)

    assert not retried.graph.ok and retried.graph.plan_id == frozen.plan_id
    assert host.record(retried).prepared == frozen.prepared
    assert host.record(retried).lowered_batch == frozen.lowered_batch
    assert len(host.plans.list(pot_id=POT)) == plan_count
    assert host.backend.mutation.current_version(POT) == before_version + 1
    assert host.sections() == {f"docsection:{DOC}:one"}
    _assert_bytes(host, "two", revision=2)


def test_resource_cleanup_preserves_each_source_environment_incarnation(host):
    assert _import(host, _source(host, "one")).graph_written
    for environment, source in (
        ("prod", "source:a"),
        ("stage", "source:b"),
        (None, "source:c"),
    ):
        result = host.runtime.graph.mutate(
            SemanticMutationRequest.parse(
                {
                    "operations": [
                        {
                            "op": "assert_claim",
                            "subgraph": "knowledge",
                            "predicate": "DOCUMENTS",
                            "subject": {
                                "key": f"docsection:{DOC}:one",
                                "type": "DocumentSection",
                            },
                            "object": {"key": "service:reader", "type": "Service"},
                            "environment": environment,
                            "evidence": [{"source_ref": source}],
                            "description": "Reader docs",
                        }
                    ]
                },
                pot_id=POT,
            )
        )
        assert result.ok
    before = host.backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=POT, predicate_in=("DOCUMENTS",))
    )
    assert len(before) == 3
    replaced = _import(host, _source(host, "two"))
    assert replaced.graph_written
    targets = {
        c.target.record_id
        for op in host.record(replaced).lowered_batch.invalidations
        for c in op.closures
    }
    assert {row.record_id for row in before} <= targets
    assert (
        host.backend.claim_query.find_claims(
            ClaimQueryFilter(pot_id=POT, predicate_in=("DOCUMENTS",))
        )
        == []
    )
