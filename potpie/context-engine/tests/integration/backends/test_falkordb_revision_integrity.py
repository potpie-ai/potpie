# ruff: noqa: F811 - imported pytest fixtures are injected as test parameters.
"""Modern native receipts prevent deleted or regressed revision reuse."""

import pytest

from potpie_context_engine.adapters.outbound.graph.falkordb_atomic import _ensure_state
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.reconciliation import MutationBatch

from tests.integration.backends.test_falkordb_reliability import (
    backend,
    batch,
    context,
    graph as graph,
    snapshot,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("corruption", ("missing", "zero", "regressed"))
def test_modern_receipt_floor_blocks_revision_reads_lookup_bootstrap_and_execution(
    graph, corruption
):
    store, first, second = backend(graph), batch(), batch("second")
    store.mutation.compare_and_apply(
        first,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=context(first),
    )
    store.mutation.compare_and_apply(
        second,
        expected_pot_id="p",
        expected_version=1,
        provenance_context=context(second, "m2", version=1, query=store.claim_query),
    )
    if corruption == "missing":
        graph.query("MATCH (v:PotpieRevision {pot_id:'p'}) DELETE v")
        expected = 0
    else:
        expected = 0 if corruption == "zero" else 1
        graph.query(
            "MATCH (v:PotpieRevision {pot_id:'p'}) SET v.version=$version",
            params={"version": expected},
        )
    before, third = snapshot(graph), batch("must not commit")
    operations = (
        lambda: store.mutation.current_version("p"),
        lambda: store.mutation.lookup_execution(
            first, expected_pot_id="p", mutation_id="m1"
        ),
        lambda: store.mutation.lookup_execution(
            third, expected_pot_id="p", mutation_id="unseen"
        ),
        lambda: _ensure_state(graph, "p"),
        lambda: store.mutation.compare_and_apply(
            third,
            expected_pot_id="p",
            expected_version=expected,
            provenance_context=context(third, "m3", version=expected),
        ),
        lambda: store.mutation.compare_and_apply(
            first,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(first),
        ),
        lambda: store.mutation.reset_pot("p", mutation_id="reset-corrupt"),
        lambda: store.snapshot.export_data(pot_id="p"),
    )
    for operation in operations:
        with pytest.raises(MutationPersistenceError, match="durable execution floor 2"):
            operation()
        assert snapshot(graph) == before


def test_modern_empty_receipt_requires_revision_even_when_floor_is_zero(graph):
    store, empty = backend(graph), MutationBatch()
    store.mutation.compare_and_apply(
        empty,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=context(empty),
    )
    graph.query("MATCH (v:PotpieRevision {pot_id:'p'}) DELETE v")
    with pytest.raises(MutationPersistenceError, match="durable execution floor 0"):
        store.mutation.current_version("p")
    with pytest.raises(MutationPersistenceError, match="durable execution floor 0"):
        _ensure_state(graph, "p")
    assert graph.ro_query(
        "MATCH (v:PotpieRevision {pot_id:'p'}) RETURN count(v)"
    ).result_set == [[0]]


def test_revision_floor_is_pot_scoped(graph):
    store, plan = backend(graph), batch()
    store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    graph.query("MATCH (v:PotpieRevision {pot_id:'p'}) DELETE v")
    assert store.mutation.current_version("unrelated") == 0
    _ensure_state(graph, "unrelated")
    assert store.mutation.current_version("unrelated") == 0


def test_journal_publication_cannot_advance_regressed_revision(graph):
    store, plan = backend(graph), batch()
    store.journal.activate(pot_id="p", rollback_enabled=True)
    store.mutation.compare_and_apply(
        plan, expected_pot_id="p", expected_version=0, provenance_context=context(plan)
    )
    graph.query("MATCH (v:PotpieRevision {pot_id:'p'}) SET v.version=0")
    before = snapshot(graph)
    with pytest.raises(MutationPersistenceError, match="durable execution floor 1"):
        store.journal.set_rollback_enabled(pot_id="p", enabled=False)
    assert snapshot(graph) == before


def test_duplicate_nested_json_key_cannot_lower_modern_receipt_revision_floor(graph):
    store, first = backend(graph), batch()
    store.mutation.compare_and_apply(
        first,
        expected_pot_id="p",
        expected_version=0,
        provenance_context=context(first),
    )
    payload = graph.ro_query(
        "MATCH (r:PotpieMutationReceipt {pot_id:'p',mutation_id:'m1'}) RETURN r.payload"
    ).result_set[0][0]
    marker = '"version_after": 1'
    assert payload.count(marker) == 1
    damaged = payload.replace(marker, '"version_after": 1, "version_after": 0')
    graph.query(
        "MATCH (r:PotpieMutationReceipt {pot_id:'p',mutation_id:'m1'}),"
        "(v:PotpieRevision {pot_id:'p'}) SET r.payload=$payload,v.version=0",
        params={"payload": damaged},
    )
    before, second = snapshot(graph), batch("must not commit")
    operations = (
        lambda: store.mutation.current_version("p"),
        lambda: store.mutation.lookup_execution(
            first, expected_pot_id="p", mutation_id="m1"
        ),
        lambda: store.mutation.lookup_execution(
            second, expected_pot_id="p", mutation_id="m2"
        ),
        lambda: _ensure_state(graph, "p"),
        lambda: store.mutation.compare_and_apply(
            second,
            expected_pot_id="p",
            expected_version=0,
            provenance_context=context(second, "m2"),
        ),
    )
    for operation in operations:
        with pytest.raises(
            MutationPersistenceError, match="malformed native execution receipt"
        ) as raised:
            operation()
        assert isinstance(raised.value.__cause__, ValueError)
        assert "duplicate storage key: version_after" in str(raised.value.__cause__)
        assert snapshot(graph) == before
