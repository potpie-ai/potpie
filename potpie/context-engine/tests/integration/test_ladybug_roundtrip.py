"""Native Ladybug persistence, transaction rollback and lifecycle acceptance."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

if os.environ.get("POTPIE_REQUIRE_LADYBUG") != "1":
    pytest.importorskip("ladybug")

from potpie_context_engine.adapters.outbound.graph import ladybug_storage
from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import (
    LadybugGraphBackend,
)
from potpie_context_engine.core.graph_mutations import (
    EdgeUpsert,
    EntityUpsert,
    InvalidationOp,
    ProvenanceContext,
)
from potpie_context_engine.core.mutation_execution import MutationPersistenceError
from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.reconciliation import MutationBatch


def batch(*, date="2026-01-01T00:00:00+00:00", fact="API depends on DB"):
    return MutationBatch(
        entity_upserts=[
            EntityUpsert("service:api", ("Service",)),
            EntityUpsert("service:db", ("Service",)),
        ],
        edge_upserts=[
            EdgeUpsert(
                "DEPENDS_ON",
                "service:api",
                "service:db",
                {
                    "claim_key": "claim:dependency",
                    "subgraph": "infra_topology",
                    "truth": "source_observation",
                    "source_ref": "fixture:dependency",
                    "source_refs": ["fixture:dependency"],
                    "environment": "prod",
                    "valid_at": date,
                    "valid_from": date,
                    "description": fact,
                    "fact": fact,
                },
            )
        ],
    )


def apply(backend, *, pot="p", mutation="once", plan=None):
    return backend.mutation.apply(
        plan or batch(),
        expected_pot_id=pot,
        provenance_context=ProvenanceContext(mutation_id=mutation),
    )


def rows(backend, pot="p"):
    return backend.claim_query.find_claims(
        ClaimQueryFilter(pot_id=pot, include_invalidated=True)
    )


def test_reopen_receipt_revision_and_native_graph(tmp_path):
    backend = LadybugGraphBackend(home=tmp_path, db_path=tmp_path / "custom.db")
    first = apply(backend)
    reopened = LadybugGraphBackend(home=tmp_path, db_path=tmp_path / "custom.db")
    assert apply(reopened) == first
    assert reopened.mutation.current_version("p") == 1
    assert len(rows(reopened)) == 1
    with reopened._locked_store():
        conn = reopened._connection
        for table, expected in (
            ("Entity", 2),
            ("Claim", 1),
            ("ExecutionReceipt", 1),
            ("GraphRevision", 1),
        ):
            assert (
                ladybug_storage.execute(
                    conn, f"MATCH (n:{table}) RETURN count(n) AS count"
                )[0]["count"]
                == expected
            )
        for rel in ("CLAIM_SUBJECT", "CLAIM_OBJECT"):
            assert (
                ladybug_storage.execute(
                    conn, f"MATCH ()-[r:{rel}]->() RETURN count(r) AS count"
                )[0]["count"]
                == 1
            )


def test_pots_with_identical_keys_remain_isolated(tmp_path):
    backend = LadybugGraphBackend(home=tmp_path)
    apply(backend, pot="a")
    apply(backend, pot="b")
    backend.mutation.reset_pot("a")
    assert not rows(backend, "a")
    assert len(rows(LadybugGraphBackend(home=tmp_path), "b")) == 1


def test_write_failure_rolls_back_native_state_and_live_snapshot(tmp_path, monkeypatch):
    backend = LadybugGraphBackend(home=tmp_path)
    apply(backend)
    before = rows(backend)
    original = ladybug_storage.execute

    def fail_after_deletion(conn, query, params=None):
        if query.startswith("CREATE (:Claim"):
            raise OSError("injected claim persistence failure")
        return original(conn, query, params)

    with monkeypatch.context() as patch:
        patch.setattr(ladybug_storage, "execute", fail_after_deletion)
        with pytest.raises(MutationPersistenceError):
            apply(
                backend, mutation="failed", plan=batch(date="2026-02-01T00:00:00+00:00")
            )
    assert backend._inner.store.rows == before
    assert rows(backend) == before
    reopened = LadybugGraphBackend(home=tmp_path)
    assert rows(reopened) == before
    assert reopened.mutation.current_version("p") == 1
    assert (
        reopened.mutation.lookup_execution(
            batch(date="2026-02-01T00:00:00+00:00"),
            expected_pot_id="p",
            mutation_id="failed",
        ).state
        == "absent"
    )


def test_snapshot_and_journal_survive_reopen(tmp_path):
    backend = LadybugGraphBackend(home=tmp_path)
    apply(backend)
    activated = backend.journal.activate(pot_id="p")
    assert backend.journal.journal_capability("p").durable
    apply(
        backend,
        mutation="close",
        plan=MutationBatch(
            invalidations=[
                InvalidationOp(
                    target_entity_key=None,
                    target_edge=None,
                    reason="end observation",
                    valid_to="2026-01-15T00:00:00+00:00",
                    target_claim_keys=("claim:dependency",),
                )
            ]
        ),
    )
    apply(backend, mutation="next", plan=batch(date="2026-02-01T00:00:00+00:00"))
    reopened = LadybugGraphBackend(home=tmp_path)
    state = reopened.journal.journal_state("p")
    assert state.generation == activated.generation
    assert state.sequence == 2
    historical = rows(reopened)
    assert len(historical) == 2
    assert len({r.record_id for r in historical}) == 2
    assert sum(r.invalid_at is not None for r in historical) == 1
    payload = reopened.snapshot.export_data(pot_id="p")
    other = LadybugGraphBackend(home=tmp_path / "other")
    other.snapshot.import_data(pot_id="copy", payload=payload)
    assert len(rows(other, "copy")) == len(rows(reopened))
    assert (
        reopened.profile
        == reopened.bind_definition(reopened.definition).profile
        == "ladybug"
    )
    assert reopened.capabilities().profile == "ladybug"


def test_stale_version_rejected_and_completed_replay_wins(tmp_path):
    from potpie_context_engine.core.errors import GraphMutationVersionConflict

    backend = LadybugGraphBackend(home=tmp_path)
    first = apply(backend)
    reopened = LadybugGraphBackend(home=tmp_path)
    assert (
        reopened.mutation.compare_and_apply(
            batch(),
            expected_pot_id="p",
            expected_version=0,
            provenance_context=ProvenanceContext(mutation_id="once"),
        )
        == first
    )
    with pytest.raises(GraphMutationVersionConflict):
        reopened.mutation.compare_and_apply(
            batch(),
            expected_pot_id="p",
            expected_version=0,
            provenance_context=ProvenanceContext(mutation_id="stale"),
        )
    assert reopened.mutation.current_version("p") == 1


def test_journal_restore_and_receipt_retry_survive_reopen(tmp_path):
    backend = LadybugGraphBackend(home=tmp_path)
    backend.journal.activate(pot_id="p", rollback_enabled=True)
    for mutation, summary in (("initial", "before"), ("changed", "after")):
        apply(
            backend,
            mutation=mutation,
            plan=MutationBatch(
                entity_upserts=[
                    EntityUpsert("service:api", ("Service",), {"summary": summary}),
                ]
            ),
        )
    reopened = LadybugGraphBackend(home=tmp_path)
    plan = reopened.journal.plan_restore(
        pot_id="p", target_commit_id="changed", expected_head="changed"
    )
    receipt = reopened.journal.apply_restore(plan, mutation_id="undo", actor="test")
    again = LadybugGraphBackend(home=tmp_path)
    assert (
        again.journal.apply_restore(plan, mutation_id="undo", actor="test") == receipt
    )
    assert again.journal.journal_state("p").head == "undo"
    assert (
        again.claim_query.entity_properties(pot_id="p", entity_key="service:api")[
            "summary"
        ]
        == "before"
    )


def test_commit_acknowledgement_failure_recovers_from_native_receipt(
    tmp_path, monkeypatch
):
    from potpie_context_engine.adapters.outbound.graph.backends import ladybug_backend

    backend = LadybugGraphBackend(home=tmp_path)
    original = ladybug_backend.execute

    def lose_acknowledgement(conn, query, params=None):
        value = original(conn, query, params)
        if query == "COMMIT":
            raise OSError("injected lost commit acknowledgement")
        return value

    with monkeypatch.context() as patch:
        patch.setattr(ladybug_backend, "execute", lose_acknowledgement)
        with pytest.raises(MutationPersistenceError):
            apply(backend)
    assert backend._inner.store.rows == []
    recovered = apply(backend)
    assert recovered.ok
    assert backend.mutation.current_version("p") == 1
    assert len(rows(backend)) == 1
    assert apply(LadybugGraphBackend(home=tmp_path)) == recovered


def test_separate_process_can_open_after_parent_operation(tmp_path):
    backend = LadybugGraphBackend(home=tmp_path)
    first = apply(backend)
    code = """
from pathlib import Path
import sys
from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import LadybugGraphBackend
from potpie_context_engine.core.graph_mutations import ProvenanceContext
from potpie_context_engine.core.reconciliation import MutationBatch
b = LadybugGraphBackend(home=Path(sys.argv[1]))
assert b.mutation.current_version('p') == 1
b.mutation.apply(MutationBatch(reset_pot=True), expected_pot_id='p', provenance_context=ProvenanceContext(mutation_id='child-reset'))
assert b.mutation.current_version('p') == 2
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert backend.mutation.current_version("p") == 2
    assert rows(backend) == []
    # Reset preserves original execution history for recovery.
    assert apply(backend) == first
    assert rows(backend) == []


def test_killed_native_writer_rolls_back_graph_and_keeps_receipt(tmp_path):
    backend = LadybugGraphBackend(home=tmp_path)
    first = apply(backend)
    code = """
from pathlib import Path
import os
import sys
from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import LadybugGraphBackend
from potpie_context_engine.adapters.outbound.graph.ladybug_storage import execute
b = LadybugGraphBackend(home=Path(sys.argv[1]))
with b._locked_store():
    execute(b._connection, 'MATCH (e:Entity) DETACH DELETE e')
    os._exit(55)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 55, completed.stderr
    reopened = LadybugGraphBackend(home=tmp_path)
    assert len(rows(reopened)) == 1
    assert reopened.mutation.current_version("p") == 1
    assert apply(reopened) == first
    with reopened._locked_store():
        assert (
            ladybug_storage.execute(
                reopened._connection,
                "MATCH ()-[r:CLAIM_SUBJECT]->() RETURN count(r) AS count",
            )[0]["count"]
            == 1
        )


@pytest.mark.parametrize("same_execution", [True, False])
def test_concurrent_process_mutations_are_serialized(tmp_path, same_execution):
    backend = LadybugGraphBackend(home=tmp_path)
    code = """
from pathlib import Path
import sys
import time
from potpie_context_engine.adapters.outbound.graph.backends.ladybug_backend import LadybugGraphBackend
from potpie_context_engine.core.graph_mutations import EdgeUpsert, EntityUpsert, ProvenanceContext
from potpie_context_engine.core.reconciliation import MutationBatch
home, worker, same = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'
b = LadybugGraphBackend(home=home)
(home / (worker + '.ready')).touch()
deadline = time.monotonic() + 20
while not (home / 'go').exists():
    assert time.monotonic() < deadline, 'parent did not release workers'
    time.sleep(0.01)
key = 'same' if same else worker
plan = MutationBatch(
    entity_upserts=[EntityUpsert('service:' + key, ('Service',)), EntityUpsert('service:db', ('Service',))],
    edge_upserts=[EdgeUpsert('DEPENDS_ON', 'service:' + key, 'service:db', {'claim_key': 'claim:' + key, 'source_ref': 'fixture:' + key, 'truth': 'source_observation'})],
)
assert b.mutation.apply(plan, expected_pot_id='p', provenance_context=ProvenanceContext(mutation_id=key)).ok
"""
    workers = []
    try:
        for name in ("a", "b"):
            workers.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        code,
                        str(tmp_path),
                        name,
                        str(same_execution),
                    ],
                    env={
                        **os.environ,
                        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        deadline = time.monotonic() + 20
        while not all((tmp_path / (name + ".ready")).exists() for name in ("a", "b")):
            assert time.monotonic() < deadline, "children did not reach the barrier"
            assert all(worker.poll() is None for worker in workers), (
                "child exited before barrier"
            )
            time.sleep(0.01)
        (tmp_path / "go").touch()
        for worker in workers:
            _, stderr = worker.communicate(timeout=30)
            assert worker.returncode == 0, stderr
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=10)
    expected = 1 if same_execution else 2
    assert backend.mutation.current_version("p") == expected
    assert len(rows(backend)) == expected
    with backend._locked_store():
        assert (
            ladybug_storage.execute(
                backend._connection,
                "MATCH (r:ExecutionReceipt) RETURN count(r) AS count",
            )[0]["count"]
            == expected
        )
