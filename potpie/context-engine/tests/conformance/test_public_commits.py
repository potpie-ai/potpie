"""Public previews across restart, permissions, races, and rollback undo."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from potpie_context_engine.core.commit_service import GraphCommitService
from potpie_context_engine.core.graph_mutations import EntityUpsert, ProvenanceContext
from potpie_context_engine.core.journal_context import (
    JournalWriteContext,
    journal_write_context,
)
from potpie_context_engine.core.reconciliation import MutationBatch

from potpie_context_engine.adapters.outbound.graph.backends.embedded_backend import (
    EmbeddedGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.local_commit_mirror import (
    LocalCommitMirror,
)
from potpie_context_engine.adapters.outbound.graph.local_rollback_previews import (
    LocalRollbackPreviews,
)


def setup(tmp_path):
    backend = EmbeddedGraphBackend(home=tmp_path)
    backend.journal.activate(pot_id="p", rollback_enabled=True)
    identity = {"actor": "alice", "access": "admin"}

    async def authorize(pot, access):
        if (
            pot != "p"
            or {"read": 0, "write": 1, "admin": 2}[access]
            > {"read": 0, "write": 1, "admin": 2}[identity["access"]]
        ):
            raise PermissionError("insufficient grant")

    service = GraphCommitService(
        journal=backend.journal,
        mirror=LocalCommitMirror(tmp_path / "commits.sqlite"),
        previews=LocalRollbackPreviews(tmp_path / "previews.sqlite"),
        host="local:test",
        actor=lambda: identity["actor"],
        authorize=authorize,
    )
    mutate(backend, "c1", "before")
    mutate(backend, "c2", "after")
    return backend, service, identity


def mutate(backend, commit, summary):
    backend.mutation.apply(
        MutationBatch(
            entity_upserts=[
                EntityUpsert("service:a", ("Entity", "Service"), {"summary": summary})
            ]
        ),
        expected_pot_id="p",
        provenance_context=ProvenanceContext(mutation_id=commit, actor_user_id="alice"),
    )


async def preview(service, target="c2", head="c2"):
    result = await service.revert_preview_async(target, pot_id="p", expected_head=head)
    assert result["ok"], result
    return result["preview"].preview_id


async def test_restart_apply_retry_and_undo(tmp_path):
    _backend, service, _ = setup(tmp_path)
    key = await preview(service)
    restarted = EmbeddedGraphBackend(home=tmp_path)
    service.journal = restarted.journal
    service.previews = LocalRollbackPreviews(tmp_path / "previews.sqlite")
    result = await service.apply_preview_async(key, pot_id="p")
    assert result["ok"] and not result["replayed"]
    assert (
        restarted.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "before"
    )
    # Success survives expiry and the stale HEAD caused by the apply itself.
    service.clock = lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    retry = await service.apply_preview_async(key, pot_id="p")
    assert retry["replayed"] and retry["commit"] == result["commit"]
    service.clock = lambda: datetime.now(timezone.utc)
    undo_key = await preview(
        service, result["commit"]["commit_id"], result["commit"]["commit_id"]
    )
    assert (await service.apply_preview_async(undo_key, pot_id="p"))["ok"]
    assert (
        restarted.claim_query.entity_properties(pot_id="p", entity_key="service:a")[
            "summary"
        ]
        == "after"
    )
    history = await service.commits_async(pot_id="p", limit=2)
    assert history["coverage"]["complete"] and len(history["headers"]) == 2
    detail = await service.commit_show_async(result["commit"]["commit_id"], pot_id="p")
    assert detail["historical_view"].startswith("partial")


async def test_binding_expiry_stale_and_revoked_grants(tmp_path):
    backend, service, identity = setup(tmp_path)
    key = await preview(service)
    identity["actor"] = "bob"
    assert (await service.apply_preview_async(key, pot_id="p"))[
        "status"
    ] == "preview_scope_mismatch"
    identity["actor"] = "alice"
    identity["access"] = "read"
    with pytest.raises(PermissionError):
        await service.apply_preview_async(key, pot_id="p")
    with pytest.raises(PermissionError):
        await preview(service)
    identity["access"] = "write"
    service.clock = lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    assert (await service.apply_preview_async(key, pot_id="p"))[
        "status"
    ] == "preview_expired"
    service.clock = lambda: datetime.now(timezone.utc)
    mutate(backend, "c3", "later")
    assert (await service.apply_preview_async(key, pot_id="p"))[
        "status"
    ] == "preview_stale"
    assert backend.journal.journal_state("p").sequence == 3


async def test_admin_effects_cannot_be_undone_by_writer(tmp_path):
    backend, service, identity = setup(tmp_path)
    with journal_write_context(JournalWriteContext(required_access="admin")):
        mutate(backend, "admin", "admin change")
    identity["access"] = "write"
    with pytest.raises(PermissionError):
        await preview(service, "admin", "admin")
    identity["access"] = "admin"
    key = await preview(service, "admin", "admin")
    identity["access"] = "write"
    with pytest.raises(PermissionError):
        await service.apply_preview_async(key, pot_id="p")


async def test_resource_generation_and_pending_guard_fence_preview(tmp_path):
    backend, service, _ = setup(tmp_path)
    key = await preview(service)
    backend.journal.begin_resource(pot_id="p", operation_id="import", owner="worker")
    assert (await service.apply_preview_async(key, pot_id="p"))[
        "status"
    ] == "preview_stale"
    blocked = await service.revert_preview_async("c2", pot_id="p", expected_head="c2")
    assert not blocked["ok"] and "resource" in blocked["reasons"][0]["message"]
    barrier = backend.journal.complete_resource(
        pot_id="p", operation_id="import", owner="worker"
    )
    crossed = await service.rollback_preview_async(
        "c1", pot_id="p", expected_head=barrier.commit_id
    )
    assert not crossed["ok"]
    safe = await service.revert_preview_async(
        "c2", pot_id="p", expected_head=barrier.commit_id
    )
    assert safe["ok"]


async def test_concurrent_apply_has_one_native_commit(tmp_path):
    backend, service, _ = setup(tmp_path)
    key = await preview(service)
    results = await asyncio.gather(
        *(service.apply_preview_async(key, pot_id="p") for _ in range(4))
    )
    assert all(r["ok"] for r in results), results
    assert len({r["commit"]["commit_id"] for r in results}) == 1
    assert backend.journal.journal_state("p").sequence == 3


def test_a_runtime_without_host_commit_wiring_refuses_history_and_rollback(tmp_path):
    import getpass

    from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
        LocalJsonGraphPlanStore,
    )
    from potpie_context_engine.core.commit_service import CommitAccessDenied
    from potpie_context_engine.core.runtime import build_graph_runtime

    runtime = build_graph_runtime(
        EmbeddedGraphBackend(home=tmp_path),
        LocalJsonGraphPlanStore(home=tmp_path),
        commit_mirror=LocalCommitMirror(tmp_path / "index.sqlite"),
        preview_store=LocalRollbackPreviews(tmp_path / "previews.sqlite"),
    )
    runtime.backend.journal.activate(pot_id="p", rollback_enabled=True)

    for call in (
        lambda: runtime.workbench.journal_status_async(pot_id="p"),
        lambda: runtime.workbench.commits_async(pot_id="p"),
        lambda: runtime.workbench.revert_preview_async(
            "c1", pot_id="p", expected_head="c1"
        ),
        lambda: runtime.workbench.apply_preview_async("preview", pot_id="p"),
    ):
        with pytest.raises(CommitAccessDenied):
            asyncio.run(call())
    # An unnamed actor is recorded as such, never as the process owner.
    assert runtime.commit_service.actor() == "unnamed"
    assert getpass.getuser() not in runtime.commit_service.actor()
