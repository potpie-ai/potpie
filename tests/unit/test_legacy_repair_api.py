"""Typed repair preview/apply, request safety and administrative authorization."""

# ruff: noqa: S101 - pytest assertions

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from potpie.pots.contracts import PotInfo
from potpie.runtime import (
    ContextSelector,
    DestructiveConfirmation,
    LocalEngineClient,
    OperationCoordinator,
)
from potpie.runtime.commit_access import (
    authorize_local_commit,
    authorize_scoped_commit,
    authorization_scope,
    current_commit_actor,
    local_commit_actor,
)
from potpie.runtime.local_engine import (
    ServiceEngineOperations,
    build_local_resource_manager,
)
from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    EngineOperation,
    SafetyClass,
    effective_operation_spec,
    operation_schema,
)
from potpie.runtime.resource_manager import AuthorizationScope
from potpie_context_engine import (
    ContextEngine,
    ContextIdentity,
    EngineConfig,
    EngineDependencies,
    Failure,
    Success,
)
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
    load_store,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.runtime import (
    RuntimeCompositionError,
    build_graph_runtime,
)
from potpie_context_engine.requests import RepairRequest
from potpie_context_engine.results import RepairResult, result_from_payload
from tests.unit.test_local_engine import _Pots


def _compose(tmp_path, *, scoped=False):
    backend = InMemoryGraphBackend(
        store=load_store(
            {
                "format_version": "2",
                "rows": [],
                "labels": [["p", "service:a", ["Entity", "Service"]]],
                "entity_properties": [
                    [
                        "p",
                        "service:a",
                        {
                            "uuid": "existing-uuid",
                            "observed_at": "2025-01-01T05:30:00+05:30",
                        },
                    ]
                ],
            }
        )
    )
    runtime = build_graph_runtime(
        backend=backend,
        plan_store=LocalJsonGraphPlanStore(tmp_path),
        commit_authorize=authorize_scoped_commit if scoped else authorize_local_commit,
        commit_actor=current_commit_actor if scoped else local_commit_actor,
    )
    pot = PotInfo(pot_id="p", name="default", active=True)
    services = SimpleNamespace(
        backend=backend,
        graph_workbench=runtime.workbench,
        graph=runtime.graph,
        pots=_Pots([pot], pot),
    )
    return backend, services


def test_effective_safety_is_bound_to_typed_mode_and_published_in_catalog():
    spec = ENGINE_OPERATION_CATALOG[EngineOperation.REPAIR]
    assert spec.destructive
    preview = effective_operation_spec(spec, RepairRequest(legacy_mode="dry_run"))
    assert preview.safety == SafetyClass.SHARED_CONTEXT_READ and not preview.destructive
    assert effective_operation_spec(spec, RepairRequest()) == spec
    assert (
        effective_operation_spec(
            spec, RepairRequest(legacy_mode="apply", repair_id="run")
        )
        == spec
    )
    assert operation_schema(spec)["request_policy"] == "legacy_repair_v1"


@pytest.mark.parametrize(
    "values",
    [
        {"legacy_mode": "unknown"},
        {"legacy_mode": "apply"},
        {"repair_id": "run"},
        {"legacy_mode": "dry_run", "targets": ("entity_labels",)},
    ],
)
def test_invalid_repair_requests_fail_before_dispatch(values):
    with pytest.raises(ValueError):
        RepairRequest(**values)


@pytest.mark.anyio
async def test_public_local_client_previews_without_confirmation_and_applies_with_it(
    tmp_path,
):
    backend, services = _compose(tmp_path)
    manager = build_local_resource_manager(services)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="p"),
        authentication={"kind": "local_cli"},
        resource_manager=manager,
        coordinator=OperationCoordinator(),
    )
    try:
        preview = await client.repair(RepairRequest(legacy_mode="dry_run"))
        assert isinstance(preview, Success), preview
        assert preview.value.status == "ready" and preview.value.repairable_records == 1
        assert backend.mutation.current_version("p") == 0
        request = RepairRequest(legacy_mode="apply", repair_id=preview.value.repair_id)
        rejected = await client.repair(request)
        assert (
            isinstance(rejected, Failure)
            and rejected.error.code == "destructive_intent_invalid"
        )
        assert backend.mutation.current_version("p") == 0
        applied = await client.repair(
            request, confirmation=DestructiveConfirmation(confirmed=True)
        )
        assert isinstance(applied, Success), applied
        assert (
            applied.value.status == "completed" and applied.value.completed_batches == 1
        )
        decoded = result_from_payload(RepairResult, asdict(applied.value))
        assert decoded == applied.value
        replay = await client.repair(
            request, confirmation=DestructiveConfirmation(confirmed=True)
        )
        assert replay.value.executions == applied.value.executions
        assert backend.mutation.current_version("p") == 1
        second = await client.repair(RepairRequest(legacy_mode="dry_run"))
        assert second.value.repairable_records == 0
    finally:
        await manager.shutdown()


@pytest.mark.anyio
async def test_reader_can_preview_but_admin_required_for_apply_through_facade(tmp_path):
    backend, services = _compose(tmp_path, scoped=True)
    operations = ServiceEngineOperations(services)
    identity = ContextIdentity("p")
    engine = ContextEngine(
        context=identity,
        config=EngineConfig(),
        dependencies=EngineDependencies(
            context=operations,
            graph=operations,
            workbench=operations,
            ingestion=operations,
            nudge=operations,
        ),
    )
    reader = AuthorizationScope(
        actor_id="reader",
        operation="repair",
        context=identity,
        attributes={"commit_access": "reader"},
    )
    with authorization_scope(reader):
        preview = await engine.repair(RepairRequest(legacy_mode="dry_run"))
        assert isinstance(preview, Success), preview
        denied = await engine.repair(
            RepairRequest(legacy_mode="apply", repair_id=preview.value.repair_id)
        )
        assert (
            isinstance(denied, Failure) and denied.error.code == "commit_access_denied"
        )
    assert backend.mutation.current_version("p") == 0
    admin = AuthorizationScope(
        actor_id="admin",
        operation="repair",
        context=identity,
        attributes={"commit_access": "admin"},
    )
    with authorization_scope(admin):
        result = await engine.repair(
            RepairRequest(legacy_mode="apply", repair_id=preview.value.repair_id)
        )
    assert isinstance(result, Success) and result.value.ok
    assert backend.mutation.current_version("p") == 1
    await engine.close()


@pytest.mark.anyio
async def test_dry_run_refuses_destructive_confirmation(tmp_path):
    _, services = _compose(tmp_path)
    manager = build_local_resource_manager(services)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="p"),
        authentication={"kind": "local_cli"},
        resource_manager=manager,
        coordinator=OperationCoordinator(),
    )
    result = await client.repair(
        RepairRequest(legacy_mode="dry_run"),
        confirmation=DestructiveConfirmation(confirmed=True),
    )
    assert (
        isinstance(result, Failure)
        and result.error.code == "destructive_confirmation_unexpected"
    )
    await manager.shutdown()


def test_sync_repair_facade_cannot_bypass_authorization(tmp_path):
    backend, services = _compose(tmp_path, scoped=True)
    reader = AuthorizationScope(
        actor_id="reader",
        operation="repair",
        context=ContextIdentity("p"),
        attributes={"commit_access": "reader"},
    )
    with authorization_scope(reader):
        preview = services.graph_workbench.repair_legacy(pot_id="p", mode="dry_run")
        with pytest.raises(PermissionError):
            services.graph_workbench.repair_legacy(
                pot_id="p", mode="apply", repair_id=preview.repair_id
            )
    assert backend.mutation.current_version("p") == 0


@pytest.mark.anyio
async def test_sync_repair_facade_refuses_active_loop_without_side_effects(tmp_path):
    backend, services = _compose(tmp_path)
    with pytest.raises(RuntimeCompositionError, match="async"):
        services.graph_workbench.repair_legacy(pot_id="p", mode="dry_run")
    assert backend.mutation.current_version("p") == 0


@pytest.mark.anyio
async def test_cancelled_public_repair_recovers_receipt_before_late_finalizer(
    tmp_path, monkeypatch
):
    import asyncio
    import threading

    backend, services = _compose(tmp_path)
    manager = build_local_resource_manager(services)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="p"),
        authentication={"kind": "local_cli"},
        resource_manager=manager,
        coordinator=OperationCoordinator(),
    )
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = LocalJsonGraphPlanStore.compare_and_set

    def pause_finalization(store, *, expected, replacement):
        if replacement.status == "committed" and not entered.is_set():
            entered.set()
            try:
                if not release.wait(15):
                    raise TimeoutError(
                        "acceptance finalization barrier was not released"
                    )
                return original(store, expected=expected, replacement=replacement)
            finally:
                finished.set()
        return original(store, expected=expected, replacement=replacement)

    try:
        preview = await client.repair(RepairRequest(legacy_mode="dry_run"))
        request = RepairRequest(legacy_mode="apply", repair_id=preview.value.repair_id)
        confirmation = DestructiveConfirmation(confirmed=True)
        monkeypatch.setattr(
            LocalJsonGraphPlanStore, "compare_and_set", pause_finalization
        )
        task = asyncio.create_task(client.repair(request, confirmation=confirmation))
        assert await asyncio.to_thread(entered.wait, 10)
        assert backend.mutation.current_version("p") == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        recovered = await client.repair(request, confirmation=confirmation)
        assert isinstance(recovered, Success) and recovered.value.ok
        assert recovered.value.completed_batches == 1
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
        replay = await client.repair(request, confirmation=confirmation)
        assert replay.value.executions == recovered.value.executions
        assert backend.mutation.current_version("p") == 1
    finally:
        release.set()
        await manager.shutdown()
