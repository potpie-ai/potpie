from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

import asyncio

import pytest

from potpie.runtime import (
    ENGINE_OPERATION_CATALOG,
    EngineOperation,
    OperationCoordinator,
    OperationSpec,
    SafetyClass,
)
from potpie_context_engine import ContextIdentity
from potpie_context_engine.requests import (
    ExportSnapshotRequest,
    RecordRequest,
    SearchRequest,
)


@pytest.mark.anyio
async def test_shared_reads_for_same_context_run_together() -> None:
    coordinator = OperationCoordinator()
    gate = asyncio.Event()
    both_entered = asyncio.Event()
    entered = 0

    async def read() -> None:
        nonlocal entered
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.SEARCH],
            context=ContextIdentity("context-a"),
            request=SearchRequest(),
        ):
            entered += 1
            if entered == 2:
                both_entered.set()
            await gate.wait()

    first = asyncio.create_task(read())
    second = asyncio.create_task(read())
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    gate.set()
    await asyncio.gather(first, second)

    assert entered == 2


@pytest.mark.anyio
async def test_same_context_write_waits_for_shared_read() -> None:
    coordinator = OperationCoordinator()
    read_entered = asyncio.Event()
    release_read = asyncio.Event()
    write_entered = asyncio.Event()

    async def read() -> None:
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.SEARCH],
            context=ContextIdentity("context-a"),
            request=SearchRequest(),
        ):
            read_entered.set()
            await release_read.wait()

    async def write() -> None:
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.RECORD],
            context=ContextIdentity("context-a"),
            request=RecordRequest(),
        ):
            write_entered.set()

    read_task = asyncio.create_task(read())
    await read_entered.wait()
    write_task = asyncio.create_task(write())
    await asyncio.sleep(0)

    assert not write_entered.is_set()

    release_read.set()
    await asyncio.gather(read_task, write_task)
    assert write_entered.is_set()


@pytest.mark.anyio
async def test_writes_for_unrelated_contexts_run_together() -> None:
    coordinator = OperationCoordinator()
    gate = asyncio.Event()
    both_entered = asyncio.Event()
    entered: set[str] = set()

    async def write(context_value: str) -> None:
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.RECORD],
            context=ContextIdentity(context_value),
            request=RecordRequest(),
        ):
            entered.add(context_value)
            if len(entered) == 2:
                both_entered.set()
            await gate.wait()

    first = asyncio.create_task(write("context-a"))
    second = asyncio.create_task(write("context-b"))
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    gate.set()
    await asyncio.gather(first, second)

    assert entered == {"context-a", "context-b"}


@pytest.mark.anyio
async def test_resource_mutations_conflict_on_typed_resource_identity() -> None:
    coordinator = OperationCoordinator()
    spec = OperationSpec(
        operation=EngineOperation.RECORD,
        request_type=RecordRequest,
        result_type=object,
        safety=SafetyClass.EXCLUSIVE_RESOURCE_MUTATION,
        resource_type="source",
        resource_identity_fields=("idempotency_key",),
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with coordinator.coordinate(
            spec=spec,
            context=ContextIdentity("context-a"),
            request=RecordRequest(idempotency_key="source-1"),
        ):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with coordinator.coordinate(
            spec=spec,
            context=ContextIdentity("context-b"),
            request=RecordRequest(idempotency_key="source-1"),
        ):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)

    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set()


@pytest.mark.anyio
async def test_daemon_lifecycle_controls_are_process_exclusive() -> None:
    coordinator = OperationCoordinator()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with coordinator.lifecycle_control():
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with coordinator.lifecycle_control():
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)

    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set()


@pytest.mark.anyio
async def test_snapshot_exports_serialize_by_normalized_destination(tmp_path) -> None:
    coordinator = OperationCoordinator()
    spec = ENGINE_OPERATION_CATALOG[EngineOperation.EXPORT_SNAPSHOT]
    destination = tmp_path / "snapshot.json"
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with coordinator.coordinate(
            spec=spec,
            context=ContextIdentity("context-a"),
            request=ExportSnapshotRequest(destination=str(destination)),
        ):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        equivalent = destination.parent / "." / destination.name
        async with coordinator.coordinate(
            spec=spec,
            context=ContextIdentity("context-b"),
            request=ExportSnapshotRequest(destination=str(equivalent)),
        ):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)

    assert not second_entered.is_set()
    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set()


@pytest.mark.anyio
async def test_snapshot_exports_to_distinct_destinations_run_together(tmp_path) -> None:
    coordinator = OperationCoordinator()
    spec = ENGINE_OPERATION_CATALOG[EngineOperation.EXPORT_SNAPSHOT]
    gate = asyncio.Event()
    both_entered = asyncio.Event()
    entered = 0

    async def export(name: str) -> None:
        nonlocal entered
        async with coordinator.coordinate(
            spec=spec,
            context=ContextIdentity("context-a"),
            request=ExportSnapshotRequest(destination=str(tmp_path / name)),
        ):
            entered += 1
            if entered == 2:
                both_entered.set()
            await gate.wait()

    first = asyncio.create_task(export("first.json"))
    second = asyncio.create_task(export("second.json"))
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    gate.set()
    await asyncio.gather(first, second)


@pytest.mark.anyio
async def test_cancelling_queued_writer_wakes_waiting_reader() -> None:
    coordinator = OperationCoordinator()
    first_read_entered = asyncio.Event()
    release_first_read = asyncio.Event()
    second_read_entered = asyncio.Event()

    async def first_read() -> None:
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.SEARCH],
            context=ContextIdentity("context-a"),
            request=SearchRequest(),
        ):
            first_read_entered.set()
            await release_first_read.wait()

    async def write() -> None:
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.RECORD],
            context=ContextIdentity("context-a"),
            request=RecordRequest(),
        ):
            raise AssertionError("cancelled writer must not enter")

    async def second_read() -> None:
        async with coordinator.coordinate(
            spec=ENGINE_OPERATION_CATALOG[EngineOperation.SEARCH],
            context=ContextIdentity("context-a"),
            request=SearchRequest(),
        ):
            second_read_entered.set()

    first_task = asyncio.create_task(first_read())
    await first_read_entered.wait()
    writer_task = asyncio.create_task(write())
    await asyncio.sleep(0)
    second_task = asyncio.create_task(second_read())
    await asyncio.sleep(0)
    assert not second_read_entered.is_set()

    writer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer_task
    await asyncio.wait_for(second_read_entered.wait(), timeout=1)

    release_first_read.set()
    await asyncio.gather(first_task, second_task)
