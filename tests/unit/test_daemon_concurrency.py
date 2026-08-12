"""What the daemon lets run at once, and what it still refuses to.

The daemon held one exclusive lock across every ``/rpc`` and ``/attr`` call, so
a single cold ``resolve`` — sixteen seconds while the embedder loads — froze
every other terminal on the machine. The lock is now a reader/writer lock and
the classification lives in :mod:`potpie.daemon.surfaces`.

Two properties have to hold together, and a test for either alone would pass on
a broken implementation: reads must overlap (or nothing was fixed) and writes
must not (or a lost update was shipped).
"""

from __future__ import annotations

import asyncio

import pytest

from potpie.daemon.concurrency import RpcAccessLock
from potpie.daemon.surfaces import (
    READ_ONLY_RPC_MEMBERS,
    READ_ONLY_RPC_SURFACES,
    RPC_SURFACES,
    is_read_only,
)

pytestmark = pytest.mark.asyncio


async def _hold(lock: RpcAccessLock, *, read_only: bool, log: list[str], name: str):
    async with lock.for_call(read_only=read_only):
        log.append(f"enter:{name}")
        await asyncio.sleep(0.05)
        log.append(f"exit:{name}")


async def test_two_reads_run_at_the_same_time() -> None:
    """The whole point. Serialized, these interleave as enter/exit/enter/exit."""
    lock = RpcAccessLock()
    log: list[str] = []

    await asyncio.gather(
        _hold(lock, read_only=True, log=log, name="a"),
        _hold(lock, read_only=True, log=log, name="b"),
    )

    assert log[:2] == ["enter:a", "enter:b"]


async def test_a_slow_read_does_not_block_a_fast_one() -> None:
    """The reported symptom, in the shape it was reported: one long call must
    not stop a short one that arrives while it runs."""
    lock = RpcAccessLock()
    finished: list[str] = []

    async def _slow() -> None:
        async with lock.shared():
            await asyncio.sleep(0.3)
            finished.append("slow")

    async def _fast() -> None:
        await asyncio.sleep(0.02)
        async with lock.shared():
            finished.append("fast")

    await asyncio.gather(_slow(), _fast())

    assert finished == ["fast", "slow"]


async def test_two_writes_never_overlap() -> None:
    lock = RpcAccessLock()
    log: list[str] = []

    await asyncio.gather(
        _hold(lock, read_only=False, log=log, name="a"),
        _hold(lock, read_only=False, log=log, name="b"),
    )

    assert log in (
        ["enter:a", "exit:a", "enter:b", "exit:b"],
        ["enter:b", "exit:b", "enter:a", "exit:a"],
    )


async def test_a_write_excludes_a_read() -> None:
    """Read-modify-write on the JSON stores is what the exclusion is for."""
    lock = RpcAccessLock()
    log: list[str] = []

    async def _writer() -> None:
        async with lock.exclusive():
            log.append("write:enter")
            await asyncio.sleep(0.1)
            log.append("write:exit")

    async def _reader() -> None:
        await asyncio.sleep(0.02)
        async with lock.shared():
            log.append("read")

    await asyncio.gather(_writer(), _reader())

    assert log == ["write:enter", "write:exit", "read"]


async def test_a_waiting_write_is_not_starved_by_arriving_reads() -> None:
    """Agent traffic is overwhelmingly reads. A reader-preferring lock would let
    a ``record`` wait out an unbounded stream of ``resolve`` calls, and "the
    write lands eventually" is not a property to leave to arrival timing."""
    lock = RpcAccessLock()
    order: list[str] = []

    async def _reader(name: str, delay: float) -> None:
        await asyncio.sleep(delay)
        async with lock.shared():
            order.append(name)
            await asyncio.sleep(0.05)

    async def _writer() -> None:
        await asyncio.sleep(0.02)
        async with lock.exclusive():
            order.append("write")

    await asyncio.gather(
        _reader("read-before", 0.0),
        _writer(),
        _reader("read-after", 0.04),
    )

    assert order == ["read-before", "write", "read-after"]


async def test_a_cancelled_writer_does_not_wedge_the_readers() -> None:
    """A client that hangs up mid-wait must not leave every later read
    deferring to a writer that no longer exists."""
    lock = RpcAccessLock()

    async with lock.shared():
        waiting = asyncio.create_task(_hold(lock, read_only=False, log=[], name="w"))
        await asyncio.sleep(0.02)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

    async with asyncio.timeout(2):
        async with lock.shared():
            pass

    assert lock.readers == 0 and lock.writing is False


# --- the classification -----------------------------------------------------


async def test_unknown_targets_are_exclusive() -> None:
    """The fail-safe direction. A method nobody classified costs throughput,
    never correctness — which is why this is a declaration and not a rule about
    how method names are spelled."""
    assert is_read_only("pots", "some_new_method") is False
    assert is_read_only("nonexistent_surface", "read") is False


async def test_the_known_writes_are_not_declared_read_only() -> None:
    for surface, method in (
        ("agent_context", "record"),
        ("graph", "mutate"),
        ("graph", "record"),
        ("graph_workbench", "commit"),
        ("graph_workbench", "propose"),
        ("pots", "create_pot"),
        ("pots", "archive_pot"),
        ("pots", "use_pot"),
        ("pots", "add_source"),
        ("config", "set"),
        ("resources", "import_dir"),
        ("resources", "delete"),
        ("setup", "run"),
        ("skills", "install"),
        # Reads the ledger *and* advances the consumer cursor.
        ("ledger", "pull"),
    ):
        assert is_read_only(surface, method) is False, f"{surface}.{method}"


async def test_the_hot_reads_are_declared() -> None:
    for surface, method in (
        ("agent_context", "resolve"),
        ("agent_context", "search"),
        ("agent_context", "status"),
        ("graph", "read"),
        ("graph", "search_entities"),
        ("pots", "list_pots"),
        ("pots", "aggregate_status"),
        ("backend.claim_query", "find_claims"),
        ("backend.analytics", "anything_at_all"),
    ):
        assert is_read_only(surface, method) is True, f"{surface}.{method}"


async def test_every_declared_read_target_is_a_served_surface() -> None:
    """A typo in the declaration would silently mean "always exclusive", which
    no other test can see."""
    declared = set(READ_ONLY_RPC_SURFACES) | set(READ_ONLY_RPC_MEMBERS)

    assert declared <= set(RPC_SURFACES)
