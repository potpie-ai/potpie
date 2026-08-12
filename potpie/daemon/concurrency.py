"""How many RPC calls the daemon lets run at once.

The daemon used to hold one ``asyncio.Lock`` across every ``/rpc`` and ``/attr``
call, so the surface was strictly one call at a time. That is more exclusion
than anything needs, and the cost is not theoretical: the shipped default host
*is* the daemon, so a single cold ``resolve`` — 16 seconds while the embedder
loads — froze every other terminal on the machine, including ``potpie pot
list``. A memory product that stops answering because it is busy answering is a
memory product agents learn to stop calling.

What the exclusion was actually buying, read off the stores rather than
assumed:

* the local JSON stores (pots, plans, inbox, repo defaults) are read-modify-write
  — two concurrent writers lose an update;
* they *publish* with ``tmp.replace(path)``, an atomic rename, so a reader
  concurrent with a writer sees one whole version or the other, never a torn
  one;
* the graph backends are already driven from threads (``asyncio.to_thread`` in
  the writer, ``run_in_threadpool`` here) and the resource-index drain thread
  reads and writes alongside RPC calls today, so concurrency inside a single
  backend is the status quo this did not create.

So the exclusion writes need is *writer against everything*, not *everything
against everything*, and this is the ordinary reader/writer lock that says so.
Readers share; a writer runs alone.

Writers take priority: once one is waiting, new readers queue behind it. Under
an agent workload — which is overwhelmingly reads — a reader-preferring lock
would let a ``record`` wait out an unbounded stream of ``resolve`` calls, and
"the write eventually lands" is not a property to leave to luck.

Which calls count as reads is declared in :mod:`potpie.daemon.surfaces`, where
the rest of the RPC contract lives, and defaults to *not* a read.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class RpcAccessLock:
    """A writer-preferring reader/writer lock over the daemon's RPC surface.

    One instance per daemon process. Not reentrant: nothing on the RPC path
    re-enters it, and making it so would hide the case where something started
    to.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writing = False
        self._writers_waiting = 0

    @property
    def readers(self) -> int:
        """How many shared holders are inside. For tests and diagnostics."""
        return self._readers

    @property
    def writing(self) -> bool:
        """Whether an exclusive holder is inside. For tests and diagnostics."""
        return self._writing

    @asynccontextmanager
    async def shared(self) -> AsyncIterator[None]:
        """Run alongside other readers; never alongside a writer."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writing and self._writers_waiting == 0
            )
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Run alone: no other writer, and no reader in flight."""
        async with self._condition:
            self._writers_waiting += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writing and self._readers == 0
                )
            except BaseException:
                # A client that hangs up mid-wait must not leave readers
                # deferring to a writer that no longer exists. The holders
                # inside would notify on their way out, but only this path
                # knows the deferral is over the instant it happens.
                self._writers_waiting -= 1
                self._condition.notify_all()
                raise
            self._writers_waiting -= 1
            self._writing = True
        try:
            yield
        finally:
            async with self._condition:
                self._writing = False
                self._condition.notify_all()

    @asynccontextmanager
    async def for_call(self, *, read_only: bool) -> AsyncIterator[None]:
        """The lock this call needs, chosen by the surfaces declaration."""
        if read_only:
            async with self.shared():
                yield
        else:
            async with self.exclusive():
                yield


__all__ = ["RpcAccessLock"]
