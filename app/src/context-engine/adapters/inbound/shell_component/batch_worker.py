"""In-process background worker for context-graph batch processing.

Generic loop: each tick runs flush(windowed) -> claim+process one batch -> reap(stale).
The durable queue is the relational batch table, so no external broker is needed. The
three steps are injected as async callables, so this worker is decoupled from the concrete
reconciliation use-cases (which require a built container + DB session). The component
starts this once that wiring lands.
"""
from __future__ import annotations
import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger("context_graph.batch_worker")


class BatchWorker:
    def __init__(
        self,
        *,
        flush_windowed: Callable[[], Awaitable[None]],
        claim_one: Callable[[], Awaitable[str | None]],
        reap_stale: Callable[[], Awaitable[None]],
        poll_interval_s: float = 1.0,
    ) -> None:
        self._flush = flush_windowed
        self._claim = claim_one
        self._reap = reap_stale
        self._poll = poll_interval_s
        self._stop = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._stop.is_set():
            claimed: str | None = None
            try:
                await self._flush()
                claimed = await self._claim()
                await self._reap()
            except Exception:  # noqa: BLE001 — a transient step failure must not kill the loop
                logger.warning("batch worker tick failed", exc_info=True)
            if not claimed:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._poll)
                except asyncio.TimeoutError:
                    pass

    async def stop(self) -> None:
        self._stop.set()
