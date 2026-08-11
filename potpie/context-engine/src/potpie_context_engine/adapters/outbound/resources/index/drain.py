"""The single writer thread that fills in pending embeddings.

Embedding is the only expensive operation in the system (~145 texts/s against
0.68s to lexically index 10k chunks), so a design that embedded synchronously
would turn ``resource import`` from seconds into minutes. This is the whole
background mechanism, and it is deliberately the smallest thing that works.

**Why a thread and not a task.** The obvious alternative is
``asyncio.create_task`` in the RPC handler. The daemon hard-kills 10s after
SIGTERM with no drain hook, so an in-loop task is lost mid-batch — and worse,
it contends with every subsequent RPC for the process-wide lock while it runs.
A thread doing SQLite writes under WAL touches neither. There is precedent in
the tree: the telemetry flush thread does exactly this.

**Why no queue.** The thread carries no worklist. It wakes, calls
:meth:`ResourceIndexPort.drain`, and goes back to sleep — "what is pending" is
``embedding IS NULL`` in the database. That is what makes this crash-safe with
no delivery guarantee: a killed process leaves the rows, and the next drain (or
the next boot) resumes. The event is a latency optimization, not a channel; if
every signal were lost the loop's idle tick would still finish the work.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from potpie_context_core.ports.resource_index import DEFAULT_DRAIN_BUDGET

logger = logging.getLogger(__name__)

#: How long the loop sleeps when nothing signalled it. Long enough to be free
#: when idle, short enough that work signalled by a process that died before it
#: could notify still lands within a minute.
IDLE_INTERVAL_SECONDS = 30.0

#: Pause after a failed pass, so a persistently broken embedder cannot spin.
ERROR_BACKOFF_SECONDS = 5.0


@dataclass(slots=True)
class ResourceIndexDrain:
    """Owns one background thread draining one index.

    Start it once per process and call :meth:`signal` after every write. Both
    are idempotent: starting a running drain is a no-op, and signalling one
    that is mid-pass just guarantees another pass follows.
    """

    index: Any
    budget: int = DEFAULT_DRAIN_BUDGET
    idle_interval: float = IDLE_INTERVAL_SECONDS
    name: str = "potpie-resource-index-drain"

    _wake: threading.Event = field(default_factory=threading.Event, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> bool:
        """Start the thread unless the profile has nothing to drain.

        Refusing to start for a lexical-only profile is not an optimization: a
        thread that wakes every 30s forever to discover it has no work is a
        process that never looks idle, on a CLI whose whole local story is that
        it costs nothing when unused.
        """
        try:
            if not self.index.capabilities().semantic:
                return False
        except Exception as exc:  # noqa: BLE001 - a broken index must not block boot
            logger.debug("resource index drain not started: %s", exc)
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            # Daemon thread: the drain must never hold a process open. Losing a
            # batch at exit costs nothing — the rows stay pending and the next
            # start resumes them.
            self._thread = threading.Thread(
                target=self._run, name=self.name, daemon=True
            )
            self._thread.start()
        # Anything imported before the thread existed is already pending, so
        # the first pass has to happen without waiting for the idle tick.
        self._wake.set()
        return True

    def signal(self) -> None:
        """Tell the loop there is new work. Safe to call from any thread."""
        self._wake.set()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Ask the loop to finish its current pass and exit.

        Bounded because callers are shutdown paths: the daemon has ~10s in
        total and the service has a lifespan to leave. An unjoined thread is
        not a leak here — it is a daemon thread whose only state is a SQLite
        transaction that either committed or rolled back."""
        self._stop.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # --- the loop -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            # Cleared *before* the pass, never after: a write that lands while
            # the pass is running must leave the event set, or its rows wait a
            # full idle interval for no reason.
            self._wake.clear()
            wait = self.idle_interval
            try:
                report = self.index.drain(budget=self.budget)
                if report.embedded:
                    logger.debug(
                        "resource index drain embedded %s window(s), %s remaining",
                        report.embedded,
                        report.remaining,
                    )
                if report.remaining:
                    # More than one budget's worth is outstanding: go straight
                    # round again rather than sleeping on a known backlog.
                    continue
            except Exception as exc:  # noqa: BLE001 - the loop outlives any one failure
                logger.warning("resource index drain pass failed: %s", exc)
                wait = ERROR_BACKOFF_SECONDS
            self._wake.wait(timeout=wait)


__all__ = ["ERROR_BACKOFF_SECONDS", "IDLE_INTERVAL_SECONDS", "ResourceIndexDrain"]
