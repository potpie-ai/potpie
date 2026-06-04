"""Durable agent execution log (port).

This is the single source of truth for a reconciliation batch's agent run.
It replaces the old split of "Redis stream for liveness + Postgres checkpoint
for resume + post-hoc work_events for history" with one append-only,
durable log that is *both*:

- the **live stream** the events screen tails (text / thinking / tool calls /
  tool results / graph mutations, as they happen), and
- the **durable execution substrate**: the agent's serialized message
  history + completion bookkeeping is checkpointed into the same store after
  every tool call, so a worker crash mid-run resumes with full context
  instead of restarting.

Transport is deliberately an *adapter* concern. The default adapter
(``PostgresAgentExecutionLog``) tails via ``LISTEN/NOTIFY`` with a poll
fallback; a Redis-backed or poll-only adapter can be swapped in without
touching any caller — that is the whole point of keeping this a port.

Record model
------------
Discrete records (one row, written once) — ``append``:
  ``run_started``     run/chunk begins (payload: attempt, chunk_index, chunks_total, model)
  ``chunk_marker``    "chunk i of n" within a multi-chunk batch
  ``tool_call``       a tool was dispatched (payload: tool_name, tool_call_id, args)
  ``tool_result``     a tool returned (payload: tool_name, tool_call_id, content)
  ``mutation_applied``apply_graph_mutations succeeded (payload: event_id, counts)
  ``event_processed`` mark_event_processed was called (payload: event_id, summary)
  ``status``          coarse status mirror (queued/processing/done/failed)
  ``run_finished``    terminal success (payload: completed_event_ids, summary)
  ``run_failed``      terminal failure (payload: error)

Coalesced records (one row per model *part*, grown in place) —
``upsert_part``:
  ``text``            model response text, streamed token-by-token
  ``thinking``        model reasoning, streamed token-by-token

A coalesced part keeps a stable ``part_id`` and is re-emitted (with the
cumulative content + ``done`` flag) until it closes; subscribers replace by
``part_id``. Adapters bump the row's ``seq`` on each flush so cursor-based
tailing (``seq > cursor``) re-delivers the growing part without extra
plumbing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ExecutionRecordType = Literal[
    "run_started",
    "chunk_marker",
    "text",
    "thinking",
    "tool_call",
    "tool_result",
    "mutation_applied",
    "event_processed",
    "status",
    "run_finished",
    "run_failed",
]

# Record types that close a run — the tail iterator yields a terminal
# ``{"type": "end"}`` after one of these and stops.
TERMINAL_RECORD_TYPES: frozenset[str] = frozenset(
    {"run_finished", "run_failed"}
)


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Everything needed to continue a crashed batch run with full context.

    ``messages_json`` is the pydantic-ai message history (the model's working
    memory). ``completed_event_ids`` is durable bookkeeping so a resumed run
    does not redo ``mark_event_processed`` side effects or re-emit completed
    events. ``last_seq`` lets the resumed run continue the append-only log
    without colliding with already-streamed records.
    """

    batch_id: str
    messages_json: list[dict[str, Any]]
    tool_call_count: int
    completed_event_ids: list[str] = field(default_factory=list)
    last_seq: int = 0
    chunk_index: int = 0


class AgentExecutionLogPort(Protocol):
    """Append + checkpoint + tail one batch's durable agent execution log."""

    # ----- write side (agent loop / worker) -----

    def append(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: ExecutionRecordType,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> None:
        """Append one discrete record. Idempotent on ``(batch_id, seq)`` so a
        resumed run that re-issues a seq is a no-op rather than a duplicate."""
        ...

    def upsert_part(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: Literal["text", "thinking"],
        part_id: str,
        content: str,
        done: bool,
        event_id: str | None = None,
    ) -> None:
        """Insert-or-grow a coalesced model part keyed by ``part_id``.

        The first call inserts; subsequent calls update the same row's
        cumulative ``content`` and bump its ``seq`` to ``seq`` so tailing
        subscribers re-receive it. ``done=True`` finalizes the part.
        """
        ...

    def checkpoint(
        self,
        *,
        batch_id: str,
        messages_json: list[dict[str, Any]],
        tool_call_count: int,
        completed_event_ids: list[str],
        last_seq: int,
        chunk_index: int,
    ) -> None:
        """Persist durable resume state. Called after every tool call so a
        crash leaves at most one tool-call's worth of progress un-replayed."""
        ...

    def load_resume_state(self, batch_id: str) -> ResumeState | None:
        """Return the durable resume state for ``batch_id`` if a run is in
        flight / crashed, else ``None`` (fresh run)."""
        ...

    def clear(self, batch_id: str) -> None:
        """Drop the resume checkpoint after the batch terminates successfully.

        The append-only log rows are intentionally retained for history /
        replay; only the resume checkpoint is cleared."""
        ...

    # ----- subscribe side (HTTP stream handler, run in a threadpool) -----

    def replay_and_tail(
        self,
        *,
        batch_id: str,
        cursor_seq: int | None = None,
        idle_timeout_seconds: float = 120.0,
    ) -> Iterator[dict[str, Any]]:
        """Replay records after ``cursor_seq`` then tail live ones.

        Blocking generator. Yields dicts shaped for the NDJSON stream
        contract (always carries ``stream_id`` and ``type``). Terminates with
        a ``{"type": "end", ...}`` record after a terminal record or once
        ``idle_timeout_seconds`` elapses with no new records.
        """
        ...


class NoOpAgentExecutionLog:
    """Inert log. Used when the durable-log infra is unavailable.

    Implements the port shape with no side effects so callers never need a
    ``None`` check. Subscriptions yield a single ``end`` so HTTP handlers
    don't hang.
    """

    def append(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: ExecutionRecordType,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> None:
        del batch_id, seq, record_type, payload, event_id

    def upsert_part(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: Literal["text", "thinking"],
        part_id: str,
        content: str,
        done: bool,
        event_id: str | None = None,
    ) -> None:
        del batch_id, seq, record_type, part_id, content, done, event_id

    def checkpoint(
        self,
        *,
        batch_id: str,
        messages_json: list[dict[str, Any]],
        tool_call_count: int,
        completed_event_ids: list[str],
        last_seq: int,
        chunk_index: int,
    ) -> None:
        del batch_id, messages_json, tool_call_count
        del completed_event_ids, last_seq, chunk_index

    def load_resume_state(self, batch_id: str) -> ResumeState | None:
        del batch_id
        return None

    def clear(self, batch_id: str) -> None:
        del batch_id

    def replay_and_tail(
        self,
        *,
        batch_id: str,
        cursor_seq: int | None = None,
        idle_timeout_seconds: float = 120.0,
    ) -> Iterator[dict[str, Any]]:
        del batch_id, cursor_seq, idle_timeout_seconds
        yield {
            "type": "end",
            "status": "disabled",
            "message": "Durable agent streaming is disabled on this server.",
            "stream_id": "0",
        }


__all__ = [
    "AgentExecutionLogPort",
    "ExecutionRecordType",
    "NoOpAgentExecutionLog",
    "ResumeState",
    "TERMINAL_RECORD_TYPES",
]
