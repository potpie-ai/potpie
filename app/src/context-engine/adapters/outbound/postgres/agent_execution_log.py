"""Postgres implementation of ``AgentExecutionLogPort``.

The single durable substrate for a batch's agent run: an append-only log
that is both the live stream and the crash-resume checkpoint.

Two write surfaces, both invoked from inside the agent's asyncio loop
(stream handler) and at tool boundaries (checkpoint). Like the checkpoint
store, every write opens a **fresh** short-lived session against the same
engine — the batch-scoped session is busy with the agent's tool calls and
a shared connection would trip ``"concurrent operations are not
permitted"``.

The tail (HTTP ``StreamingResponse``, runs in a threadpool) replays rows
after a seq cursor then live-tails. Liveness uses Postgres
``LISTEN/NOTIFY`` to wake early, with a hard poll fallback so correctness
never depends on a notification being delivered.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import (
    ContextAgentCheckpointModel,
    ContextAgentExecutionLogModel,
)
from domain.ports.agent_execution_log import (
    ExecutionRecordType,
    ResumeState,
    TERMINAL_RECORD_TYPES,
)

logger = logging.getLogger(__name__)

# Single shared NOTIFY channel; the batch_id rides in the payload so we
# never hit the 63-char identifier limit and only LISTEN once. Writes emit
# this NOTIFY today so a future LISTEN-based adapter is a drop-in swap; the
# current Postgres adapter tails by polling, which is the correctness floor.
_NOTIFY_CHANNEL = "context_agent_exec_log"
# Poll cadence. Sub-second so the stream feels live; the agent emits at
# token/tool granularity so this is the dominant UI latency term.
_POLL_INTERVAL_SECONDS = 0.4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_stream_event(row: ContextAgentExecutionLogModel) -> dict[str, Any]:
    """Render a log row into the NDJSON stream contract.

    ``stream_id`` is the (stringified) seq so the existing frontend cursor /
    dedupe logic keeps working unchanged. ``part_id`` + ``done`` are passed
    through so the client can grow coalesced model parts in place.
    """
    payload = row.payload if isinstance(row.payload, dict) else {}
    rtype = row.record_type
    base: dict[str, Any] = {
        "stream_id": str(row.seq),
        "seq": row.seq,
        "event_id": row.event_id,
        "part_id": row.part_id,
        "done": bool(row.done),
        "created_at": (
            row.created_at.isoformat()
            if isinstance(row.created_at, datetime)
            else None
        ),
    }
    if rtype in ("run_finished", "run_failed"):
        return {
            **base,
            "type": "end",
            "status": payload.get("status")
            or ("done" if rtype == "run_finished" else "failed"),
            "error": payload.get("error"),
            "message": payload.get("summary") or payload.get("error"),
        }
    if rtype == "status":
        return {
            **base,
            "type": "status",
            "status": payload.get("status"),
            "stage": payload.get("stage"),
            "message": payload.get("message"),
            "metadata": payload.get("metadata"),
        }
    # Everything else is timeline activity. ``kind`` is the record type so
    # the client can style text / thinking / tool_call / tool_result /
    # mutation_applied distinctly.
    return {
        **base,
        "type": "activity",
        "kind": rtype,
        "sequence": row.seq,
        "title": payload.get("title"),
        "body": payload.get("content") or payload.get("body"),
        "payload": payload,
    }


class PostgresAgentExecutionLog:
    """``AgentExecutionLogPort`` backed by ``context_agent_execution_log``."""

    def __init__(self, session: Session) -> None:
        # Template only — every operation opens its own short session so it
        # can't collide with the agent's in-flight tool calls on the
        # batch-scoped session (see module docstring).
        self._template = session

    def _open(self) -> Session:
        bind = self._template.get_bind()
        return Session(bind=bind, autoflush=False, autocommit=False)

    # ----- write side -----

    def _notify(self, db: Session, batch_id: str) -> None:
        try:
            db.execute(
                text("SELECT pg_notify(:chan, :payload)"),
                {"chan": _NOTIFY_CHANNEL, "payload": batch_id},
            )
        except Exception:  # noqa: BLE001 - liveness must not fail the write
            logger.debug("pg_notify failed for batch %s", batch_id, exc_info=True)

    def append(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: ExecutionRecordType,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> None:
        stmt = pg_insert(ContextAgentExecutionLogModel).values(
            batch_id=batch_id,
            seq=int(seq),
            record_type=str(record_type),
            part_id=None,
            done=True,
            event_id=event_id,
            payload=dict(payload or {}),
        )
        # A resumed run that re-issues an already-streamed seq is a no-op,
        # not a duplicate.
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_context_agent_execution_log_batch_seq"
        )
        with self._open() as db:
            db.execute(stmt)
            self._notify(db, batch_id)
            db.commit()

    def upsert_part(
        self,
        *,
        batch_id: str,
        seq: int,
        record_type: str,
        part_id: str,
        content: str,
        done: bool,
        event_id: str | None = None,
    ) -> None:
        payload = {"content": content}
        stmt = pg_insert(ContextAgentExecutionLogModel).values(
            batch_id=batch_id,
            seq=int(seq),
            record_type=str(record_type),
            part_id=part_id,
            done=bool(done),
            event_id=event_id,
            payload=payload,
        )
        # Grow the existing part row in place and bump its seq to the latest
        # so a seq-cursor tail re-delivers it without extra plumbing.
        stmt = stmt.on_conflict_do_update(
            constraint="uq_context_agent_execution_log_batch_part",
            set_={
                "seq": stmt.excluded.seq,
                "record_type": stmt.excluded.record_type,
                "payload": stmt.excluded.payload,
                "done": stmt.excluded.done,
                "event_id": stmt.excluded.event_id,
            },
        )
        with self._open() as db:
            db.execute(stmt)
            self._notify(db, batch_id)
            db.commit()

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
        now = _utcnow()
        stmt = pg_insert(ContextAgentCheckpointModel).values(
            batch_id=batch_id,
            messages_json=list(messages_json),
            tool_call_count=int(tool_call_count),
            completed_event_ids=list(completed_event_ids or []),
            last_seq=int(last_seq),
            chunk_index=int(chunk_index),
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContextAgentCheckpointModel.batch_id],
            set_={
                "messages_json": stmt.excluded.messages_json,
                "tool_call_count": stmt.excluded.tool_call_count,
                "completed_event_ids": stmt.excluded.completed_event_ids,
                "last_seq": stmt.excluded.last_seq,
                "chunk_index": stmt.excluded.chunk_index,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        with self._open() as db:
            db.execute(stmt)
            db.commit()

    def load_resume_state(self, batch_id: str) -> ResumeState | None:
        with self._open() as db:
            row = db.scalar(
                select(ContextAgentCheckpointModel).where(
                    ContextAgentCheckpointModel.batch_id == batch_id
                )
            )
            if row is None:
                return None
            messages = row.messages_json
            if not isinstance(messages, list):
                messages = []
            completed = row.completed_event_ids
            if not isinstance(completed, list):
                completed = []
            return ResumeState(
                batch_id=row.batch_id,
                messages_json=list(messages),
                tool_call_count=row.tool_call_count or 0,
                completed_event_ids=[str(e) for e in completed],
                last_seq=row.last_seq or 0,
                chunk_index=row.chunk_index or 0,
            )

    def clear(self, batch_id: str) -> None:
        # Drop only the resume checkpoint; the append-only log rows are kept
        # for post-run history / replay.
        with self._open() as db:
            db.execute(
                delete(ContextAgentCheckpointModel).where(
                    ContextAgentCheckpointModel.batch_id == batch_id
                )
            )
            db.commit()

    # ----- subscribe side -----

    def _fetch_after(
        self, db: Session, batch_id: str, after_seq: int
    ) -> list[ContextAgentExecutionLogModel]:
        return list(
            db.scalars(
                select(ContextAgentExecutionLogModel)
                .where(
                    ContextAgentExecutionLogModel.batch_id == batch_id,
                    ContextAgentExecutionLogModel.seq > after_seq,
                )
                .order_by(ContextAgentExecutionLogModel.seq.asc())
            )
        )

    def replay_and_tail(
        self,
        *,
        batch_id: str,
        cursor_seq: int | None = None,
        idle_timeout_seconds: float = 120.0,
    ) -> Iterator[dict[str, Any]]:
        last_seq = int(cursor_seq or 0)
        terminal_seen = False

        # 1) Replay everything after the cursor.
        try:
            with self._open() as db:
                rows = self._fetch_after(db, batch_id, last_seq)
        except Exception as exc:  # noqa: BLE001
            logger.error("execution-log replay failed for %s: %s", batch_id, exc)
            yield {
                "type": "end",
                "status": "error",
                "message": f"replay failed: {exc}",
                "stream_id": str(last_seq),
            }
            return

        for row in rows:
            last_seq = max(last_seq, row.seq)
            yield _to_stream_event(row)
            if row.record_type in TERMINAL_RECORD_TYPES:
                terminal_seen = True
        if terminal_seen:
            return

        # 2) Tail until a terminal record or the idle window elapses.
        deadline = time.monotonic() + idle_timeout_seconds
        while True:
            time.sleep(_POLL_INTERVAL_SECONDS)
            try:
                with self._open() as db:
                    rows = self._fetch_after(db, batch_id, last_seq)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "execution-log tail failed for %s: %s", batch_id, exc
                )
                yield {
                    "type": "end",
                    "status": "error",
                    "message": f"tail failed: {exc}",
                    "stream_id": str(last_seq),
                }
                return

            if not rows:
                if time.monotonic() >= deadline:
                    yield {
                        "type": "end",
                        "status": "idle_timeout",
                        "message": "No activity within idle window.",
                        "stream_id": str(last_seq),
                    }
                    return
                continue

            deadline = time.monotonic() + idle_timeout_seconds
            for row in rows:
                last_seq = max(last_seq, row.seq)
                yield _to_stream_event(row)
                if row.record_type in TERMINAL_RECORD_TYPES:
                    return


__all__ = ["PostgresAgentExecutionLog", "_to_stream_event"]
