"""Postgres implementation of ``AgentCheckpointStorePort``.

The checkpoint writer is invoked by pydantic-ai's ``CheckpointMiddleware``
mid-agent-run, and the agent concurrently issues tool calls that hit
Postgres through other ports bound to the *same* batch-scoped session.
That race produces ``InvalidRequestError("This session is provisioning a
new connection; concurrent operations are not permitted")`` because two
coroutines try to use one connection at once.

The fix here is to make checkpoint writes use a **fresh** SQLAlchemy
session per call (opened from the batch session's bind / engine). The
batch session keeps owning ledger + batch repo writes; the checkpoint
store has its own short-lived sessions that can't collide with whatever
the agent is doing on the main session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from adapters.outbound.postgres.models import ContextAgentCheckpointModel
from domain.ports.agent_checkpoint_store import AgentCheckpointStorePort
from domain.reconciliation_batch import AgentCheckpoint


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlAlchemyAgentCheckpointStore(AgentCheckpointStorePort):
    def __init__(self, session: Session) -> None:
        # Keep the original session around in case the host explicitly wants
        # checkpoint writes on the same connection (e.g. tests under a
        # transactional rollback fixture). Each call opens its own short
        # session against the same engine to dodge concurrency collisions
        # with mid-flight agent tool calls.
        self._template = session

    def _open(self) -> Session:
        bind = self._template.get_bind()
        return Session(bind=bind, autoflush=False, autocommit=False)

    def load(self, batch_id: str) -> AgentCheckpoint | None:
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
            return AgentCheckpoint(
                batch_id=row.batch_id,
                messages_json=list(messages),
                tool_call_count=row.tool_call_count or 0,
                updated_at=_ensure_aware(row.updated_at) if row.updated_at else _utcnow(),
            )

    def save(
        self,
        batch_id: str,
        *,
        messages_json: list[dict[str, Any]],
        tool_call_count: int,
    ) -> None:
        now = _utcnow()
        stmt = pg_insert(ContextAgentCheckpointModel).values(
            batch_id=batch_id,
            messages_json=list(messages_json),
            tool_call_count=int(tool_call_count),
            updated_at=now,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ContextAgentCheckpointModel.batch_id],
            set_={
                "messages_json": stmt.excluded.messages_json,
                "tool_call_count": stmt.excluded.tool_call_count,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        with self._open() as db:
            db.execute(stmt)
            db.commit()

    def delete(self, batch_id: str) -> None:
        with self._open() as db:
            db.execute(
                delete(ContextAgentCheckpointModel).where(
                    ContextAgentCheckpointModel.batch_id == batch_id
                )
            )
            db.commit()
