"""Postgres implementation of ``AgentCheckpointStorePort``."""

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
        self._db = session

    def load(self, batch_id: str) -> AgentCheckpoint | None:
        row = self._db.scalar(
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
        self._db.execute(stmt)
        self._db.commit()

    def delete(self, batch_id: str) -> None:
        self._db.execute(
            delete(ContextAgentCheckpointModel).where(
                ContextAgentCheckpointModel.batch_id == batch_id
            )
        )
        self._db.commit()
