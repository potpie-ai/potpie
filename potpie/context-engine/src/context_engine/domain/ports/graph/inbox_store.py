"""Graph inbox persistence port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from context_engine.domain.graph_inbox import GraphInboxItem


class GraphInboxStorePort(Protocol):
    """Persist pending graph work that is not yet a canonical fact."""

    def save(self, item: GraphInboxItem) -> None:
        """Insert or replace an inbox item."""
        ...

    def get(self, *, pot_id: str, item_id: str) -> GraphInboxItem | None:
        """Return one inbox item for a pot, if present."""
        ...

    def list(
        self,
        *,
        pot_id: str,
        status: tuple[str, ...] = (),
        claimed_by: str | None = None,
        suspected_subgraph: str | None = None,
        source_ref: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[GraphInboxItem, ...]:
        """Return inbox items for worklists and operator inspection."""
        ...


__all__ = ["GraphInboxStorePort"]
