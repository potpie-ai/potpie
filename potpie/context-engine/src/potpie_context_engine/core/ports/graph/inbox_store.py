"""Graph inbox persistence port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from potpie_context_engine.core.graph_inbox import GraphInboxItem


@runtime_checkable
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


@runtime_checkable
class AsyncGraphInboxStorePort(Protocol):
    async def save_async(self, item: GraphInboxItem) -> None: ...

    async def get_async(
        self, *, pot_id: str, item_id: str
    ) -> GraphInboxItem | None: ...

    async def list_async(
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
    ) -> tuple[GraphInboxItem, ...]: ...


__all__ = ["AsyncGraphInboxStorePort", "GraphInboxStorePort"]
