"""Where the explorer reads a graph from: this machine, or a managed host.

The local daemon serves the explorer for both origins. The browser only ever
talks to loopback, so a managed host's key stays in the daemon (read from the
host registry) and never reaches a page, a URL or browser history.

``LocalExplorerSource``
    The daemon's own services, called directly, as the explorer always has.
    The whole-graph view goes through the typed ``graph_slice`` operation on
    the daemon's own resource manager.
``ManagedExplorerSource``
    The managed host's typed operations: ``host.*`` for pots (one
    ``host.pot_list`` with per-pot counts, no budget) and the engine catalog
    for graph reads, through ``potpie.runtime.remote``.

Every read takes the pot id the router already resolved on the same source,
so an id from one graph is never read against the other.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from potpie_context_engine.core.ports.claim_query import ClaimQueryFilter
from potpie_context_engine.core.ports.graph_service import (
    GraphCatalogRequest,
    GraphEntitySearchRequest,
    GraphReadRequest,
)
from potpie_context_engine.requests import (
    CatalogRequest,
    DataPlaneStatusRequest,
    GraphSliceRequest,
    NeighborhoodRequest,
    ReadRequest,
    SearchEntitiesRequest,
)

#: Builds the typed engine client for one resolved pot id.
EngineClientFactory = Callable[[str], Any]


class ExplorerSourceError(Exception):
    """A typed refusal from a source; ``error`` is the boundary error."""

    def __init__(self, error: object) -> None:
        super().__init__(str(getattr(error, "message", "explorer read failed")))
        self.error = error


class ExplorerSource(Protocol):
    """The reads and the one pointer move the explorer makes, for one origin."""

    origin: str

    @property
    def pots(self) -> Any:
        """The origin's :class:`~potpie.pots.directory.PotDirectory` (sync)."""
        ...

    async def pot_counts(self, pots: list[Any]) -> dict[str, dict[str, Any]]:
        """``{pot_id: {"source_count", "counts"}}`` for the pots it can count."""
        ...

    def engine_client(self, pot_id: str) -> Any | None: ...

    async def catalog(self, pot_id: str) -> Any: ...

    async def data_plane_status(self, pot_id: str) -> Any: ...

    async def search_entities(self, pot_id: str, fields: Mapping[str, Any]) -> Any: ...

    async def graph_slice(self, pot_id: str, *, include_invalidated: bool) -> Any: ...

    async def neighborhood(self, pot_id: str, *, key: str, depth: int) -> Any: ...

    async def read(self, pot_id: str, fields: Mapping[str, Any]) -> Any: ...


def _value(outcome: Any) -> Any:
    if getattr(outcome, "ok", False):
        return outcome.value
    raise ExplorerSourceError(outcome.error)


class LocalExplorerSource:
    """This machine's graph, through the daemon's own services."""

    origin = "local"

    def __init__(
        self,
        *,
        pots: Any,
        graph: Any,
        backend: Any,
        engine_client: EngineClientFactory | None = None,
    ) -> None:
        from potpie.pots.directory import LocalPotDirectory

        self._pots = LocalPotDirectory(pots)
        self._graph = graph
        self._backend = backend
        self._engine_client = engine_client

    @property
    def pots(self) -> Any:
        return self._pots

    def engine_client(self, pot_id: str) -> Any | None:
        return self._engine_client(pot_id) if self._engine_client else None

    async def pot_counts(self, pots: list[Any]) -> dict[str, dict[str, Any]]:
        def count() -> dict[str, dict[str, Any]]:
            return {
                pot.pot_id: {
                    "source_count": _source_count(self._pots, pot.pot_id),
                    "counts": _counts(self._graph, pot.pot_id),
                }
                for pot in pots
            }

        return await asyncio.to_thread(count)

    async def catalog(self, pot_id: str) -> Any:
        return await asyncio.to_thread(
            self._graph.catalog, GraphCatalogRequest(pot_id=pot_id)
        )

    async def data_plane_status(self, pot_id: str) -> Any:
        return await asyncio.to_thread(self._graph.data_plane_status, pot_id)

    async def search_entities(self, pot_id: str, fields: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(
            self._graph.search_entities,
            GraphEntitySearchRequest(pot_id=pot_id, **fields),
        )

    async def graph_slice(self, pot_id: str, *, include_invalidated: bool) -> Any:
        client = self.engine_client(pot_id)
        if client is not None:
            return _value(
                await client.graph_slice(
                    GraphSliceRequest(include_invalidated=include_invalidated)
                )
            )
        # An explorer assembled without the typed client (tests, embedding)
        # reads the backend directly, as the operation itself does.
        return await asyncio.to_thread(
            lambda: self._backend.inspection.slice(
                pot_id=pot_id,
                filter_=ClaimQueryFilter(
                    pot_id=pot_id, include_invalidated=include_invalidated
                ),
            )
        )

    async def neighborhood(self, pot_id: str, *, key: str, depth: int) -> Any:
        return await asyncio.to_thread(
            lambda: self._backend.inspection.neighborhood(
                pot_id=pot_id, entity_key=key, depth=depth
            )
        )

    async def read(self, pot_id: str, fields: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(
            self._graph.read, GraphReadRequest(pot_id=pot_id, **fields)
        )


class ManagedExplorerSource:
    """A managed host's graph, through its typed operations.

    Built per request from the registry (``connect()`` reuses the pooled
    connection), with a fresh pot directory, so a pointer another client moved
    since the last request is never answered from a stale listing.
    """

    origin = "managed"

    def __init__(self, connection: Any) -> None:
        from potpie.runtime.remote import ManagedPotDirectory

        self._connection = connection
        self._pots = ManagedPotDirectory(connection.pots.client)

    @classmethod
    def from_registry(cls) -> ManagedExplorerSource:
        """The registry's managed host; raises when none is usable."""
        from potpie.config import host_registry
        from potpie.runtime import remote

        try:
            return cls(remote.connect(host_registry.MANAGED))
        except remote.HostNotConfigured:
            raise host_registry.managed_unconfigured_error() from None

    @property
    def pots(self) -> Any:
        return self._pots

    def engine_client(self, pot_id: str) -> Any:
        from potpie.runtime import ContextSelector

        return self._connection.engine(ContextSelector(kind="explicit", value=pot_id))

    async def pot_counts(self, pots: list[Any]) -> dict[str, dict[str, Any]]:
        del pots  # the host counts every pot it lists, in the same answer
        listing = await asyncio.to_thread(
            lambda: self._pots.listing(include_status=True)
        )
        return {
            pot_id: {"source_count": counts.source_count, "counts": dict(counts.counts)}
            for pot_id, counts in dict(listing.status).items()
        }

    async def catalog(self, pot_id: str) -> Any:
        return _value(await self.engine_client(pot_id).catalog(CatalogRequest()))

    async def data_plane_status(self, pot_id: str) -> Any:
        return _value(
            await self.engine_client(pot_id).data_plane_status(DataPlaneStatusRequest())
        )

    async def search_entities(self, pot_id: str, fields: Mapping[str, Any]) -> Any:
        return _value(
            await self.engine_client(pot_id).search_entities(
                SearchEntitiesRequest(**fields)
            )
        )

    async def graph_slice(self, pot_id: str, *, include_invalidated: bool) -> Any:
        return _value(
            await self.engine_client(pot_id).graph_slice(
                GraphSliceRequest(include_invalidated=include_invalidated)
            )
        )

    async def neighborhood(self, pot_id: str, *, key: str, depth: int) -> Any:
        return _value(
            await self.engine_client(pot_id).neighborhood(
                NeighborhoodRequest(entity_key=key, depth=depth)
            )
        )

    async def read(self, pot_id: str, fields: Mapping[str, Any]) -> Any:
        return _value(await self.engine_client(pot_id).read(ReadRequest(**fields)))


def _counts(graph: Any, pot_id: str) -> dict[str, int]:
    try:
        dp = graph.data_plane_status(pot_id)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, int] = {}
    for key, value in dict(getattr(dp, "counts", {}) or {}).items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _source_count(pots: Any, pot_id: str) -> int:
    try:
        return len(pots.list_sources(pot_id=pot_id))
    except Exception:  # noqa: BLE001
        return 0


__all__ = [
    "EngineClientFactory",
    "ExplorerSource",
    "ExplorerSourceError",
    "LocalExplorerSource",
    "ManagedExplorerSource",
]
