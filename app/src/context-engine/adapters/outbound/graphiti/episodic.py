"""Graphiti adapter implementing EpisodicGraphPort."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from datetime import datetime
from collections.abc import Coroutine
from typing import Any, Callable, Optional, TypeVar

from domain.entity_schema import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.settings import ContextEngineSettingsPort

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class GraphitiEpisodicAdapter(EpisodicGraphPort):
    """
    Thread-local Graphiti client so Celery sync code and FastAPI async code
    do not share a driver across different event loops.
    """

    def __init__(self, settings: ContextEngineSettingsPort) -> None:
        self._settings = settings
        self._enabled = settings.is_enabled()
        self._thread_local = threading.local()
        self._search_filters_cls = None
        self._init_error: Optional[str] = None

        if not self._enabled:
            return

        try:
            from graphiti_core.search.search_filters import SearchFilters

            self._search_filters_cls = SearchFilters
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("GraphitiEpisodicAdapter disabled due to init error: %s", exc)

    def _get_graphiti(self):
        if not self._enabled or self._init_error:
            return None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        client = getattr(self._thread_local, "graphiti", None)
        client_loop = getattr(self._thread_local, "graphiti_loop", None)
        if client is not None and client_loop is current_loop:
            return client
        if client is not None and client_loop is not current_loop:
            # Graphiti/Neo4j async internals are loop-bound; do not reuse
            # a client that was created under another event loop.
            self._thread_local.graphiti = None
            self._thread_local.graphiti_loop = None
        try:
            from graphiti_core import Graphiti

            uri = self._settings.neo4j_uri()
            user = self._settings.neo4j_user()
            password = self._settings.neo4j_password()
            if not uri or user is None or password is None:
                self._init_error = "missing_neo4j_credentials"
                return None

            client = Graphiti(uri=uri, user=user, password=password)
            self._thread_local.graphiti = client
            self._thread_local.graphiti_loop = current_loop
            return client
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("Graphiti init failed: %s", exc)
            return None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._init_error is None

    def failure_reason(self) -> Optional[str]:
        """Return why Graphiti cannot run, or None if the client can be used.

        Forces a lazy connection attempt when credentials were not yet validated.
        """
        if not self._settings.is_enabled():
            return "context_graph_disabled"
        if self._init_error:
            return self._init_error
        _ = self._get_graphiti()
        if self._init_error:
            return self._init_error
        return None

    async def _close_graphiti_for_running_loop(self) -> None:
        """Close driver before the ephemeral asyncio.run loop shuts down.

        If we skip this, Neo4j async transports may try to schedule cleanup on a
        loop that asyncio.run() has already closed → RuntimeError: Event loop is closed.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        client = getattr(self._thread_local, "graphiti", None)
        client_loop = getattr(self._thread_local, "graphiti_loop", None)
        if client is None or client_loop is not loop:
            return
        try:
            await client.close()
        except Exception as exc:
            logger.debug("Graphiti close after sync run: %s", exc)
        self._thread_local.graphiti = None
        self._thread_local.graphiti_loop = None

    def _sync_run(self, factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
        async def _wrapped() -> _T:
            try:
                return await factory()
            finally:
                await self._close_graphiti_for_running_loop()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_wrapped())

        def _worker():
            return asyncio.run(_wrapped())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_worker).result()

    async def add_episode_async(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        g = self._get_graphiti()
        if g is None:
            return None

        result = await g.add_episode(
            name=name,
            episode_body=episode_body,
            source_description=source_description,
            reference_time=reference_time,
            group_id=pot_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )
        episode = getattr(result, "episode", None)
        episode_uuid = getattr(episode, "uuid", None)
        return str(episode_uuid) if episode_uuid else None

    def add_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        if not self.enabled:
            return None

        async def _run():
            return await self.add_episode_async(
                pot_id=pot_id,
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=reference_time,
            )

        return self._sync_run(_run)

    def _search_filters(
        self,
        node_labels: Optional[list[str]],
        source_description: Optional[str],
    ) -> Any | None:
        if not self._search_filters_cls:
            return None
        kwargs: dict[str, Any] = {}
        if node_labels:
            kwargs["node_labels"] = node_labels
        if source_description and source_description.strip():
            from graphiti_core.search.search_filters import ComparisonOperator, PropertyFilter

            kwargs["property_filters"] = [
                PropertyFilter(
                    property_name="source_description",
                    property_value=source_description.strip(),
                    comparison_operator=ComparisonOperator.equals,
                )
            ]
        if not kwargs:
            return None
        return self._search_filters_cls(**kwargs)

    async def search_async(
        self,
        pot_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        repo_name: Optional[str] = None,
        source_description: Optional[str] = None,
    ) -> list[Any]:
        del repo_name  # optional future filter; Graphiti search is pot-scoped
        g = self._get_graphiti()
        if g is None:
            return []

        search_filter = self._search_filters(node_labels, source_description)

        return await g.search(
            query=query,
            group_ids=[pot_id],
            num_results=limit,
            search_filter=search_filter,
        )

    def search(
        self,
        pot_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        repo_name: Optional[str] = None,
        source_description: Optional[str] = None,
    ) -> list[Any]:
        if not self.enabled:
            return []

        async def _run():
            return await self.search_async(
                pot_id=pot_id,
                query=query,
                limit=limit,
                node_labels=node_labels,
                repo_name=repo_name,
                source_description=source_description,
            )

        return self._sync_run(_run)
