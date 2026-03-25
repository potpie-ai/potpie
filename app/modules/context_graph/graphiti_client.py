"""Thin wrapper around Graphiti with project-level namespacing."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from datetime import datetime
from typing import Any, Callable, Optional

from app.core.config_provider import config_provider
from app.modules.context_graph.entity_schema import (
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ContextGraphClient:
    """
    Graphiti uses an async Neo4j driver that must stay on one event loop per
    driver instance. Mixing ``asyncio.run()`` (new loop) with an already-running
    loop (FastAPI / Pydantic AI) causes "Future attached to a different loop".

    We keep a **thread-local** Graphiti instance so:
    - Celery / sync code can ``asyncio.run()`` in its worker thread.
    - Async agents can ``await`` on the main loop without nesting ``asyncio.run()``.
    """

    def __init__(self):
        self._config = config_provider.get_context_graph_config()
        self._enabled = bool(self._config.get("enabled"))
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
            logger.warning("ContextGraphClient disabled due to init error: %s", exc)

    def _get_graphiti(self):
        """Return a Graphiti client bound to the current thread (lazy)."""
        if not self._enabled or self._init_error:
            return None
        client = getattr(self._thread_local, "graphiti", None)
        if client is not None:
            return client
        try:
            from graphiti_core import Graphiti

            client = Graphiti(
                uri=self._config.get("neo4j_uri"),
                user=self._config.get("neo4j_user"),
                password=self._config.get("neo4j_password"),
            )
            self._thread_local.graphiti = client
            return client
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("ContextGraphClient Graphiti init failed: %s", exc)
            return None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._init_error is None

    def _sync_run(self, factory: Callable[[], Any]) -> Any:
        """Run async *factory* (``lambda: coro``) from sync code without nesting loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(factory())

        # Already inside an event loop: run in a worker thread with its own loop + thread-local Graphiti.
        def _worker():
            return asyncio.run(factory())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_worker).result()

    async def add_episode_async(
        self,
        project_id: str,
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
            group_id=project_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )
        episode = getattr(result, "episode", None)
        episode_uuid = getattr(episode, "uuid", None)
        return str(episode_uuid) if episode_uuid else None

    def add_episode(
        self,
        project_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        if not self.enabled:
            return None

        async def _run():
            return await self.add_episode_async(
                project_id=project_id,
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=reference_time,
            )

        return self._sync_run(_run)

    async def search_async(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
    ) -> list[Any]:
        g = self._get_graphiti()
        if g is None:
            return []

        search_filter = None
        if node_labels:
            search_filter = self._search_filters_cls(node_labels=node_labels)

        return await g.search(
            query=query,
            group_ids=[project_id],
            num_results=limit,
            search_filter=search_filter,
        )

    def search(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
    ) -> list[Any]:
        if not self.enabled:
            return []

        async def _run():
            return await self.search_async(
                project_id=project_id,
                query=query,
                limit=limit,
                node_labels=node_labels,
            )

        return self._sync_run(_run)
