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
from domain.graph_mutations import ProvenanceRef
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.reconciliation import EpisodeDraft

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
        self._comparison_operator_cls = None
        self._date_filter_cls = None
        self._init_error: Optional[str] = None

        if not self._enabled:
            return

        try:
            from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

            self._search_filters_cls = SearchFilters
            self._comparison_operator_cls = ComparisonOperator
            self._date_filter_cls = DateFilter
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

    @staticmethod
    async def _await_close_async_http_client(http: Any) -> None:
        """Close OpenAI/Voyage/etc. async HTTP clients while the loop is still running."""
        if http is None:
            return
        close_fn = getattr(http, "close", None)
        if close_fn is None or not asyncio.iscoroutinefunction(close_fn):
            return
        try:
            is_closed = getattr(http, "is_closed", None)
            if callable(is_closed) and is_closed():
                return
        except Exception:
            pass
        try:
            await close_fn()
        except Exception as exc:
            logger.debug("Async HTTP client close: %s", exc)

    async def _close_graphiti_aux_http_clients(self, graphiti: Any) -> None:
        """Close LLM/embedder/reranker HTTP clients (Graphiti.close() only closes Neo4j).

        If these stay open, their httpx teardown can run after ``asyncio.run`` closes the
        loop → RuntimeError: Event loop is closed.
        """
        for attr in ("llm_client", "embedder", "cross_encoder"):
            wrapper = getattr(graphiti, attr, None)
            if wrapper is None:
                continue
            sub = getattr(wrapper, "client", None)
            await self._await_close_async_http_client(sub)

    async def _close_graphiti_for_running_loop(self) -> None:
        """Close driver and HTTP clients before the ephemeral asyncio.run loop shuts down.

        If we skip this, Neo4j async transports or httpx/OpenAI may try to schedule
        cleanup on a loop that asyncio.run() has already closed → RuntimeError:
        Event loop is closed.
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
            await self._close_graphiti_aux_http_clients(client)
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

    def write_episode_drafts(
        self,
        pot_id: str,
        drafts: list[EpisodeDraft],
        provenance: ProvenanceRef | None = None,
    ) -> list[Optional[str]]:
        del provenance  # reserved for future Graphiti tagging / provenance metadata
        if not self.enabled or not drafts:
            return [None] * len(drafts)
        out: list[Optional[str]] = []
        for d in drafts:
            out.append(
                self.add_episode(
                    pot_id=pot_id,
                    name=d.name,
                    episode_body=d.episode_body,
                    source_description=d.source_description,
                    reference_time=d.reference_time,
                )
            )
        return out

    def _build_search_filters(
        self,
        node_labels: Optional[list[str]],
        source_description: Optional[str],
        *,
        include_invalidated: bool,
        as_of: Optional[datetime] = None,
    ) -> Any | None:
        """Graphiti search filters; by default exclude edges with ``invalid_at`` set.

        During ingestion, ``resolve_extracted_edges`` can mark older contradicting
        facts invalid; hybrid search still matched them unless we filter here.

        When ``as_of`` is set, restrict to edges valid at that instant (valid_at
        unset or <= as_of, and invalid_at unset or > as_of). ``include_invalidated``
        is ignored in that case.
        """
        if not self._search_filters_cls or not self._comparison_operator_cls or not self._date_filter_cls:
            return None
        kwargs: dict[str, Any] = {}
        CO = self._comparison_operator_cls
        DF = self._date_filter_cls
        if as_of is not None:
            kwargs["valid_at"] = [
                [DF(date=None, comparison_operator=CO.is_null)],
                [DF(date=as_of, comparison_operator=CO.less_than_equal)],
            ]
            kwargs["invalid_at"] = [
                [DF(date=None, comparison_operator=CO.is_null)],
                [DF(date=as_of, comparison_operator=CO.greater_than)],
            ]
        elif not include_invalidated:
            kwargs["invalid_at"] = [[DF(date=None, comparison_operator=CO.is_null)]]
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
        *,
        include_invalidated: bool = False,
        as_of: Optional[datetime] = None,
    ) -> list[Any]:
        del repo_name  # optional future filter; Graphiti search is pot-scoped
        g = self._get_graphiti()
        if g is None:
            return []

        search_filter = self._build_search_filters(
            node_labels,
            source_description,
            include_invalidated=include_invalidated,
            as_of=as_of,
        )

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
        *,
        include_invalidated: bool = False,
        as_of: Optional[datetime] = None,
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
                include_invalidated=include_invalidated,
                as_of=as_of,
            )

        return self._sync_run(_run)

    async def reset_pot_async(self, pot_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "graphiti_disabled"}
        g = self._get_graphiti()
        if g is None:
            return {"ok": False, "error": self.failure_reason() or "graphiti_unavailable"}
        try:
            from graphiti_core.errors import GroupIdValidationError
            from graphiti_core.helpers import validate_group_id
            from graphiti_core.nodes import Node

            validate_group_id(pot_id)
        except GroupIdValidationError as exc:
            return {"ok": False, "error": f"invalid_pot_id: {exc}"}

        # Graphiti persists pot partitions via ``group_id`` on nodes in the driver's
        # default Neo4j database (e.g. ``neo4j``). ``Graphiti.add_episode`` compares
        # ``group_id`` to ``driver._database`` for *non-Neo4j* providers; Neo4jDriver
        # ``clone()`` is a no-op, so data always lives in the default DB — do not
        # target a separate catalog named after ``pot_id``.
        driver = g.driver
        try:
            async with driver.session() as session:
                cnt_res = await session.run(
                    "MATCH (n {group_id: $gid}) RETURN count(n) AS cnt",
                    gid=pot_id,
                )
                cnt_rec = await cnt_res.single()
                await cnt_res.consume()
                nodes_before = int(cnt_rec["cnt"]) if cnt_rec is not None else 0

            await Node.delete_by_group_id(driver, pot_id)
            async with driver.session() as session:
                await session.run(
                    """
                    MATCH (s:Saga {group_id: $gid})
                    CALL (s) {
                        DETACH DELETE s
                    } IN TRANSACTIONS OF $batch ROWS
                    """,
                    gid=pot_id,
                    batch=100,
                )
                # Entity--Entity edges in Neo4j use intermediate ``RelatesToNode_`` nodes
                # with ``group_id``. Graphiti's ``Node.delete_by_group_id`` only matches
                # ``Entity|Episodic|Community`` on Neo4j, so remove edge nodes explicitly.
                await session.run(
                    """
                    MATCH (n:RelatesToNode_ {group_id: $gid})
                    CALL (n) {
                        DETACH DELETE n
                    } IN TRANSACTIONS OF $batch ROWS
                    """,
                    gid=pot_id,
                    batch=100,
                )
                # Any remaining nodes tagged with this partition (new Graphiti labels, drift,
                # or types not covered by ``Node.delete_by_group_id``).
                sweep = await session.run(
                    """
                    MATCH (n {group_id: $gid})
                    CALL (n) {
                        DETACH DELETE n
                    } IN TRANSACTIONS OF $batch ROWS
                    """,
                    gid=pot_id,
                    batch=500,
                )
                await sweep.consume()

            async with driver.session() as session:
                verify_res = await session.run(
                    "MATCH (n {group_id: $gid}) RETURN count(n) AS cnt",
                    gid=pot_id,
                )
                verify_rec = await verify_res.single()
                await verify_res.consume()
                remaining = int(verify_rec["cnt"]) if verify_rec is not None else 0
            if remaining:
                return {
                    "ok": False,
                    "error": "group_id_reset_incomplete",
                    "group_id_nodes_before": nodes_before,
                    "group_id_nodes_remaining": remaining,
                }
        except Exception as exc:
            logger.warning("reset_pot_async failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "group_id_nodes_before": nodes_before,
            "group_id_nodes_remaining": 0,
        }

    def reset_pot(self, pot_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "graphiti_disabled"}

        async def _run():
            return await self.reset_pot_async(pot_id)

        return self._sync_run(_run)
