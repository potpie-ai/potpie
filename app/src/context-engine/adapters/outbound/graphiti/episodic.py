"""Graphiti adapter implementing EpisodicGraphPort."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from datetime import datetime
from collections.abc import Coroutine
from typing import Any, Callable, Optional, TypeVar

from adapters.outbound.graphiti.edge_extraction_normalize import (
    install_extract_edges_normalize_patch,
)
from domain.entity_schema import (
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
    GRAPHITI_CUSTOM_EXTRACTION_INSTRUCTIONS,
)
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
        # Graphiti's Neo4jDriver.__init__ schedules build_indices_and_constraints()
        # as a background task on this loop. Closing the driver while that task is
        # still mid-commit yields IncompleteCommit (defunct connection). Drain all
        # pending tasks on this loop before tearing the driver down.
        try:
            current = asyncio.current_task()
            pending = [
                t
                for t in asyncio.all_tasks(loop)
                if t is not current and not t.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:
            logger.debug("Graphiti pending-task drain: %s", exc)
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
        provenance: ProvenanceRef | None = None,
    ) -> Optional[str]:
        g = self._get_graphiti()
        if g is None:
            return None

        install_extract_edges_normalize_patch()
        result = await g.add_episode(
            name=name,
            episode_body=episode_body,
            source_description=source_description,
            reference_time=reference_time,
            group_id=pot_id,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
            custom_extraction_instructions=GRAPHITI_CUSTOM_EXTRACTION_INSTRUCTIONS,
        )
        episode = getattr(result, "episode", None)
        episode_uuid = getattr(episode, "uuid", None)
        if episode_uuid and provenance is not None:
            try:
                from adapters.outbound.graphiti.apply_episode_provenance import (
                    apply_episode_provenance,
                )

                await apply_episode_provenance(
                    g.driver, pot_id, str(episode_uuid), provenance
                )
            except Exception as exc:
                logger.warning("Episode provenance stamp failed: %s", exc)
        try:
            from adapters.outbound.graphiti.temporal_supersede import (
                apply_predicate_family_auto_supersede,
            )

            await apply_predicate_family_auto_supersede(g.driver, pot_id)
        except Exception as exc:
            logger.warning("Predicate-family auto-supersede failed: %s", exc)
        try:
            from adapters.outbound.graphiti.apply_canonical_labels import (
                apply_episodic_canonical_labels,
            )

            await apply_episodic_canonical_labels(g.driver, pot_id)
        except Exception as exc:
            logger.warning("Canonical label inference failed: %s", exc)
        try:
            from adapters.outbound.graphiti.family_conflict_detection import (
                apply_family_conflict_detection,
            )

            await apply_family_conflict_detection(g.driver, pot_id)
        except Exception as exc:
            logger.warning("Family conflict detection failed: %s", exc)
        return str(episode_uuid) if episode_uuid else None

    def add_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        provenance: ProvenanceRef | None = None,
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
                provenance=provenance,
            )

        return self._sync_run(_run)

    def write_episode_drafts(
        self,
        pot_id: str,
        drafts: list[EpisodeDraft],
        provenance: ProvenanceRef | None = None,
    ) -> list[Optional[str]]:
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
                    provenance=provenance,
                )
            )
        return out

    def _build_search_filters(
        self,
        node_labels: Optional[list[str]],
        *,
        include_invalidated: bool,
        as_of: Optional[datetime] = None,
    ) -> Any | None:
        """Graphiti search filters; by default exclude edges with ``invalid_at`` set.

        During ingestion, Graphiti may mark contradicted facts invalid; optional
        ``temporal_supersede`` can also stamp ``invalid_at`` (Neo4j). Hybrid search
        would still match invalidated edges unless we filter here.

        When ``as_of`` is set, restrict to edges valid at that instant (valid_at
        unset or <= as_of, and invalid_at unset or > as_of). ``include_invalidated``
        is ignored in that case.

        Episodic ``source_description`` filtering is applied after search by joining
        ``Episodic`` nodes (Graphiti edge search does not apply ``property_filters``).
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
        # node_labels intentionally NOT passed to Graphiti filters: upstream builds
        # `n:Label AND m:Label`, requiring both endpoints to carry it. Our inferred
        # labels (e.g. CAUSED→Incident) sit only on one endpoint, so we post-filter
        # in `_finalize_search_edges` using an OR-match on native Neo4j labels.
        if not kwargs:
            return None
        return self._search_filters_cls(**kwargs)

    @staticmethod
    def _datetime_like_to_iso(val: Any) -> Optional[str]:
        if val is None:
            return None
        fn = getattr(val, "isoformat", None)
        if callable(fn):
            try:
                out = fn()
                return str(out) if out is not None else None
            except Exception:
                return None
        s = str(val).strip()
        return s or None

    async def _load_episodic_metadata(
        self,
        driver: Any,
        pot_id: str,
        episode_uuids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Map episode uuid → ``source_description`` and ``valid_at`` (observation time)."""
        if not episode_uuids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        try:
            async with driver.session() as session:
                res = await session.run(
                    """
                    MATCH (ep:Episodic)
                    WHERE ep.uuid IN $uuids AND ep.group_id = $gid
                    RETURN ep.uuid AS uuid,
                           ep.source_description AS source_description,
                           ep.valid_at AS valid_at
                    """,
                    uuids=episode_uuids,
                    gid=pot_id,
                )
                async for record in res:
                    uid = record.get("uuid")
                    if uid is None:
                        continue
                    out[str(uid)] = {
                        "source_description": record.get("source_description"),
                        "valid_at": record.get("valid_at"),
                    }
        except Exception as exc:
            logger.warning("episodic provenance lookup failed: %s", exc)
        return out

    @staticmethod
    def _ordered_unique_sources(
        episode_order: list[str], meta: dict[str, dict[str, Any]]
    ) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for uid in episode_order:
            row = meta.get(str(uid))
            if not row:
                continue
            sd = row.get("source_description")
            if sd is None:
                continue
            s = str(sd).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            sources.append(s)
        return sources

    async def _enrich_edges_with_episode_provenance(
        self,
        driver: Any,
        pot_id: str,
        edges: list[Any],
    ) -> None:
        """Attach ``source_refs``, ``reference_time``, ``episode_uuid`` on each edge's ``attributes``."""
        if not edges:
            return
        collected: list[str] = []
        seen: set[str] = set()
        for edge in edges:
            eps = getattr(edge, "episodes", None) or []
            for u in eps:
                if not u:
                    continue
                s = str(u)
                if s not in seen:
                    seen.add(s)
                    collected.append(s)
        meta = await self._load_episodic_metadata(driver, pot_id, collected)
        for edge in edges:
            eps = [str(u) for u in (getattr(edge, "episodes", None) or []) if u]
            base = dict(edge.attributes) if isinstance(edge.attributes, dict) else {}
            sources = self._ordered_unique_sources(eps, meta)
            if sources:
                base["source_refs"] = sources
            primary_ep = eps[0] if eps else None
            ref_iso: Optional[str] = None
            if primary_ep:
                row = meta.get(str(primary_ep))
                if row:
                    ref_iso = self._datetime_like_to_iso(row.get("valid_at"))
            if ref_iso:
                base["reference_time"] = ref_iso
            if primary_ep:
                base["episode_uuid"] = primary_ep
            edge.attributes = base

    @staticmethod
    def _edge_matches_post_filters(
        edge: Any,
        *,
        source_description: Optional[str],
        episode_uuid: Optional[str],
    ) -> bool:
        want_ep = episode_uuid.strip() if episode_uuid and episode_uuid.strip() else None
        if want_ep:
            eps = [str(x) for x in (getattr(edge, "episodes", None) or [])]
            if want_ep not in eps:
                return False
        want_src = source_description.strip() if source_description and source_description.strip() else None
        if not want_src:
            return True
        attrs = getattr(edge, "attributes", None)
        refs: list[str] = []
        if isinstance(attrs, dict):
            raw = attrs.get("source_refs")
            if isinstance(raw, list):
                refs = [str(x) for x in raw if x is not None and str(x).strip()]
        return want_src in refs

    async def _finalize_search_edges(
        self,
        g: Any,
        pot_id: str,
        edges: list[Any],
        scores: list[float],
        *,
        limit: int,
        source_description: Optional[str],
        episode_uuid: Optional[str],
        node_labels: Optional[list[str]] = None,
    ) -> list[Any]:
        await self._enrich_edges_with_episode_provenance(g.driver, pot_id, edges)
        for i, edge in enumerate(edges):
            sc = float(scores[i]) if i < len(scores) else 0.0
            base = dict(edge.attributes) if isinstance(edge.attributes, dict) else {}
            base["_context_similarity_score"] = sc
            edge.attributes = base
        if node_labels:
            edges = await self._filter_edges_by_endpoint_labels(
                g.driver, pot_id, edges, node_labels
            )
        filtered = [
            e
            for e in edges
            if self._edge_matches_post_filters(
                e, source_description=source_description, episode_uuid=episode_uuid
            )
        ]
        return filtered[: max(1, min(limit, 50))]

    @staticmethod
    async def _filter_edges_by_endpoint_labels(
        driver: Any,
        pot_id: str,
        edges: list[Any],
        node_labels: list[str],
    ) -> list[Any]:
        """Keep edges where either endpoint carries any of the requested native labels.

        Worked around because graphiti_core builds ``n:Label AND m:Label`` (both
        endpoints). Our canonical-label inference usually labels only one end
        (e.g. CAUSED→Incident targets, DECIDES_FOR→Decision targets).
        """
        uuids: set[str] = set()
        for e in edges:
            s = getattr(e, "source_node_uuid", None)
            t = getattr(e, "target_node_uuid", None)
            if s:
                uuids.add(str(s))
            if t:
                uuids.add(str(t))
        if not uuids:
            return edges
        query = """
        MATCH (n:Entity {group_id: $gid})
        WHERE n.uuid IN $uuids
        RETURN n.uuid AS uuid, labels(n) AS labels
        """
        try:
            records, _, _ = await driver.execute_query(
                query, gid=pot_id, uuids=list(uuids)
            )
        except Exception as exc:
            logger.warning("Endpoint-label post-filter query failed: %s", exc)
            return edges
        wanted = {str(lb) for lb in node_labels}
        labels_by_uuid: dict[str, set[str]] = {}
        for row in records or []:
            u = row.get("uuid")
            lbls = row.get("labels") or []
            if u is not None:
                labels_by_uuid[str(u)] = {str(lb) for lb in lbls}
        out: list[Any] = []
        for e in edges:
            s = str(getattr(e, "source_node_uuid", "") or "")
            t = str(getattr(e, "target_node_uuid", "") or "")
            s_lbls = labels_by_uuid.get(s, set())
            t_lbls = labels_by_uuid.get(t, set())
            if (s_lbls & wanted) or (t_lbls & wanted):
                out.append(e)
        return out

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
        episode_uuid: Optional[str] = None,
    ) -> list[Any]:
        del repo_name  # optional future filter; Graphiti search is pot-scoped
        g = self._get_graphiti()
        if g is None:
            return []

        need_wide = (
            (source_description and source_description.strip())
            or (episode_uuid and episode_uuid.strip())
            or (node_labels and len(node_labels) > 0)
        )
        fetch_limit = min(50, max(limit * 8, 16)) if need_wide else limit
        fetch_limit = max(1, min(fetch_limit, 50))

        search_filter = self._build_search_filters(
            node_labels,
            include_invalidated=include_invalidated,
            as_of=as_of,
        )

        try:
            from graphiti_core.search.search import search as graphiti_search
            from graphiti_core.search.search_config import SearchFilters
            from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
        except Exception as exc:
            logger.warning("graphiti search helpers unavailable: %s", exc)
            edges = await g.search(
                query=query,
                group_ids=[pot_id],
                num_results=fetch_limit,
                search_filter=search_filter,
            )
            scores = [1.0 / (1 + i) for i in range(len(edges))]
            return await self._finalize_search_edges(
                g,
                pot_id,
                list(edges),
                scores,
                limit=limit,
                source_description=source_description,
                episode_uuid=episode_uuid,
                node_labels=node_labels,
            )

        search_config = EDGE_HYBRID_SEARCH_RRF.model_copy()
        search_config.limit = fetch_limit
        sf = search_filter if search_filter is not None else SearchFilters()
        sr = await graphiti_search(
            g.clients,
            query,
            [pot_id],
            search_config,
            sf,
            driver=g.driver,
        )
        edges = list(sr.edges)
        scores = list(sr.edge_reranker_scores)
        return await self._finalize_search_edges(
            g,
            pot_id,
            edges,
            scores,
            limit=limit,
            source_description=source_description,
            episode_uuid=episode_uuid,
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
        episode_uuid: Optional[str] = None,
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
                episode_uuid=episode_uuid,
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

    def list_open_conflicts(self, pot_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        async def _run() -> list[dict[str, Any]]:
            g = self._get_graphiti()
            if g is None:
                return []
            from adapters.outbound.graphiti.family_conflict_detection import (
                list_open_conflicts_async,
            )

            return await list_open_conflicts_async(g.driver, pot_id)

        try:
            return self._sync_run(_run)
        except Exception as exc:
            logger.warning("list_open_conflicts failed: %s", exc)
            return []

    def resolve_open_conflict(
        self, pot_id: str, issue_uuid: str, action: str
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "graphiti_disabled"}

        async def _run() -> dict[str, Any]:
            g = self._get_graphiti()
            if g is None:
                return {
                    "ok": False,
                    "error": self.failure_reason() or "graphiti_unavailable",
                }
            from adapters.outbound.graphiti.family_conflict_detection import (
                resolve_conflict_supersede_older_async,
            )

            a = action.strip().lower()
            if a in ("supersede_older", "supersede"):
                return await resolve_conflict_supersede_older_async(
                    g.driver, pot_id, issue_uuid
                )
            return {"ok": False, "error": "unsupported_action"}

        try:
            return self._sync_run(_run)
        except Exception as exc:
            logger.warning("resolve_open_conflict failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def relabel_nodes_from_edges_for_pot(self, pot_id: str) -> dict[str, Any]:
        """Backfill canonical labels from episodic edge patterns (idempotent; ignores infer flag)."""
        if not self._enabled:
            return {"ok": False, "error": "graphiti_disabled"}

        async def _run():
            g = self._get_graphiti()
            if g is None:
                return {
                    "ok": False,
                    "error": self.failure_reason() or "graphiti_unavailable",
                }
            from adapters.outbound.graphiti.apply_canonical_labels import (
                relabel_nodes_from_edges,
            )

            return await relabel_nodes_from_edges(g.driver, pot_id)

        return self._sync_run(_run)

    def classify_modified_edges_for_pot(
        self, pot_id: str, *, dry_run: bool = True
    ) -> dict[str, Any]:
        """Reclassify ``MODIFIED`` episodic edges for this pot (Neo4j)."""
        if not self.enabled:
            return {"ok": False, "error": "graphiti_disabled"}

        async def _run():
            g = self._get_graphiti()
            if g is None:
                return {
                    "ok": False,
                    "error": self.failure_reason() or "graphiti_unavailable",
                }
            from adapters.outbound.graphiti.classify_modified_edges import (
                classify_modified_edges_for_group,
            )

            return await classify_modified_edges_for_group(
                g.driver, pot_id, dry_run=dry_run
            )

        return self._sync_run(_run)
