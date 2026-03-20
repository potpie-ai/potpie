"""Thin async wrapper around Graphiti for the context graph.

Uses the same Neo4j instance as the code graph (NEO4J_URI). Respects CONTEXT_GRAPH_ENABLED.
Callers in Celery tasks must use BaseTask.run_async(coro); agent tools may use asyncio.run().
"""

from datetime import datetime
from typing import Any

from app.core.config_provider import config_provider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Optional: graphiti-core may not be installed in all environments
try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType

    _GRAPHITI_AVAILABLE = True
except ImportError:
    Graphiti = None  # type: ignore[misc, assignment]
    EpisodeType = None  # type: ignore[misc, assignment]
    _GRAPHITI_AVAILABLE = False


class ContextGraphClient:
    """Async client for adding episodes and searching the context graph."""

    def __init__(self) -> None:
        cfg = config_provider.get_context_graph_config()
        self._enabled = bool(cfg.get("enabled") and _GRAPHITI_AVAILABLE)
        self._graphiti: Any = None
        if self._enabled and cfg.get("neo4j_uri") and Graphiti is not None:
            try:
                self._graphiti = Graphiti(
                    uri=cfg["neo4j_uri"],
                    user=cfg.get("neo4j_user") or "",
                    password=cfg.get("neo4j_password") or "",
                )
            except Exception as e:
                logger.warning("Context graph: failed to init Graphiti client: %s", e)
                self._enabled = False
                self._graphiti = None
        elif self._enabled and not _GRAPHITI_AVAILABLE:
            logger.warning("Context graph: graphiti-core not installed")
            self._enabled = False

    async def add_episode(
        self,
        project_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> str:
        """Add an episode to the context graph. Returns Graphiti episode UUID or empty string if disabled."""
        if not self._enabled or self._graphiti is None:
            return ""
        if EpisodeType is None:
            return ""
        try:
            result = await self._graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=reference_time,
                group_id=project_id,
            )
            # AddEpisodeResults has .episode.uuid
            episode = getattr(result, "episode", result)
            return str(getattr(episode, "uuid", "") or "")
        except Exception as e:
            logger.exception("Context graph add_episode failed: %s", e)
            raise

    async def search(self, project_id: str, query: str, limit: int = 10) -> list:
        """Search the context graph within the project namespace. Returns list of edges (facts)."""
        if not self._enabled or self._graphiti is None:
            return []
        try:
            return await self._graphiti.search(
                query=query,
                group_ids=[project_id],
                num_results=limit,
            )
        except Exception as e:
            logger.exception("Context graph search failed: %s", e)
            raise
