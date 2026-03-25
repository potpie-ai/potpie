"""Thin wrapper delegating to context-engine Graphiti adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.modules.context_graph.wiring import PotpieContextEngineSettings
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter


class ContextGraphClient:
    """Compatibility shim for intelligence tools and legacy imports."""

    def __init__(self) -> None:
        self._impl = GraphitiEpisodicAdapter(PotpieContextEngineSettings())

    @property
    def enabled(self) -> bool:
        return self._impl.enabled

    async def add_episode_async(
        self,
        project_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        return await self._impl.add_episode_async(
            project_id=project_id,
            name=name,
            episode_body=episode_body,
            source_description=source_description,
            reference_time=reference_time,
        )

    def add_episode(
        self,
        project_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        return self._impl.add_episode(
            project_id=project_id,
            name=name,
            episode_body=episode_body,
            source_description=source_description,
            reference_time=reference_time,
        )

    async def search_async(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
    ) -> list[Any]:
        return await self._impl.search_async(
            project_id=project_id,
            query=query,
            limit=limit,
            node_labels=node_labels,
        )

    def search(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
    ) -> list[Any]:
        return self._impl.search(
            project_id=project_id,
            query=query,
            limit=limit,
            node_labels=node_labels,
        )
