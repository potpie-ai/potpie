"""Graphiti / episodic knowledge graph (port)."""

from datetime import datetime
from typing import Any, Optional, Protocol


class EpisodicGraphPort(Protocol):
    @property
    def enabled(self) -> bool:
        ...

    def add_episode(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        """Return episode UUID or None if disabled/failed."""

    async def add_episode_async(
        self,
        pot_id: str,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
    ) -> Optional[str]:
        ...

    def search(
        self,
        pot_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        repo_name: str | None = None,
    ) -> list[Any]:
        ...

    async def search_async(
        self,
        pot_id: str,
        query: str,
        limit: int = 10,
        node_labels: Optional[list[str]] = None,
        repo_name: str | None = None,
    ) -> list[Any]:
        ...
