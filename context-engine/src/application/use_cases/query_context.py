"""Read-side queries for agents and APIs."""

from __future__ import annotations

from typing import Any, Optional

from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.structural_graph import StructuralGraphPort


def get_change_history(
    structural: StructuralGraphPort,
    project_id: str,
    *,
    function_name: Optional[str] = None,
    file_path: Optional[str] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    return structural.get_change_history(
        project_id=project_id,
        function_name=function_name,
        file_path=file_path,
        limit=max(1, min(limit, 100)),
    )


def get_file_owners(
    structural: StructuralGraphPort,
    project_id: str,
    file_path: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    return structural.get_file_owners(
        project_id=project_id,
        file_path=file_path,
        limit=max(1, min(limit, 50)),
    )


def get_decisions(
    structural: StructuralGraphPort,
    project_id: str,
    *,
    file_path: Optional[str] = None,
    function_name: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return structural.get_decisions(
        project_id=project_id,
        file_path=file_path,
        function_name=function_name,
        limit=max(1, min(limit, 100)),
    )


def search_project_context(
    episodic: EpisodicGraphPort,
    project_id: str,
    query: str,
    *,
    limit: int = 8,
    node_labels: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    if not episodic.enabled:
        return []
    results = episodic.search(
        project_id=project_id,
        query=query,
        limit=max(1, min(limit, 50)),
        node_labels=node_labels,
    )
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "uuid": str(getattr(item, "uuid", "")),
                "name": getattr(item, "name", None),
                "summary": getattr(item, "summary", None),
                "fact": getattr(item, "fact", None),
            }
        )
    return rows


async def search_project_context_async(
    episodic: EpisodicGraphPort,
    project_id: str,
    query: str,
    *,
    limit: int = 8,
    node_labels: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    if not episodic.enabled:
        return []
    results = await episodic.search_async(
        project_id=project_id,
        query=query,
        limit=max(1, min(limit, 50)),
        node_labels=node_labels,
    )
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "uuid": str(getattr(item, "uuid", "")),
                "name": getattr(item, "name", None),
                "summary": getattr(item, "summary", None),
                "fact": getattr(item, "fact", None),
            }
        )
    return rows
