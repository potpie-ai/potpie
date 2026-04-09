"""Read-side queries for agents and APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.structural_graph import StructuralGraphPort


def get_change_history(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    function_name: Optional[str] = None,
    file_path: Optional[str] = None,
    limit: int = 10,
    repo_name: Optional[str] = None,
    pr_number: Optional[int] = None,
) -> list[dict[str, Any]]:
    return structural.get_change_history(
        pot_id=pot_id,
        function_name=function_name,
        file_path=file_path,
        limit=max(1, min(limit, 100)),
        repo_name=repo_name,
        pr_number=pr_number,
    )


def get_file_owners(
    structural: StructuralGraphPort,
    pot_id: str,
    file_path: str,
    limit: int = 5,
    repo_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    return structural.get_file_owners(
        pot_id=pot_id,
        file_path=file_path,
        limit=max(1, min(limit, 50)),
        repo_name=repo_name,
    )


def get_decisions(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    file_path: Optional[str] = None,
    function_name: Optional[str] = None,
    limit: int = 20,
    repo_name: Optional[str] = None,
    pr_number: Optional[int] = None,
) -> list[dict[str, Any]]:
    return structural.get_decisions(
        pot_id=pot_id,
        file_path=file_path,
        function_name=function_name,
        limit=max(1, min(limit, 100)),
        repo_name=repo_name,
        pr_number=pr_number,
    )


def get_pr_review_context(
    structural: StructuralGraphPort,
    pot_id: str,
    pr_number: int,
    repo_name: Optional[str] = None,
) -> dict[str, Any]:
    if pr_number < 1:
        return {
            "found": False,
            "pr_number": pr_number,
            "pr_title": None,
            "pr_summary": None,
            "review_threads": [],
        }
    return structural.get_pr_review_context(pot_id, pr_number, repo_name=repo_name)


def get_pr_diff(
    structural: StructuralGraphPort,
    pot_id: str,
    pr_number: int,
    *,
    file_path: Optional[str] = None,
    limit: int = 30,
    repo_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    if pr_number < 1:
        return []
    return structural.get_pr_diff(
        pot_id=pot_id,
        pr_number=pr_number,
        file_path=file_path,
        limit=max(1, min(limit, 200)),
        repo_name=repo_name,
    )


def get_project_graph(
    structural: StructuralGraphPort,
    pot_id: str,
    *,
    pr_number: Optional[int] = None,
    limit: int = 12,
) -> dict[str, Any]:
    return structural.get_project_graph(
        pot_id,
        pr_number,
        max(1, min(limit, 50)),
    )


def _edge_datetime_iso(obj: Any) -> str | None:
    if obj is None:
        return None
    fn = getattr(obj, "isoformat", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return None
    return None


def search_pot_context(
    episodic: EpisodicGraphPort,
    pot_id: str,
    query: str,
    *,
    limit: int = 8,
    node_labels: Optional[list[str]] = None,
    repo_name: Optional[str] = None,
    source_description: Optional[str] = None,
    include_invalidated: bool = False,
    as_of: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    if not episodic.enabled:
        return []
    results = episodic.search(
        pot_id=pot_id,
        query=query,
        limit=max(1, min(limit, 50)),
        node_labels=node_labels,
        repo_name=repo_name,
        source_description=source_description,
        include_invalidated=include_invalidated,
        as_of=as_of,
    )
    rows: list[dict[str, Any]] = []
    for item in results:
        row = {
            "uuid": str(getattr(item, "uuid", "")),
            "name": getattr(item, "name", None),
            "summary": getattr(item, "summary", None),
            "fact": getattr(item, "fact", None),
        }
        for key in ("created_at", "valid_at", "invalid_at", "expired_at"):
            val = getattr(item, key, None)
            if val is not None:
                iso = _edge_datetime_iso(val)
                if iso is not None:
                    row[key] = iso
        rows.append(row)
    return rows


async def search_pot_context_async(
    episodic: EpisodicGraphPort,
    pot_id: str,
    query: str,
    *,
    limit: int = 8,
    node_labels: Optional[list[str]] = None,
    repo_name: Optional[str] = None,
    source_description: Optional[str] = None,
    include_invalidated: bool = False,
    as_of: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    if not episodic.enabled:
        return []
    results = await episodic.search_async(
        pot_id=pot_id,
        query=query,
        limit=max(1, min(limit, 50)),
        node_labels=node_labels,
        repo_name=repo_name,
        source_description=source_description,
        include_invalidated=include_invalidated,
        as_of=as_of,
    )
    rows: list[dict[str, Any]] = []
    for item in results:
        row = {
            "uuid": str(getattr(item, "uuid", "")),
            "name": getattr(item, "name", None),
            "summary": getattr(item, "summary", None),
            "fact": getattr(item, "fact", None),
        }
        for key in ("created_at", "valid_at", "invalid_at", "expired_at"):
            val = getattr(item, key, None)
            if val is not None:
                iso = _edge_datetime_iso(val)
                if iso is not None:
                    row[key] = iso
        rows.append(row)
    return rows
