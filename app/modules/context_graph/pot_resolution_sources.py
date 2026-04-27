"""Build ResolvedPot github repos from project_sources (+ legacy project.repo_name)."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from integrations.application.project_sources_service import (
    ensure_github_repository_source,
    list_github_repository_sources,
)
from app.modules.projects.projects_model import Project
from domain.ports.pot_resolution import ResolvedPot, ResolvedPotRepo, single_github_repo_pot

logger = logging.getLogger(__name__)


def github_resolved_pot_from_project(db: Session, project: Project) -> ResolvedPot | None:
    if not project:
        return None
    ready = (project.status or "").lower() == "ready"
    rows = list_github_repository_sources(db, project.id)
    if not rows and project.repo_name and str(project.repo_name).strip():
        try:
            ensure_github_repository_source(db, project.id, str(project.repo_name).strip())
        except Exception:
            logger.warning(
                "Failed to auto-create GitHub source for project %s; continuing with legacy fallback",
                project.id,
                exc_info=True,
            )
        rows = list_github_repository_sources(db, project.id)
    seen_repos: set[str] = set()
    repos: list[ResolvedPotRepo] = []
    for ps in rows:
        rn = (ps.scope_json or {}).get("repo_name")
        if not rn:
            continue
        normalized = str(rn).strip().lower()
        if normalized in seen_repos:
            continue
        seen_repos.add(normalized)
        repos.append(
            ResolvedPotRepo(
                pot_id=project.id,
                repo_id=project.id,
                provider="github",
                provider_host="github.com",
                repo_name=str(rn).strip(),
                ready=ready,
            )
        )
    if repos:
        display = project.repo_name or repos[0].repo_name
        return ResolvedPot(
            pot_id=project.id,
            name=display,
            repos=repos,
            ready=ready,
        )
    if not project.repo_name or not str(project.repo_name).strip():
        logger.debug("No GitHub sources and no repo_name for project %s", project.id)
        return None
    return single_github_repo_pot(
        pot_id=project.id,
        repo_name=str(project.repo_name).strip(),
        ready=ready,
        name=str(project.repo_name).strip(),
    )
