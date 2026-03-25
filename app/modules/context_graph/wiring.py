"""Potpie-specific wiring for context-engine (ports → host services)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.context_graph.code_provider_source_control import (
    CodeProviderSourceControl,
)
from app.modules.projects.projects_model import Project
from bootstrap.container import ContextEngineContainer, build_container
from domain.ports.project_resolution import ProjectResolutionPort, ResolvedProject
from domain.ports.settings import ContextEngineSettingsPort

if TYPE_CHECKING:
    pass


class PotpieContextEngineSettings(ContextEngineSettingsPort):
    def __init__(self, cp=None) -> None:
        self._cp = cp or config_provider

    def is_enabled(self) -> bool:
        return bool(self._cp.get_context_graph_config().get("enabled"))

    def neo4j_uri(self) -> str | None:
        c = self._cp.get_neo4j_config()
        return c.get("uri")

    def neo4j_user(self) -> str | None:
        c = self._cp.get_neo4j_config()
        return c.get("username")

    def neo4j_password(self) -> str | None:
        c = self._cp.get_neo4j_config()
        return c.get("password")


class SqlalchemyProjectResolution(ProjectResolutionPort):
    def __init__(self, db: Session) -> None:
        self._db = db

    def resolve(self, project_id: str) -> ResolvedProject | None:
        project = self._db.query(Project).filter(Project.id == project_id).first()
        if not project or not project.repo_name:
            return None
        ready = (project.status or "").lower() == "ready"
        return ResolvedProject(
            project_id=project.id,
            repo_name=project.repo_name,
            ready=ready,
        )


class UserScopedSqlalchemyProjectResolution(ProjectResolutionPort):
    """Resolve projects for a single user (HTTP / agent surfaces)."""

    def __init__(self, db: Session, user_id: str) -> None:
        self._db = db
        self._user_id = user_id

    def resolve(self, project_id: str) -> ResolvedProject | None:
        project = (
            self._db.query(Project)
            .filter(Project.id == project_id, Project.user_id == self._user_id)
            .first()
        )
        if not project or not project.repo_name:
            return None
        ready = (project.status or "").lower() == "ready"
        return ResolvedProject(
            project_id=project.id,
            repo_name=project.repo_name,
            ready=ready,
        )

    def known_project_ids(self) -> list[str]:
        rows = (
            self._db.query(Project.id)
            .filter(
                Project.user_id == self._user_id,
                Project.repo_name.isnot(None),
                Project.repo_name != "",
                func.lower(Project.status) == "ready",
            )
            .all()
        )
        return [r[0] for r in rows]


def build_container_for_session(db: Session) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    return build_container(
        settings=PotpieContextEngineSettings(),
        projects=SqlalchemyProjectResolution(db),
        source_for_repo=source_for_repo,
    )


def build_container_for_user_session(db: Session, user_id: str) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    return build_container(
        settings=PotpieContextEngineSettings(),
        projects=UserScopedSqlalchemyProjectResolution(db, user_id),
        source_for_repo=source_for_repo,
    )
