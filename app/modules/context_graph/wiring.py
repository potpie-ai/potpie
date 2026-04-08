"""Potpie-specific wiring for context-engine (ports → host services)."""

from __future__ import annotations

import os

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.context_graph.code_provider_source_control import (
    CodeProviderSourceControl,
)
from app.modules.context_graph.pot_resolution_sources import (
    github_resolved_pot_from_project,
)
from integrations.adapters.outbound.postgres.project_source_model import ProjectSource
from app.modules.projects.projects_model import Project
from bootstrap.container import ContextEngineContainer, build_container
from domain.ports.pot_resolution import (
    PotResolutionPort,
    RepoRef,
    ResolvedPot,
    ResolvedPotRepo,
)
from domain.ports.settings import ContextEngineSettingsPort


class PotpieContextEngineSettings(ContextEngineSettingsPort):
    def __init__(self, cp=None) -> None:
        self._cp = cp or config_provider

    def is_enabled(self) -> bool:
        return bool(self._cp.get_context_graph_config().get("enabled"))

    def neo4j_uri(self) -> str | None:
        v = (
            os.getenv("CONTEXT_ENGINE_NEO4J_URI") or os.getenv("CONTEXT_ENGINE_NEO4J_URL") or ""
        ).strip()
        if v:
            return v
        c = self._cp.get_neo4j_config()
        return c.get("uri")

    def neo4j_user(self) -> str | None:
        v = (
            os.getenv("CONTEXT_ENGINE_NEO4J_USERNAME") or os.getenv("CONTEXT_ENGINE_NEO4J_USER") or ""
        ).strip()
        if v:
            return v
        c = self._cp.get_neo4j_config()
        return c.get("username")

    def neo4j_password(self) -> str | None:
        if os.getenv("CONTEXT_ENGINE_NEO4J_PASSWORD") is not None:
            v = os.getenv("CONTEXT_ENGINE_NEO4J_PASSWORD", "").strip()
            return v
        c = self._cp.get_neo4j_config()
        return c.get("password")

    def backfill_max_prs_per_run(self) -> int:
        return int(self._cp.get_context_graph_config().get("backfill_max_prs_per_run", 100))


class SqlalchemyPotResolution(PotResolutionPort):
    def __init__(self, db: Session) -> None:
        self._db = db

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        project = self._db.query(Project).filter(Project.id == pot_id).first()
        if not project:
            return None
        return github_resolved_pot_from_project(self._db, project)

    def known_pot_ids(self) -> list[str]:
        rows = (
            self._db.query(Project.id)
            .filter(
                Project.repo_name.isnot(None),
                Project.repo_name != "",
                func.lower(Project.status) == "ready",
            )
            .all()
        )
        return [r[0] for r in rows]

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        want = ref.repo_name.lower()
        ids: set[str] = set()
        rows = (
            self._db.query(Project.id)
            .filter(
                Project.repo_name.isnot(None),
                func.lower(Project.repo_name) == want,
            )
            .all()
        )
        ids.update(r[0] for r in rows)
        repo_json = ProjectSource.scope_json["repo_name"].astext
        src_rows = (
            self._db.query(ProjectSource.project_id)
            .filter(
                ProjectSource.provider == "github",
                ProjectSource.source_kind == "repository",
                func.lower(repo_json) == want,
            )
            .distinct()
            .all()
        )
        ids.update(r[0] for r in src_rows)
        return list(ids)

    def list_pot_repos(self, pot_id: str) -> list[ResolvedPotRepo]:
        r = self.resolve_pot(pot_id)
        return list(r.repos) if r else []

    def get_repo_in_pot(self, pot_id: str, ref: RepoRef) -> ResolvedPotRepo | None:
        r = self.resolve_pot(pot_id)
        if not r:
            return None
        want = ref.repo_name.lower()
        for rr in r.repos:
            if rr.repo_name.lower() == want:
                return rr
        return None


class UserScopedSqlalchemyPotResolution(PotResolutionPort):
    """Resolve pots for a single user (HTTP / agent surfaces)."""

    def __init__(self, db: Session, user_id: str) -> None:
        self._db = db
        self._user_id = user_id

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        project = (
            self._db.query(Project)
            .filter(Project.id == pot_id, Project.user_id == self._user_id)
            .first()
        )
        if not project:
            return None
        return github_resolved_pot_from_project(self._db, project)

    def known_pot_ids(self) -> list[str]:
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

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        want = ref.repo_name.lower()
        ids: set[str] = set()
        rows = (
            self._db.query(Project.id)
            .filter(
                Project.user_id == self._user_id,
                Project.repo_name.isnot(None),
                func.lower(Project.repo_name) == want,
            )
            .all()
        )
        ids.update(r[0] for r in rows)
        repo_json = ProjectSource.scope_json["repo_name"].astext
        src_rows = (
            self._db.query(ProjectSource.project_id)
            .join(Project, Project.id == ProjectSource.project_id)
            .filter(
                Project.user_id == self._user_id,
                ProjectSource.provider == "github",
                ProjectSource.source_kind == "repository",
                func.lower(repo_json) == want,
            )
            .distinct()
            .all()
        )
        ids.update(r[0] for r in src_rows)
        return list(ids)

    def list_pot_repos(self, pot_id: str) -> list[ResolvedPotRepo]:
        r = self.resolve_pot(pot_id)
        return list(r.repos) if r else []

    def get_repo_in_pot(self, pot_id: str, ref: RepoRef) -> ResolvedPotRepo | None:
        r = self.resolve_pot(pot_id)
        if not r:
            return None
        want = ref.repo_name.lower()
        for rr in r.repos:
            if rr.repo_name.lower() == want:
                return rr
        return None


def build_container_for_session(db: Session) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    return build_container(
        settings=PotpieContextEngineSettings(),
        pots=SqlalchemyPotResolution(db),
        source_for_repo=source_for_repo,
    )


def build_container_for_user_session(db: Session, user_id: str) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    return build_container(
        settings=PotpieContextEngineSettings(),
        pots=UserScopedSqlalchemyPotResolution(db, user_id),
        source_for_repo=source_for_repo,
    )
