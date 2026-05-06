"""Potpie-specific wiring for context-engine (ports → host services)."""

from __future__ import annotations

import json
import os

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from bootstrap.queue_factory import get_context_graph_job_queue
from adapters.outbound.reconciliation.factory import try_pydantic_deep_reconciliation_agent
from app.modules.code_provider.provider_factory import CodeProviderFactory
from app.modules.context_graph.code_provider_source_control import (
    CodeProviderSourceControl,
)
from app.modules.context_graph.context_graph_pot_member_model import ContextGraphPotMember
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
from app.modules.context_graph.context_graph_pot_source_model import (
    ContextGraphPotSource,
)
from adapters.outbound.source_resolvers import (
    CompositeSourceResolver,
    DocumentationUriResolver,
    GitHubPullRequestResolver,
)
from bootstrap.container import ContextEngineContainer, build_container
from domain.context_status import StatusSource
from domain.source_references import SourceReferenceRecord
from domain.ports.pot_resolution import (
    PotResolutionPort,
    RepoRef,
    ResolvedPot,
    ResolvedPotRepo,
)
from domain.ports.pot_source_listing import PotSourceListingPort
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


def _resolved_pot_from_context_graph_row(db: Session, row: ContextGraphPot) -> ResolvedPot:
    rrows = (
        db.query(ContextGraphPotRepository)
        .filter(ContextGraphPotRepository.pot_id == row.id)
        .order_by(ContextGraphPotRepository.created_at.asc())
        .all()
    )
    repos: list[ResolvedPotRepo] = []
    for r in rrows:
        repos.append(
            ResolvedPotRepo(
                pot_id=row.id,
                repo_id=r.id,
                provider=r.provider,
                provider_host=r.provider_host,
                repo_name=f"{r.owner}/{r.repo}",
                remote_url=r.remote_url,
                default_branch=r.default_branch,
                ready=True,
            )
        )
    display = (row.display_name or "").strip() or None
    name = display or (repos[0].repo_name if repos else row.id)
    return ResolvedPot(
        pot_id=row.id,
        name=name,
        repos=repos,
        ready=True,
    )


def _user_can_access_context_graph_pot(db: Session, user_id: str, pot_id: str) -> bool:
    if (
        db.query(ContextGraphPotMember)
        .filter(
            ContextGraphPotMember.pot_id == pot_id,
            ContextGraphPotMember.user_id == user_id,
        )
        .first()
    ):
        return True
    pot = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    return bool(pot is not None and pot.user_id == user_id)


class SqlalchemyPotResolution(PotResolutionPort):
    """Worker / session-wide resolver: ``context_graph_pots`` and attached repositories only."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        cg = (
            self._db.query(ContextGraphPot)
            .filter(ContextGraphPot.id == pot_id)
            .first()
        )
        if cg is None or cg.archived_at is not None:
            return None
        return _resolved_pot_from_context_graph_row(self._db, cg)

    def known_pot_ids(self) -> list[str]:
        cg = [
            r[0]
            for r in self._db.query(ContextGraphPot.id)
            .filter(ContextGraphPot.archived_at.is_(None))
            .all()
        ]
        return sorted(set(cg))

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        want = ref.repo_name.lower()
        full_name = func.lower(
            func.concat(
                ContextGraphPotRepository.owner,
                "/",
                ContextGraphPotRepository.repo,
            )
        )
        cg_repo = [
            r[0]
            for r in self._db.query(ContextGraphPotRepository.pot_id)
            .join(
                ContextGraphPot,
                ContextGraphPot.id == ContextGraphPotRepository.pot_id,
            )
            .filter(
                full_name == want,
                ContextGraphPot.archived_at.is_(None),
            )
            .all()
        ]
        return sorted(set(cg_repo))

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


class UserScopedContextGraphPotResolution(PotResolutionPort):
    """Resolve context-graph pots the caller may access (member or legacy pot owner row)."""

    def __init__(self, db: Session, user_id: str) -> None:
        self._db = db
        self._user_id = user_id

    def resolve_pot(self, pot_id: str) -> ResolvedPot | None:
        row = self._db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
        if row is None or row.archived_at is not None:
            return None
        if not _user_can_access_context_graph_pot(self._db, self._user_id, pot_id):
            return None
        return _resolved_pot_from_context_graph_row(self._db, row)

    def known_pot_ids(self) -> list[str]:
        cg = [
            r[0]
            for r in self._db.query(ContextGraphPotMember.pot_id)
            .filter(ContextGraphPotMember.user_id == self._user_id)
            .all()
        ]
        legacy = [
            r[0]
            for r in self._db.query(ContextGraphPot.id)
            .filter(
                ContextGraphPot.user_id == self._user_id,
                ContextGraphPot.archived_at.is_(None),
            )
            .all()
        ]
        return sorted(set(cg + legacy))

    def find_pots_for_repo(self, ref: RepoRef) -> list[str]:
        want = ref.repo_name.lower()
        full_name = func.lower(
            func.concat(
                ContextGraphPotRepository.owner,
                "/",
                ContextGraphPotRepository.repo,
            )
        )
        cg_repo_member = [
            r[0]
            for r in self._db.query(ContextGraphPotRepository.pot_id)
            .join(
                ContextGraphPotMember,
                ContextGraphPotMember.pot_id == ContextGraphPotRepository.pot_id,
            )
            .join(ContextGraphPot, ContextGraphPot.id == ContextGraphPotRepository.pot_id)
            .filter(
                ContextGraphPotMember.user_id == self._user_id,
                ContextGraphPot.archived_at.is_(None),
                full_name == want,
            )
            .all()
        ]
        cg_repo_owner = [
            r[0]
            for r in self._db.query(ContextGraphPotRepository.pot_id)
            .join(ContextGraphPot, ContextGraphPot.id == ContextGraphPotRepository.pot_id)
            .filter(
                ContextGraphPot.user_id == self._user_id,
                ContextGraphPot.archived_at.is_(None),
                full_name == want,
            )
            .all()
        ]
        return sorted(set(cg_repo_member + cg_repo_owner))

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


class SqlalchemyPotSourceListing(PotSourceListingPort):
    """Map ``ContextGraphPotSource`` rows to the engine's compact ``StatusSource`` view."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def list_pot_sources(self, pot_id: str) -> list[StatusSource]:
        rows = (
            self._db.query(ContextGraphPotSource)
            .filter(ContextGraphPotSource.pot_id == pot_id)
            .order_by(ContextGraphPotSource.created_at.asc())
            .all()
        )
        return [pot_source_row_to_status_source(r) for r in rows]


def pot_source_row_to_status_source(row: ContextGraphPotSource) -> StatusSource:
    provider_host: str | None = None
    scope_summary: str | None = None
    if row.scope_json:
        try:
            scope = json.loads(row.scope_json)
        except (TypeError, ValueError):
            scope = {}
        if isinstance(scope, dict):
            host_val = scope.get("provider_host")
            if isinstance(host_val, str) and host_val.strip():
                provider_host = host_val.strip()
            scope_summary = _scope_summary_for(row.source_kind, scope)
    return StatusSource(
        source_id=row.id,
        pot_id=row.pot_id,
        source_kind=row.source_kind,
        provider=row.provider,
        provider_host=provider_host,
        sync_enabled=bool(row.sync_enabled),
        sync_mode=row.sync_mode,
        last_sync_at=row.last_sync_at,
        last_success_at=None,
        last_error_at=None,
        last_error=row.last_error,
        last_verified_at=None,
        verification_state=None,
        scope_summary=scope_summary,
        health_score=row.health_score,
    )


def _scope_summary_for(source_kind: str, scope: dict) -> str | None:
    if source_kind == "repository":
        repo_name = scope.get("repo_name")
        if isinstance(repo_name, str) and repo_name:
            return repo_name
    if source_kind == "issue_tracker_team":
        name = scope.get("team_name") or scope.get("team_id")
        if isinstance(name, str) and name:
            return f"team:{name}"
    return None


def _resolve_repo_for_pot(db: Session) -> "callable":
    """Return a ``(pot_id, ref) -> repo_name | None`` closure over ``db``.

    The GitHub PR resolver needs the pot's primary GitHub repo to actually
    fetch the PR. If a pot is multi-repo, callers can stamp the desired repo
    on a ref via ``resolver_hint['repo_name']``. Otherwise we pick the first
    GitHub repository source attached to the pot.
    """

    def _resolver(pot_id: str, ref: SourceReferenceRecord) -> str | None:
        hint = ref.resolver_hint or {}
        repo_hint = hint.get("repo_name")
        if isinstance(repo_hint, str) and repo_hint:
            return repo_hint
        rows = (
            db.query(ContextGraphPotSource)
            .filter(
                ContextGraphPotSource.pot_id == pot_id,
                ContextGraphPotSource.provider == "github",
                ContextGraphPotSource.source_kind == "repository",
            )
            .order_by(ContextGraphPotSource.created_at.asc())
            .all()
        )
        for row in rows:
            if not row.scope_json:
                continue
            try:
                scope = json.loads(row.scope_json)
            except (TypeError, ValueError):
                continue
            if isinstance(scope, dict):
                repo_name = scope.get("repo_name")
                if isinstance(repo_name, str) and repo_name:
                    return repo_name
        return None

    return _resolver


def _build_source_resolver(db: Session, source_for_repo):
    """Compose the default host source-resolver chain.

    ``GitHubPullRequestResolver`` covers ``github:pr:*`` and legacy ``change``
    refs that embed PR numbers; ``DocumentationUriResolver`` covers docs
    refs with a stored summary/title (no live fetch by default). Adding a
    ``content_fetcher`` here later enables live doc snippets.
    """
    return CompositeSourceResolver(
        children=[
            GitHubPullRequestResolver(
                source_for_repo=source_for_repo,
                repo_resolver=_resolve_repo_for_pot(db),
            ),
            DocumentationUriResolver(),
        ]
    )


def build_container_for_session(db: Session) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    container = build_container(
        settings=PotpieContextEngineSettings(),
        pots=SqlalchemyPotResolution(db),
        source_for_repo=source_for_repo,
        reconciliation_agent=try_pydantic_deep_reconciliation_agent(),
        jobs=get_context_graph_job_queue(),
    )
    container.pot_source_listing = SqlalchemyPotSourceListing(db)
    resolver = _build_source_resolver(db, source_for_repo)
    container.source_resolver = resolver
    if container.resolution_service is not None:
        container.resolution_service.set_source_resolver(resolver)
    return container


def build_container_for_user_session(db: Session, user_id: str) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    container = build_container(
        settings=PotpieContextEngineSettings(),
        pots=UserScopedContextGraphPotResolution(db, user_id),
        source_for_repo=source_for_repo,
        reconciliation_agent=try_pydantic_deep_reconciliation_agent(),
        jobs=get_context_graph_job_queue(),
    )
    container.pot_source_listing = SqlalchemyPotSourceListing(db)
    resolver = _build_source_resolver(db, source_for_repo)
    container.source_resolver = resolver
    if container.resolution_service is not None:
        container.resolution_service.set_source_resolver(resolver)
    return container
