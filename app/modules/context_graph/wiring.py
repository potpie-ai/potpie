"""Potpie-specific wiring for context-engine (ports → host services)."""

from __future__ import annotations

import json
import logging
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
from adapters.outbound.connectors._bench_stubs import (
    AlertingStubConnector,
    DeployStubConnector,
    RepoDocsStubConnector,
    SlackStubConnector,
)
from adapters.outbound.connectors.github import GitHubConnector
from adapters.outbound.connectors.linear.connector import LinearConnector
from adapters.outbound.connectors.notion import NotionConnector
from application.services.source_connector_registry import SourceConnectorRegistry
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

logger = logging.getLogger(__name__)


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

    def graph_db_backend(self) -> str:
        return (os.getenv("GRAPH_DB_BACKEND") or "neo4j").strip().lower() or "neo4j"

    def falkordb_url(self) -> str | None:
        v = (os.getenv("CONTEXT_ENGINE_FALKORDB_URL") or os.getenv("FALKORDB_URL") or "").strip()
        return v or None

    def falkordb_graph_name(self) -> str:
        v = (
            os.getenv("CONTEXT_ENGINE_FALKORDB_GRAPH_NAME")
            or os.getenv("FALKORDB_GRAPH_NAME")
            or ""
        ).strip()
        return v or "context_graph"

    def falkordb_mode(self) -> str:
        v = (os.getenv("CONTEXT_ENGINE_FALKORDB_MODE") or os.getenv("FALKORDB_MODE") or "").strip().lower()
        return v or "server"

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

    # Read by DefaultPolicyAdapter's tenant-boundary contract: this resolver
    # returns None for pots the caller cannot access, so per-actor pot
    # authorization is enforced here. Do NOT set this on the wide
    # (worker/system) resolvers.
    actor_scoped = True

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


def _build_connector_registry(db: Session, source_for_repo) -> SourceConnectorRegistry:
    """Compose the default host connector registry.

    Phase 2 replaced the standalone ``CompositeSourceResolver`` chain with a
    :class:`SourceConnectorRegistry`. The GitHub connector wraps the same
    ``GitHubPullRequestResolver`` internally and resolves a pot's primary
    GitHub repo via the closure returned by :func:`_resolve_repo_for_pot`
    (which also honours ``resolver_hint['repo_name']`` for multi-repo pots).

    The Linear connector resolves its access token per call via
    :class:`ContextEngineLinearFetcher`, which walks
    ``pot_id → project_sources → integrations`` and decrypts the stored
    OAuth token; this keeps the connector multi-tenant without coupling
    it to any single workspace.
    """
    from integrations.adapters.outbound.linear.context_engine_fetcher import (
        ContextEngineLinearFetcher,
    )

    registry = SourceConnectorRegistry()
    registry.register(
        # Same rationale as the LinearConnector below: in the host the
        # GitHub signature is verified at the HTTP ingress
        # (`integrations_router.py /github/webhook`, fail-closed) and the
        # repo→pot routing path does not invoke the connector's
        # ``normalize_webhook``. Connector-level enforcement is reserved
        # for the standalone container path (``bootstrap/container.py``),
        # where the connector itself is the ingress.
        GitHubConnector(
            source_for_repo=source_for_repo,
            repo_resolver=_resolve_repo_for_pot(db),
            allow_unsigned=True,
        )
    )
    registry.register(
        # The host verifies the Linear signature at the HTTP ingress
        # (`integrations_router.py /linear/webhook`, fail-closed over the
        # raw body). By the time the connector runs it is post-verification
        # and behind the async event bus — it has neither the raw body nor
        # the `Linear-Signature` header, so it must NOT re-enforce here
        # (doing so would falsely reject every event). Signature
        # enforcement at the connector is only for the standalone path,
        # where the connector itself is the ingress.
        LinearConnector(
            fetcher=ContextEngineLinearFetcher(db),
            allow_unsigned=True,
        )
    )
    # Passive bench-stub connectors — register them in the host registry
    # too so the benchmark probe sees the full set of source kinds
    # (notion/slack/repo_docs/alerting/deploy). They advertise no fetch
    # capability so production traffic without a real reader still fails
    # closed; they exist solely as ``source_system`` markers.
    registry.register(NotionConnector())
    registry.register(SlackStubConnector())
    registry.register(RepoDocsStubConnector())
    registry.register(AlertingStubConnector())
    registry.register(DeployStubConnector())
    return registry


from typing import Callable, Optional, Tuple

PotTokenResolver = Callable[
    [Optional[str], Optional[str]],
    Tuple[Optional[str], Optional[str]],
]
"""``(user_id, repo_name) -> (token, kind)`` — see :func:`_default_pot_token_resolver`."""


def _default_pot_token_resolver(user_id: str | None, repo_name: str | None):
    """Default chain: contextvar → GitHub App → user OAuth → env.

    Returns ``(token, kind)`` so callers / telemetry can tell which branch
    produced the token. Wraps :func:`_resolve_auth` so the existing chain
    (which already prefers App-installation tokens, surviving the user
    leaving the pot) stays the source of truth.
    """
    from app.modules.intelligence.tools.sandbox.client import _resolve_auth

    resolved = _resolve_auth(user_id, repo_name)
    return resolved.token, resolved.kind


_POT_TOKEN_RESOLVER = _default_pot_token_resolver


def set_pot_token_resolver(resolver) -> None:
    """Override the per-attachment token resolver process-wide.

    Lets alternate hosts (tests, on-prem deployments without GitHub-App
    install) plug in their own resolution. Pass ``None`` to restore the
    default chain.
    """
    global _POT_TOKEN_RESOLVER
    _POT_TOKEN_RESOLVER = resolver or _default_pot_token_resolver


def _build_pot_sandbox_resolver(db: Session):
    """Resolve ``pot_id -> PotSandboxConfig`` (every repo attached to the pot).

    Closes over the request-scoped DB session. Returns ``None`` when the pot
    has no attached repos — ``build_sandbox_tools`` then silently skips
    sandbox tools for that batch.

    Tenant choice: the first attacher (oldest ``created_at`` row) owns the
    pot's sandbox container, so per-pot Daytona/Docker state collapses into
    one ``(user_id, pot_id)`` key regardless of who attached subsequent
    repos. Per-repo clone tokens flow through the pluggable
    :func:`set_pot_token_resolver` so GitHub-App installation tokens win
    over user OAuth (and the choice is observable in telemetry).
    """
    from adapters.outbound.agent_tools.sandbox import (
        PotSandboxConfig,
        RepoAttachment,
    )

    def _resolver(pot_id: str):
        rows = (
            db.query(ContextGraphPotRepository)
            .filter(ContextGraphPotRepository.pot_id == pot_id)
            .order_by(ContextGraphPotRepository.created_at.asc())
            .all()
        )
        if not rows:
            return None
        first = rows[0]
        repos: list[RepoAttachment] = []
        for r in rows:
            full = f"{r.owner}/{r.repo}"
            try:
                token, kind = _POT_TOKEN_RESOLVER(r.added_by_user_id, full)
            except Exception:
                token, kind = None, "error"
            repos.append(
                RepoAttachment(
                    owner=r.owner,
                    repo=r.repo,
                    default_branch=r.default_branch or "main",
                    repo_url=r.remote_url,
                    auth_token=token,
                    auth_kind=kind,
                )
            )
        return PotSandboxConfig(
            user_id=first.added_by_user_id,
            pot_id=pot_id,
            provider_host=first.provider_host,
            repos=repos,
        )

    return _resolver


async def _sandbox_client_factory():
    """Async wrapper that hands the agent the process-wide ``SandboxClient``."""
    from app.modules.intelligence.tools.sandbox.client import get_sandbox_client

    return get_sandbox_client()


def _sandbox_tools_disabled() -> bool:
    return (os.getenv("CONTEXT_ENGINE_DISABLE_SANDBOX_TOOLS") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _build_pot_user_resolver(db: Session):
    """Resolve ``pot_id -> user_id`` (the account the pot is configured under).

    Used to scope host-coupled tools (web search / page extraction) to the
    pot owner's provider keys and quota in the worker path, which has no
    request user. Prefers the explicit pot owner row, falling back to the
    oldest repo attacher so a legacy pot with no ``user_id`` still resolves.
    """

    def _resolver(pot_id: str | None) -> str | None:
        if not pot_id:
            return None
        pot = (
            db.query(ContextGraphPot)
            .filter(ContextGraphPot.id == pot_id)
            .first()
        )
        if pot is not None and getattr(pot, "user_id", None):
            return pot.user_id
        repo = (
            db.query(ContextGraphPotRepository)
            .filter(ContextGraphPotRepository.pot_id == pot_id)
            .order_by(ContextGraphPotRepository.created_at.asc())
            .first()
        )
        if repo is not None and getattr(repo, "added_by_user_id", None):
            return repo.added_by_user_id
        return None

    return _resolver


def _attach_agent_tools(agent, db: Session, *, source_for_repo) -> None:
    """Best-effort: wire every extra tool surface onto the reconciliation agent.

    ``add_extra_tools`` *replaces* the agent's builder list, so all surfaces
    (sandbox + GitHub + Linear + web) must be assembled and registered in a
    single call. Each surface is added independently and any failing import
    is skipped so a missing optional dependency degrades that surface only —
    the agent still runs with whatever tools resolved.

    Every surface stays scoped to the account the pot is configured under:
    sandbox/GitHub via the per-repo token chain, Linear via the pot's
    connected OAuth integration, web via the pot owner's provider keys.
    """
    if agent is None:
        return
    attach = getattr(agent, "add_extra_tools", None)
    if attach is None:
        return

    builders: list = []

    if not _sandbox_tools_disabled():
        try:
            from adapters.outbound.agent_tools.sandbox import build_sandbox_tools

            builders.append(
                build_sandbox_tools(
                    client_factory=_sandbox_client_factory,
                    pot_resolver=_build_pot_sandbox_resolver(db),
                )
            )
        except Exception:
            logger.exception("failed to build sandbox tool surface")

    try:
        from adapters.outbound.connectors.github.agent_tools import (
            build_github_tools,
        )

        def _allowed_repos_for_pot(pot_id: str) -> set[str]:
            """``owner/repo`` (lowercased) attached to the pot — the only
            repos a reconciliation agent may read via the GitHub tools.
            Binds agent-supplied ``repo_name`` to the pot's tenancy so a
            prompt-injected agent cannot pull a foreign private repo
            through the shared org credential (security review C-5)."""
            rows = (
                db.query(ContextGraphPotRepository)
                .filter(ContextGraphPotRepository.pot_id == pot_id)
                .all()
            )
            return {
                f"{r.owner}/{r.repo}".strip().lower()
                for r in rows
                if r.owner and r.repo
            }

        builders.append(
            build_github_tools(
                source_for_repo,
                allowed_repos_for_pot=_allowed_repos_for_pot,
            )
        )
    except Exception:
        logger.exception("failed to build github tool surface")

    try:
        from adapters.outbound.connectors.linear.agent_tools import (
            build_linear_tools,
        )
        from integrations.adapters.outbound.linear.context_engine_fetcher import (
            ContextEngineLinearFetcher,
        )

        builders.append(build_linear_tools(ContextEngineLinearFetcher(db)))
    except Exception:
        logger.exception("failed to build linear tool surface")

    try:
        from app.modules.context_graph.agent_web_tools import build_web_tools

        builders.append(
            build_web_tools(db=db, user_resolver=_build_pot_user_resolver(db))
        )
    except Exception:
        logger.exception("failed to build web tool surface")

    if builders:
        attach(builders)


def build_container_for_session(db: Session) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    agent = try_pydantic_deep_reconciliation_agent()
    _attach_agent_tools(agent, db, source_for_repo=source_for_repo)
    container = build_container(
        settings=PotpieContextEngineSettings(),
        pots=SqlalchemyPotResolution(db),
        connectors=_build_connector_registry(db, source_for_repo),
        reconciliation_agent=agent,
        jobs=get_context_graph_job_queue(),
    )
    container.pot_source_listing = SqlalchemyPotSourceListing(db)
    return container


def build_container_for_user_session(db: Session, user_id: str) -> ContextEngineContainer:
    def source_for_repo(repo_name: str):
        provider = CodeProviderFactory.create_provider_with_fallback(repo_name)
        return CodeProviderSourceControl(provider)

    agent = try_pydantic_deep_reconciliation_agent()
    _attach_agent_tools(agent, db, source_for_repo=source_for_repo)
    container = build_container(
        settings=PotpieContextEngineSettings(),
        pots=UserScopedContextGraphPotResolution(db, user_id),
        connectors=_build_connector_registry(db, source_for_repo),
        reconciliation_agent=agent,
        jobs=get_context_graph_job_queue(),
    )
    container.pot_source_listing = SqlalchemyPotSourceListing(db)
    return container
