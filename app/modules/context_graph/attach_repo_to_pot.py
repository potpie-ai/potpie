"""``attach_repo_to_pot`` — single entry point used by HTTP / MCP / CLI.

Wraps three side effects in one idempotent verb so every caller stays in sync:

  1. Upsert the ``context_graph_pot_repositories`` row.
  2. Mirror the repo into ``context_graph_pot_sources`` for source-picker UX.
  3. Submit the ``repository.added`` ingestion event so the reconciliation
     agent walks the repo and seeds the graph.

Idempotent on ``(pot_id, provider, provider_host, owner, repo)`` for a
*live* attachment — repeat calls keep the row, re-mirror the source (cheap),
and do **not** re-emit the bootstrap event. The one exception is a repo that
was **deleted** (its source mirror gone but the repo row orphaned): re-adding
it is treated as a fresh attach so ingestion re-queues — idempotency must not
suppress re-queue for deleted repos. Callers that want a strict "already
attached" 409 can check ``AttachRepoResult.already_attached`` and raise their
own error (deleted-then-re-added repos report ``already_attached=False``).
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
from app.modules.context_graph.context_graph_pot_source_model import (
    ContextGraphPotSource,
)
from app.modules.context_graph.pot_sources_service import (
    mirror_repository_into_sources,
    repository_source_exists,
)

logger = logging.getLogger(__name__)


class UnknownPotError(LookupError):
    """Raised when ``pot_id`` does not match a row in ``context_graph_pots``."""


@dataclass(frozen=True, slots=True)
class AttachRepoResult:
    repository_id: str
    source_id: str
    already_attached: bool
    bootstrap_event_id: str | None
    repository: ContextGraphPotRepository
    source: ContextGraphPotSource


def _allowed_provider_hosts() -> set[str]:
    """Hosts a repo may be attached under. ``github.com`` plus any
    GitHub-Enterprise hosts the operator configured via
    ``CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS`` (comma-separated)."""
    extra = os.getenv("CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS", "")
    hosts = {"github.com"}
    hosts.update(h.strip().lower() for h in extra.split(",") if h.strip())
    return hosts


def attach_repo_to_pot(
    db: Session,
    *,
    pot_id: str,
    provider: str,
    provider_host: str,
    owner: str,
    repo: str,
    external_repo_id: str | None,
    remote_url: str | None,
    default_branch: str | None,
    submitted_by_user_id: str,
) -> AttachRepoResult:
    pot_row = db.query(ContextGraphPot).filter(ContextGraphPot.id == pot_id).first()
    if pot_row is None:
        raise UnknownPotError(f"Unknown pot_id: {pot_id}")

    owner = owner.strip()
    repo = repo.strip()
    provider = provider.strip()
    provider_host = provider_host.strip()
    if not owner or not repo:
        raise ValueError("owner and repo required")
    # SSRF / token-exfil guard: a pot owner could otherwise register an
    # internal ``provider_host`` and make the sandbox clone/fetch target it
    # carrying the injected auth token (security review M-2).
    if provider_host.lower() not in _allowed_provider_hosts():
        raise ValueError(
            f"provider_host not allowed: {provider_host!r} "
            "(set CONTEXT_ENGINE_ALLOWED_PROVIDER_HOSTS to permit a "
            "GitHub Enterprise host)"
        )

    existing = (
        db.query(ContextGraphPotRepository)
        .filter(
            ContextGraphPotRepository.pot_id == pot_id,
            ContextGraphPotRepository.provider == provider,
            ContextGraphPotRepository.provider_host == provider_host,
            ContextGraphPotRepository.owner == owner,
            ContextGraphPotRepository.repo == repo,
        )
        .first()
    )
    if existing is not None:
        # A surviving repo row with no mirrored source means the repo was
        # deleted via the Sources tab: ``delete_pot_source`` always drops
        # the source row but only best-effort drops the repo row (it
        # orphans it when it can't re-derive owner/repo from scope_json).
        # Re-adding a *deleted* repo must re-queue ingestion — idempotency
        # applies to live attachments, not to deleted repos.
        was_deleted = not repository_source_exists(db, existing)
        source = mirror_repository_into_sources(
            db, existing, added_by_user_id=submitted_by_user_id
        )
        db.commit()
        db.refresh(source)
        if not was_deleted:
            return AttachRepoResult(
                repository_id=existing.id,
                source_id=source.id,
                already_attached=True,
                bootstrap_event_id=None,
                repository=existing,
                source=source,
            )
        # Deleted-then-re-added: behave like a fresh attach. Re-warm the
        # sandbox clone and re-emit ``repository.added`` so the
        # reconciliation agent re-walks the repo. Stable entity keys make
        # this converge against existing Graphiti data, so no graph
        # teardown is needed. ``already_attached=False`` so the strict-409
        # callers (``add_pot_repository``) let the re-add through.
        _dispatch_prewarm(existing, submitted_by_user_id=submitted_by_user_id)
        bootstrap_event_id = _emit_bootstrap_event(
            db,
            pot_id=pot_id,
            repo_row=existing,
            submitted_by_user_id=submitted_by_user_id,
        )
        return AttachRepoResult(
            repository_id=existing.id,
            source_id=source.id,
            already_attached=False,
            bootstrap_event_id=bootstrap_event_id,
            repository=existing,
            source=source,
        )

    row = ContextGraphPotRepository(
        id=str(uuid.uuid4()),
        pot_id=pot_id,
        provider=provider,
        provider_host=provider_host,
        owner=owner,
        repo=repo,
        external_repo_id=(external_repo_id or "").strip() or None,
        remote_url=(remote_url or "").strip() or None,
        default_branch=(default_branch or "").strip() or None,
        added_by_user_id=submitted_by_user_id,
    )
    db.add(row)
    db.flush()

    if not (pot_row.primary_repo_name or "").strip():
        pot_row.primary_repo_name = f"{owner}/{repo}"

    source = mirror_repository_into_sources(
        db, row, added_by_user_id=submitted_by_user_id
    )
    db.commit()
    db.refresh(row)
    db.refresh(source)

    _dispatch_prewarm(row, submitted_by_user_id=submitted_by_user_id)

    bootstrap_event_id = _emit_bootstrap_event(
        db,
        pot_id=pot_id,
        repo_row=row,
        submitted_by_user_id=submitted_by_user_id,
    )
    return AttachRepoResult(
        repository_id=row.id,
        source_id=source.id,
        already_attached=False,
        bootstrap_event_id=bootstrap_event_id,
        repository=row,
        source=source,
    )


def _dispatch_prewarm(
    repo_row: ContextGraphPotRepository,
    *,
    submitted_by_user_id: str,
) -> None:
    """Best-effort: kick off the bare clone before the agent's first batch.

    Soft-fails when the prewarm module isn't importable (CLI / minimal test
    contexts) so a missing piece in the host doesn't block attach.
    """
    try:
        from app.modules.context_graph.pot_sandbox_provisioning import (
            dispatch_pot_repo_prewarm,
        )
    except Exception:
        return
    dispatch_pot_repo_prewarm(
        user_id=submitted_by_user_id,
        pot_id=repo_row.pot_id,
        owner=repo_row.owner,
        repo=repo_row.repo,
        default_branch=repo_row.default_branch,
        repo_url=repo_row.remote_url,
    )


def _emit_bootstrap_event(
    db: Session,
    *,
    pot_id: str,
    repo_row: ContextGraphPotRepository,
    submitted_by_user_id: str,
) -> str | None:
    """Submit ``repository.added`` so the agent seeds the graph.

    Best-effort: returns ``None`` (and logs) if the context-engine isn't
    importable, the container can't be built, or the submit call raises. The
    repo row + source mirror are already committed by the caller, so the
    attach itself stays successful.
    """
    try:
        from app.modules.context_graph.wiring import (
            build_container_for_user_session,
        )
        from domain.ingestion_event_models import IngestionSubmissionRequest
        from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION
    except Exception:
        logger.exception(
            "attach_repo_to_pot: context-engine import failed; skipping bootstrap event"
        )
        return None

    try:
        container = build_container_for_user_session(db, submitted_by_user_id)
    except Exception:
        logger.exception(
            "attach_repo_to_pot: container build failed for pot=%s", pot_id
        )
        return None

    repo_full = f"{repo_row.owner}/{repo_row.repo}"
    request = IngestionSubmissionRequest(
        pot_id=pot_id,
        ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
        source_channel="repo_attach",
        source_system=repo_row.provider or "github",
        event_type="repository",
        action="added",
        source_id=f"repo_added:{repo_full}",
        provider=repo_row.provider,
        provider_host=repo_row.provider_host,
        repo_name=repo_full,
        payload={
            "owner": repo_row.owner,
            "repo": repo_row.repo,
            "default_branch": repo_row.default_branch,
            "remote_url": repo_row.remote_url,
            "external_repo_id": repo_row.external_repo_id,
            "submitted_by_user_id": submitted_by_user_id,
        },
    )
    try:
        receipt = container.ingestion_submission(db).submit(
            request, sync=False, wait=False
        )
    except Exception:
        logger.exception(
            "attach_repo_to_pot: submit failed for pot=%s repo=%s",
            pot_id,
            repo_full,
        )
        return None
    logger.info(
        "attach_repo_to_pot: enqueued event=%s pot=%s repo=%s",
        receipt.event_id,
        pot_id,
        repo_full,
    )
    return receipt.event_id
