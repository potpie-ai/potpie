"""GitLab webhook → registry-dispatched ingestion.

The HTTP route here is intentionally generic: it forwards the raw payload
+ headers to the GitLab connector's ``normalize_webhook`` and submits the
returned :class:`ContextEvent` through the standard ingestion service.
All GitLab-specific parsing / secret-token checking lives in the
connector; this module is just transport glue — the same shape as
``integrations/github.py``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from potpie_context_engine.adapters.inbound.http.deps import get_container_or_503
from potpie_context_engine.adapters.outbound.postgres.session import (
    make_session_factory,
)
from potpie_context_core.actor import Actor
from potpie_context_core.ports.pot_resolution import RepoRef
from potpie_context_engine.domain.ingestion_event_models import (
    IngestionSubmissionRequest,
)
from potpie_context_engine.domain.ingestion_kinds import (
    INGESTION_KIND_AGENT_RECONCILIATION,
)

logger = logging.getLogger(__name__)

gitlab_router = APIRouter()


@gitlab_router.post("/gitlab")
async def gitlab_webhook(request: Request):
    body = await request.body()
    headers = {k.lower(): v for k, v in request.headers.items()}

    container = get_container_or_503()
    if not container.settings.is_enabled():
        return {"processed": False, "reason": "context_graph_disabled"}

    connector = container.connectors.find_for_webhook("gitlab")
    if connector is None:
        raise HTTPException(
            status_code=503,
            detail="gitlab connector not registered",
        )

    try:
        event = connector.normalize_webhook(body, headers)
    except PermissionError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc) or "secret token mismatch"
        ) from exc

    if event is None:
        return {"processed": False, "reason": "ignored_event"}

    project_path = event.repo_name or ""
    if not project_path or "/" not in project_path:
        raise HTTPException(status_code=400, detail="missing project in event")

    # A GitLab project may live under nested groups (``group/sub/project``);
    # the project name is the last segment and everything before it is the
    # owning namespace.
    owner, repo = project_path.rsplit("/", 1)
    ref = RepoRef(
        provider=event.provider or "gitlab",
        provider_host=event.provider_host or "gitlab.com",
        owner=owner,
        repo=repo,
    )
    pot_ids = container.pots.find_pots_for_repo(ref)
    if not pot_ids:
        return {
            "processed": False,
            "reason": "no_pot_for_repo",
            "repo_name": project_path,
        }
    if len(pot_ids) > 1:
        logger.warning(
            "gitlab_webhook: %s pots match %s; routing to first (%s)",
            len(pot_ids),
            project_path,
            pot_ids[0],
        )
    pot_id = pot_ids[0]

    factory = make_session_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    delivery_id = headers.get("x-gitlab-event-uuid") or ""
    sender_login = (event.payload.get("sender_login") or "").strip() or None
    actor = Actor(
        user_id=f"webhook:gitlab:{delivery_id}"
        if delivery_id
        else "webhook:gitlab:unknown",
        surface="webhook",
        client_name=f"gitlab:{sender_login}" if sender_login else "gitlab",
        auth_method="webhook_signature",
    )

    session = factory()
    try:
        svc = container.ingestion_submission(session)
        req = IngestionSubmissionRequest(
            pot_id=str(pot_id),
            ingestion_kind=INGESTION_KIND_AGENT_RECONCILIATION,
            source_channel="webhook",
            source_system=event.source_system,
            event_type=event.event_type,
            action=event.action,
            payload=dict(event.payload),
            source_id=event.source_id,
            source_event_id=event.source_event_id,
            repo_name=event.repo_name,
            provider=event.provider,
            provider_host=event.provider_host,
            actor=actor,
        )
        receipt = svc.submit(req)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    logger.info("gitlab_webhook ingest receipt=%s", receipt)
    return {
        "processed": True,
        "event_id": receipt.event_id,
        "status": receipt.status,
        "duplicate": receipt.duplicate,
        "error": receipt.error,
    }
