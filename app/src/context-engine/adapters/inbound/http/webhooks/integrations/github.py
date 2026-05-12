"""GitHub webhook → registry-dispatched ingestion (Phase 2).

The HTTP route here is intentionally generic: it forwards the raw payload
+ headers to the GitHub connector's ``normalize_webhook`` and submits the
returned :class:`ContextEvent` through the standard ingestion service.
All GitHub-specific parsing / signature checking lives in the connector;
this module is just transport glue.
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, HTTPException, Request

from adapters.outbound.postgres.session import make_session_factory
from adapters.inbound.http.deps import get_container_or_503
from domain.actor import Actor
from domain.ingestion_event_models import IngestionSubmissionRequest
from domain.ingestion_kinds import INGESTION_KIND_AGENT_RECONCILIATION

logger = logging.getLogger(__name__)

github_router = APIRouter()


@github_router.post("/github")
async def github_webhook(request: Request):
    body = await request.body()
    headers = dict(request.headers)

    container = get_container_or_503()
    if not container.settings.is_enabled():
        return {"processed": False, "reason": "context_graph_disabled"}

    connector = container.connectors.find_for_webhook("github")
    if connector is None:
        raise HTTPException(
            status_code=503,
            detail="github connector not registered",
        )

    try:
        event = connector.normalize_webhook(body, headers)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc) or "signature mismatch") from exc

    if event is None:
        return {"processed": False, "reason": "ignored_event"}

    repo_name = event.repo_name or ""
    if not repo_name:
        raise HTTPException(status_code=400, detail="missing repository in event")

    mapping_raw = os.getenv("CONTEXT_ENGINE_REPO_TO_POT", "").strip()
    if not mapping_raw:
        raise HTTPException(
            status_code=503,
            detail='CONTEXT_ENGINE_REPO_TO_POT env required, e.g. {"owner/repo":"pot-uuid"}',
        )
    repo_to_pot = json.loads(mapping_raw)
    pot_id = repo_to_pot.get(repo_name)
    if not pot_id:
        return {"processed": False, "reason": "no_pot_for_repo", "repo_name": repo_name}

    factory = make_session_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    delivery_id = (
        headers.get("X-GitHub-Delivery") or headers.get("x-github-delivery") or ""
    )
    sender_login = (event.payload.get("sender_login") or "").strip() or None
    actor = Actor(
        user_id=f"webhook:github:{delivery_id}" if delivery_id else "webhook:github:unknown",
        surface="webhook",
        client_name=f"github:{sender_login}" if sender_login else "github",
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
            actor=actor,
        )
        receipt = svc.submit(req)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    logger.info("github_webhook ingest receipt=%s", receipt)
    return {
        "processed": True,
        "event_id": receipt.event_id,
        "status": receipt.status,
        "duplicate": receipt.duplicate,
        "error": receipt.error,
    }
