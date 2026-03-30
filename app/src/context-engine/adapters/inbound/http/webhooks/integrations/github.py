"""GitHub webhook: merged PR → ingest (standalone service)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os

from fastapi import APIRouter, Header, HTTPException, Request

from adapters.inbound.http.deps import get_container_or_503
from adapters.outbound.postgres.session import make_session_factory
from application.use_cases.ingest_single_pr import ingest_single_pull_request

logger = logging.getLogger(__name__)

github_router = APIRouter()


def _verify_signature(body: bytes, signature: str | None, secret: str) -> bool:
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected}")


@github_router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None, alias="X-Hub-Signature-256"),
):
    body = await request.body()
    secret = (os.getenv("GITHUB_WEBHOOK_SECRET") or "").strip()
    if secret and not _verify_signature(body, x_hub_signature_256, secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

    event = request.headers.get("X-GitHub-Event", "")
    if event != "pull_request":
        return {"processed": False, "reason": "ignored_event", "event": event}

    action = payload.get("action")
    pr = payload.get("pull_request") or {}
    merged = pr.get("merged")
    if action != "closed" or not merged:
        return {"processed": False, "reason": "not_merged_pr"}

    repo = (payload.get("repository") or {}).get("full_name")
    pr_number = pr.get("number")
    if not repo or pr_number is None:
        raise HTTPException(status_code=400, detail="missing repository or PR number")

    mapping_raw = os.getenv("CONTEXT_ENGINE_REPO_TO_POT", "").strip()
    if not mapping_raw:
        raise HTTPException(
            status_code=503,
            detail="CONTEXT_ENGINE_REPO_TO_POT env required, e.g. {\"owner/repo\":\"pot-uuid\"}",
        )
    repo_to_pot = json.loads(mapping_raw)
    pot_id = repo_to_pot.get(repo)
    if not pot_id:
        return {"processed": False, "reason": "no_pot_for_repo", "repo_name": repo}

    container = get_container_or_503()
    if not container.settings.is_enabled():
        return {"processed": False, "reason": "context_graph_disabled"}

    factory = make_session_factory()
    if factory is None:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    session = factory()
    try:
        result = ingest_single_pull_request(
            settings=container.settings,
            pots=container.pots,
            source=container.source_for_repo(repo),
            ledger=container.ledger(session),
            episodic=container.episodic,
            structural=container.structural,
            pot_id=str(pot_id),
            pr_number=int(pr_number),
            is_live_bridge=True,
        )
    finally:
        session.close()

    logger.info("github_webhook ingest result=%s", result)
    return {"processed": True, "result": result}
