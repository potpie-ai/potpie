import logging
from typing import Dict, Any

from fastapi import Request, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.code_provider.github.github_webhook_service import GitHubWebhookService
from app.modules.utils.APIRouter import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/github/webhook")
async def process_github_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """Process GitHub webhook events for cache management"""
    try:
        # Get webhook payload
        payload = await request.json()
        
        # Get event type from headers
        event_type = request.headers.get("X-GitHub-Event")
        
        if not event_type:
            logger.warning("No X-GitHub-Event header found")
            raise HTTPException(status_code=400, detail="Missing X-GitHub-Event header")
        
        webhook_service = GitHubWebhookService(db)
        
        # Process different event types
        if event_type == "repository":
            result = await webhook_service.process_repository_event(payload)
        elif event_type == "ping":
            result = await webhook_service.process_ping_event(payload)
        else:
            logger.info(f"Ignoring webhook event type: {event_type}")
            result = {"status": "ignored", "event_type": event_type}
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed") 