import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.code_provider.github.github_webhook_service import GitHubWebhookService

router = APIRouter(prefix="/api/v1/github", tags=["GitHub Webhooks"])


@router.post("/webhook")
async def handle_github_webhook(
    request: Request,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Handle GitHub webhooks for repository visibility changes.
    
    Supported events:
    - 'public': Repository changed from private to public
    - 'repository': Repository events including privatization, publicization, deletion
    
    This endpoint automatically purges cache when repository visibility changes.
    """
    try:
        webhook_service = GitHubWebhookService(db)
        result = await webhook_service.process_webhook(request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook processing error: {str(e)}")


@router.get("/webhook/stats")
async def get_webhook_stats(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get webhook service statistics and status"""
    try:
        webhook_service = GitHubWebhookService(db)
        return await webhook_service.get_webhook_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting webhook stats: {str(e)}") 