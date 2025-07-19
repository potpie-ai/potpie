import hashlib
import hmac
import json
import logging
import os
from typing import Dict, Any, Optional

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.code_provider.github.github_service import GithubService

logger = logging.getLogger(__name__)


class GitHubWebhookService:
    def __init__(self, db: Session):
        self.db = db
        self.github_service = GithubService(db)
        self.webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
        self.stats = {
            "webhooks_received": 0,
            "cache_purges": 0,
            "errors": 0
        }

    async def process_webhook(self, request: Request) -> Dict[str, Any]:
        """Process incoming GitHub webhook and purge cache if needed"""
        try:
            self.stats["webhooks_received"] += 1
            
            # Get webhook payload
            payload = await request.body()
            headers = dict(request.headers)
            
            # Verify webhook signature if secret is configured
            if self.webhook_secret:
                if not self._verify_webhook_signature(payload, headers):
                    logger.warning("Webhook signature verification failed")
                    raise HTTPException(status_code=401, detail="Invalid webhook signature")
            
            # Parse webhook payload
            try:
                webhook_data = json.loads(payload.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in webhook payload: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON payload")
            
            # Get event type from headers
            event_type = headers.get('x-github-event', '')
            logger.info(f"Received webhook event: {event_type} for repository: {webhook_data.get('repository', {}).get('full_name', 'unknown')}")
            
            # Process based on event type
            if event_type == 'public':
                return await self._handle_public_event(webhook_data)
            elif event_type == 'repository':
                return await self._handle_repository_event(webhook_data)
            else:
                logger.info(f"Ignoring webhook event type: {event_type}")
                return {"status": "ignored", "event_type": event_type}
                
        except HTTPException:
            raise
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error processing webhook: {e}")
            raise HTTPException(status_code=500, detail=f"Webhook processing error: {str(e)}")

    async def _handle_public_event(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repository publicization event"""
        try:
            repository = webhook_data.get('repository', {})
            repo_name = repository.get('full_name')
            
            if not repo_name:
                logger.warning("No repository name found in public event")
                return {"status": "error", "message": "No repository name found"}
            
            # Clear cache for this repository
            await self._clear_repository_cache(repo_name)
            
            logger.info(f"Cache cleared for publicized repository: {repo_name}")
            return {
                "status": "success",
                "action": "cache_cleared",
                "repository": repo_name,
                "event": "public"
            }
            
        except Exception as e:
            logger.error(f"Error handling public event: {e}")
            return {"status": "error", "message": str(e)}

    async def _handle_repository_event(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle repository events (privatization, deletion, etc.)"""
        try:
            repository = webhook_data.get('repository', {})
            repo_name = repository.get('full_name')
            action = webhook_data.get('action')
            
            if not repo_name:
                logger.warning("No repository name found in repository event")
                return {"status": "error", "message": "No repository name found"}
            
            # Clear cache for repository events that affect visibility
            if action in ['privatized', 'deleted', 'transferred']:
                await self._clear_repository_cache(repo_name)
                logger.info(f"Cache cleared for repository event: {action} - {repo_name}")
                return {
                    "status": "success",
                    "action": "cache_cleared",
                    "repository": repo_name,
                    "event": "repository",
                    "repository_action": action
                }
            else:
                logger.info(f"Ignoring repository event action: {action} for {repo_name}")
                return {
                    "status": "ignored",
                    "action": action,
                    "repository": repo_name,
                    "event": "repository"
                }
                
        except Exception as e:
            logger.error(f"Error handling repository event: {e}")
            return {"status": "error", "message": str(e)}

    async def _clear_repository_cache(self, repo_name: str):
        """Clear cache for a specific repository"""
        try:
            await self.github_service.clear_repository_cache(repo_name)
            self.stats["cache_purges"] += 1
            logger.info(f"Cache cleared for repository: {repo_name}")
        except Exception as e:
            logger.error(f"Error clearing cache for {repo_name}: {e}")
            raise

    def _verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """Verify GitHub webhook signature using HMAC SHA256"""
        try:
            signature = headers.get('x-hub-signature-256', '')
            if not signature or not signature.startswith('sha256='):
                return False
            
            expected_signature = signature[7:]  # Remove 'sha256=' prefix
            
            # Calculate signature
            calculated_signature = hmac.new(
                self.webhook_secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, calculated_signature)
            
        except Exception as e:
            logger.error(f"Error verifying webhook signature: {e}")
            return False

    async def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook service statistics"""
        return {
            "webhooks_received": self.stats["webhooks_received"],
            "cache_purges": self.stats["cache_purges"],
            "errors": self.stats["errors"],
            "webhook_secret_configured": bool(self.webhook_secret)
        } 