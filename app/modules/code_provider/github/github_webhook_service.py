import logging
import json
from typing import Dict, Any

from sqlalchemy.orm import Session
from app.modules.code_provider.github.github_service import GithubService

logger = logging.getLogger(__name__)


class GitHubWebhookService:
    def __init__(self, db: Session):
        self.db = db
        self.github_service = GithubService(db)
    
    async def process_repository_event(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process repository webhook events and clear cache when needed"""
        try:
            action = webhook_payload.get("action")
            repository = webhook_payload.get("repository", {})
            repo_name = repository.get("full_name")
            
            if not repo_name:
                logger.warning("No repository name found in webhook payload")
                return {"status": "ignored", "reason": "no_repo_name"}
            
            # Events that require cache invalidation
            cache_invalidation_events = ["publicized", "privatized", "deleted"]
            
            if action in cache_invalidation_events:
                await self.github_service.clear_repository_cache(repo_name)
                logger.info(f"Cache cleared for repository {repo_name} due to action: {action}")
                return {
                    "status": "processed",
                    "action": action,
                    "repo_name": repo_name,
                    "message": f"Cache cleared for {repo_name}"
                }
            else:
                logger.info(f"Action '{action}' does not require cache invalidation for {repo_name}")
                return {
                    "status": "ignored",
                    "action": action,
                    "repo_name": repo_name,
                    "message": f"No cache invalidation needed for action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Error processing webhook event: {e}")
            return {"status": "error", "error": str(e)}
    
    async def process_ping_event(self, webhook_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process ping webhook events"""
        return {"status": "pong", "message": "Webhook endpoint is active"} 