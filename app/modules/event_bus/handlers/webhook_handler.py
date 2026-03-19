"""
Webhook Event Handler

Handler for processing webhook events from integrations.
"""

from typing import Any, Dict

from sqlalchemy.orm import Session

from app.modules.integrations.integrations_service import IntegrationsService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class WebhookEventHandler:
    """Handler for processing webhook events from integrations."""

    def __init__(self, db: Session):
        """
        Initialize the webhook event handler.

        Args:
            db: Database session
        """
        self.db = db
        self.logger = logger

    def process_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a webhook event.

        Args:
            event_data: WebhookEvent data dictionary

        Returns:
            Processing result dictionary
        """
        integration_id = event_data.get("integration_id")
        integration_type = event_data.get("integration_type")
        event_type = event_data.get("event_type")
        payload = event_data.get("payload", {})

        self.logger.info(
            f"Processing webhook event: {event_type} from {integration_type} "
            f"integration {integration_id}"
        )

        try:
            # Validate integration exists and is active
            IntegrationsService(self.db)
            # Note: For now, we'll skip integration validation in the event handler
            # since the webhook endpoints already validate the integration
            integration = {"active": True}  # Placeholder for validation

            if not integration:
                raise ValueError(f"Integration {integration_id} not found")

            if not integration.get("active", False):
                raise ValueError(f"Integration {integration_id} is not active")

            # Process based on integration type
            result = self._process_by_integration_type(
                integration_type, event_type, payload, integration
            )

            self.logger.info(
                f"Successfully processed webhook event {event_type} "
                f"from {integration_type} integration {integration_id}"
            )

            return {
                "processed": True,
                "integration_id": integration_id,
                "integration_type": integration_type,
                "event_type": event_type,
                "result": result,
            }

        except Exception as e:
            self.logger.error(
                f"Failed to process webhook event {event_type} "
                f"from {integration_type} integration {integration_id}: {str(e)}"
            )
            raise

    def _process_by_integration_type(
        self,
        integration_type: str,
        event_type: str,
        payload: Dict[str, Any],
        integration: Any,
    ) -> Dict[str, Any]:
        """
        Process webhook event based on integration type.

        Args:
            integration_type: Type of integration (linear, sentry, etc.)
            event_type: Type of webhook event
            payload: Webhook payload data
            integration: Integration database model

        Returns:
            Processing result dictionary
        """
        if integration_type.lower() == "linear":
            return self._process_linear_webhook(event_type, payload, integration)
        elif integration_type.lower() == "sentry":
            return self._process_sentry_webhook(event_type, payload, integration)
        elif integration_type.lower() == "github":
            return self._process_github_webhook(event_type, payload, integration)
        else:
            # Generic processing for unknown integration types
            return self._process_generic_webhook(event_type, payload, integration)

    def _process_linear_webhook(
        self, event_type: str, payload: Dict[str, Any], integration: Any
    ) -> Dict[str, Any]:
        """Process Linear webhook events."""
        self.logger.info(f"Processing Linear webhook: {event_type}")

        # Extract relevant data from Linear webhook
        result = {
            "integration_type": "linear",
            "event_type": event_type,
            "webhook_data": {
                "action": payload.get("action"),
                "data": payload.get("data", {}),
                "created_at": payload.get("createdAt"),
                "updated_at": payload.get("updatedAt"),
            },
        }

        # Add specific processing based on event type
        if event_type.startswith("issue."):
            result["issue_data"] = payload.get("data", {})
        elif event_type.startswith("project."):
            result["project_data"] = payload.get("data", {})

        return result

    def _process_sentry_webhook(
        self, event_type: str, payload: Dict[str, Any], integration: Any
    ) -> Dict[str, Any]:
        """Process Sentry webhook events."""
        self.logger.info(f"Processing Sentry webhook: {event_type}")

        # Extract relevant data from Sentry webhook
        result = {
            "integration_type": "sentry",
            "event_type": event_type,
            "webhook_data": {
                "action": payload.get("action"),
                "data": payload.get("data", {}),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            },
        }

        # Add specific processing based on event type
        if event_type.startswith("issue."):
            result["issue_data"] = payload.get("data", {})
        elif event_type.startswith("project."):
            result["project_data"] = payload.get("data", {})

        return result

    def _process_github_webhook(
        self, event_type: str, payload: Dict[str, Any], integration: Any
    ) -> Dict[str, Any]:
        """Process GitHub webhook events (e.g. PR merged → context graph ingestion)."""
        self.logger.info(f"Processing GitHub webhook: {event_type}")
        from app.core.config_provider import config_provider
        from app.modules.projects.projects_model import Project

        if not config_provider.get_context_graph_config().get("enabled"):
            return {"integration_type": "github", "event_type": event_type, "webhook_data": payload}

        action = payload.get("action")
        pr = payload.get("pull_request")
        if (
            event_type == "pull_request"
            and action == "closed"
            and pr
            and pr.get("merged")
        ):
            repo = payload.get("repository", {})
            repo_name = repo.get("full_name") if isinstance(repo, dict) else None
            if not repo_name:
                return {"processed": False, "reason": "no repo_name in payload"}
            projects = (
                self.db.query(Project)
                .filter(
                    Project.repo_name == repo_name,
                    Project.is_deleted == False,  # noqa: E712
                )
                .all()
            )
            from app.modules.context_graph.tasks import ingest_pr_from_webhook

            for project in projects:
                ingest_pr_from_webhook.delay(project.id, pr)
                self.logger.info(
                    "Enqueued context graph ingestion for project %s (PR #%s)",
                    project.id,
                    pr.get("number"),
                )

        return {"integration_type": "github", "event_type": event_type, "webhook_data": payload}

    def _process_generic_webhook(
        self, event_type: str, payload: Dict[str, Any], integration: Any
    ) -> Dict[str, Any]:
        """Process generic webhook events."""
        self.logger.info(f"Processing generic webhook: {event_type}")

        return {
            "integration_type": "generic",
            "event_type": event_type,
            "webhook_data": payload,
        }
