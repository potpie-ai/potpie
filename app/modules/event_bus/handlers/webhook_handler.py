"""
Webhook Event Handler

Handler for processing webhook events from integrations.
"""

from typing import Any, Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from integrations.application.integrations_service import IntegrationsService
from app.modules.context_graph.context_graph_pot_model import ContextGraphPot
from app.modules.context_graph.context_graph_pot_repository_model import (
    ContextGraphPotRepository,
)
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

    def _process_github_webhook(
        self, event_type: str, payload: Dict[str, Any], integration: Any
    ) -> Dict[str, Any]:
        """Route merged GitHub PR events to every pot whose repository matches.

        Routing rule:
        - match by ``external_repo_id`` (GitHub numeric id) when available;
        - fall back to ``(provider=github, provider_host, owner, repo)``.
        - If the same repo is attached to multiple pots, submit ingestion to
          each pot (graph partition key is the pot).
        """
        action = payload.get("action")
        pull_request = payload.get("pull_request") or {}
        merged = bool(pull_request.get("merged"))
        if event_type != "pull_request" or action != "closed" or not merged:
            return {
                "integration_type": "github",
                "event_type": event_type,
                "processed": False,
                "reason": "not_merged_pull_request_event",
            }

        repository = payload.get("repository") or {}
        repo_name = repository.get("full_name")
        repository_id = repository.get("id")
        pr_number = pull_request.get("number")
        if not repo_name or not pr_number:
            return {
                "integration_type": "github",
                "event_type": event_type,
                "processed": False,
                "reason": "missing_repo_or_pr_number",
            }

        matches = self._find_matching_pot_repositories(
            repo_name=str(repo_name),
            external_repo_id=str(repository_id) if repository_id else None,
        )
        if not matches:
            self.logger.info(
                "GitHub merged PR %s#%s ignored: no pot attached for repo",
                repo_name,
                pr_number,
            )
            return {
                "integration_type": "github",
                "event_type": event_type,
                "processed": False,
                "reason": "no_pot_attached_to_repo",
                "repo_name": repo_name,
            }

        routed: list[dict[str, Any]] = []
        source_event_id = payload.get("delivery") or payload.get("delivery_id")
        for pot_id, primary_repo_name in matches:
            try:
                receipt = self._submit_github_merged_pr(
                    pot_id=pot_id,
                    repo_name=primary_repo_name,
                    pr_number=int(pr_number),
                    repository_id=str(repository_id) if repository_id else None,
                    source_event_id=str(source_event_id) if source_event_id else None,
                    payload=payload,
                )
                routed.append(
                    {
                        "pot_id": pot_id,
                        "repo_name": primary_repo_name,
                        "event_id": receipt.get("event_id"),
                        "status": receipt.get("status"),
                    }
                )
            except Exception as e:
                self.logger.exception(
                    "GitHub merged PR routing failed for pot %s: %s", pot_id, e
                )
                routed.append(
                    {
                        "pot_id": pot_id,
                        "repo_name": primary_repo_name,
                        "error": str(e),
                    }
                )

        return {
            "integration_type": "github",
            "event_type": event_type,
            "processed": True,
            "repo_name": repo_name,
            "pr_number": int(pr_number),
            "routed": routed,
        }

    def _find_matching_pot_repositories(
        self,
        *,
        repo_name: str,
        external_repo_id: str | None,
    ) -> List[tuple[str, str]]:
        """Return (pot_id, repo_name) tuples for every non-archived pot/repo match."""
        base = (
            self.db.query(ContextGraphPotRepository)
            .join(
                ContextGraphPot,
                ContextGraphPot.id == ContextGraphPotRepository.pot_id,
            )
            .filter(ContextGraphPot.archived_at.is_(None))
        )
        rows: list[ContextGraphPotRepository] = []
        if external_repo_id:
            rows = list(
                base.filter(
                    ContextGraphPotRepository.provider == "github",
                    ContextGraphPotRepository.external_repo_id == external_repo_id,
                ).all()
            )
        if not rows:
            want = repo_name.lower()
            full_name = func.lower(
                func.concat(
                    ContextGraphPotRepository.owner,
                    "/",
                    ContextGraphPotRepository.repo,
                )
            )
            rows = list(
                base.filter(
                    ContextGraphPotRepository.provider == "github",
                    full_name == want,
                ).all()
            )
        return [(r.pot_id, f"{r.owner}/{r.repo}") for r in rows]

    def _submit_github_merged_pr(
        self,
        *,
        pot_id: str,
        repo_name: str,
        pr_number: int,
        repository_id: str | None,
        source_event_id: str | None,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        from app.modules.context_graph.wiring import build_container_for_session
        from domain.ingestion_kinds import INGESTION_KIND_GITHUB_MERGED_PR
        from domain.ingestion_event_models import IngestionSubmissionRequest

        container = build_container_for_session(self.db)
        request = IngestionSubmissionRequest(
            pot_id=pot_id,
            ingestion_kind=INGESTION_KIND_GITHUB_MERGED_PR,
            source_channel="webhook",
            source_system="github",
            event_type="pull_request",
            action="merged",
            repo_name=repo_name,
            source_event_id=source_event_id,
            payload={
                "pr_number": pr_number,
                "repo_name": repo_name,
                "repository_id": repository_id,
                "is_live_bridge": True,
            },
        )
        receipt = container.ingestion_submission(self.db).submit(request, sync=False)
        return {"event_id": receipt.event_id, "status": receipt.status}
