"""Jira OAuth integration following Atlassian 3LO documentation."""

from typing import Dict, Optional, Any
from starlette.config import Config
from .atlassian_oauth_base import AtlassianOAuthBase
import httpx
import logging


class JiraOAuth(AtlassianOAuthBase):
    """Jira OAuth integration handler using Atlassian 3LO"""

    @property
    def product_name(self) -> str:
        return "jira"

    @property
    def default_scope(self) -> str:
        return self.config(
            "JIRA_OAUTH_SCOPE",
            default="read:jira-user read:jira-work write:jira-work manage:jira-webhook offline_access manage:jira-configuration",
        )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Override with Jira-specific env vars if provided
        self.client_id = config("JIRA_CLIENT_ID", default=self.client_id)
        self.client_secret = config("JIRA_CLIENT_SECRET", default=self.client_secret)

    async def create_webhook(
        self,
        cloud_id: str,
        access_token: str,
        webhook_url: str,
        events: Optional[list] = None,
        jql: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a Jira webhook for a Jira Cloud site (cloud_id).

        Returns the created webhook object (including id) on success.
        """
        if events is None:
            events = [
                "jira:issue_created",
                "jira:issue_updated",
                "jira:issue_deleted",
                "comment_created",
            ]

        if not cloud_id:
            raise Exception("cloud_id (site id) is required to create webhook")

        url = f"{self.API_BASE_URL}/ex/jira/{cloud_id}/rest/api/3/webhook"

        # Use the dynamic registration format required for OAuth2/Connect apps:
        # { "url": "https://app/webhook", "webhooks": [ { "events": [...], "jqlFilter": "..." } ] }
        # Note: OAuth webhooks require a non-empty JQL filter with supported operators only
        # Supported fields: project, issueKey, issuetype, status, priority, assignee, reporter
        webhook_entry: Dict[str, Any] = {"events": events}

        # Set JQL filter - default to matching all issues if not specified
        if jql is not None:
            webhook_entry["jqlFilter"] = jql
        else:
            # Match all issues across all projects using a condition that's always true
            # Using "priority != NonExistentPriority" as a workaround to match all issues
            webhook_entry["jqlFilter"] = (
                "priority != NonExistentPriority OR priority = NonExistentPriority"
            )

        payload: Dict[str, Any] = {"url": webhook_url, "webhooks": [webhook_entry]}

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Log the payload we are about to send for diagnostics
        try:
            logging.info(
                "Creating Jira webhook for site %s -> %s", cloud_id, webhook_url
            )
            logging.info("Jira create_webhook payload: %s", payload)
        except Exception:
            # best-effort logging; don't fail the request because logging failed
            pass

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        # Log the response for debugging
        logging.info(
            "Webhook creation response status: %s, body: %s",
            response.status_code,
            response.text,
        )

        if response.status_code not in (200, 201):
            logging.error(
                "Failed to create Jira webhook (%s): %s",
                response.status_code,
                response.text,
            )
            raise Exception(
                f"Failed to create Jira webhook: {response.status_code} {response.text}"
            )

        # Successful response; return parsed JSON
        try:
            result = response.json()
            logging.info("Webhook creation result: %s", result)
            return result
        except Exception as e:
            logging.warning("Failed to parse webhook response as JSON: %s", e)
            return {"status_code": response.status_code, "text": response.text}

    async def delete_webhook(
        self, cloud_id: str, access_token: str, webhook_id: str
    ) -> bool:
        """Delete a Jira webhook by id for a given cloud_id (site)."""
        if not cloud_id or not webhook_id:
            raise Exception("cloud_id and webhook_id are required to delete webhook")

        # OAuth apps must use bulk delete endpoint
        url = f"{self.API_BASE_URL}/ex/jira/{cloud_id}/rest/api/3/webhook"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Webhook IDs must be sent as an array in the request body
        payload = {"webhookIds": [int(webhook_id)]}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE", url, headers=headers, json=payload
            )

        if response.status_code not in (200, 202, 204):
            logging.error(
                "Failed to delete Jira webhook (%s): %s",
                response.status_code,
                response.text,
            )
            return False

        logging.info(f"Successfully deleted webhook {webhook_id} for site {cloud_id}")
        return True
