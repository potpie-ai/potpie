import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)


class ParseWebhookHelper:
    def __init__(self):
        self.url = os.getenv("SLACK_PARSE_WEBHOOK_URL", None)

    async def send_slack_notification(self, project_id, error_msg=None):
        message = {"text": f"Project ID: {project_id}\nStatus: ERROR"}

        if error_msg:
            message["text"] += f"\nError Message: {error_msg}"

        try:
            if self.url:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        self.url,
                        content=json.dumps(message),
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code != 200:
                        logger.warning(
                            "Failed to send message to Slack: %s %s",
                            response.status_code,
                            response.text,
                        )
        except Exception as e:
            logger.warning("Error sending message to Slack: %s", e)
