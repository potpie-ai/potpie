import json
import os

import requests

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ParseWebhookHelper:
    def __init__(self):
        self.url = os.getenv("SLACK_PARSE_WEBHOOK_URL", None)

    async def send_slack_notification(self, project_id, error_msg=None):
        message = {"text": f"Project ID: {project_id}\nStatus: ERROR"}

        if error_msg:
            message["text"] += f"\nError Message: {error_msg}"

        try:
            if self.url:
                response = requests.post(
                    self.url,
                    data=json.dumps(message),
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    logger.error(
                        f"Failed to send message to Slack: {response.status_code} {response.text}"
                    )

        except Exception as e:
            logger.error(f"Error sending message to Slack: {e}")
