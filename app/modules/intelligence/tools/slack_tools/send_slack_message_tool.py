from typing import Dict, Any, Optional
import asyncio
import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.slack_tools.slack_helper import (
    send_slack_message, 
    get_slack_webhook_for_workflow
)


class SendSlackMessageInput(BaseModel):
    message: str = Field(description="The message text to send to Slack")
    webhook_url: Optional[str] = Field(default=None, description="The Slack webhook URL to send the message to")
    workflow_id: Optional[str] = Field(default=None, description="The workflow ID to get the webhook URL from secret storage")


class SendSlackMessageTool:
    name = "Send Slack Message"
    description = """Send a message to a Slack channel via webhook URL.
        :param message: string, the message text to send to Slack.
        :param webhook_url: string, optional - the Slack webhook URL to send the message to. If not provided, will try to get from workflow secrets.
        :param workflow_id: string, optional - the workflow ID to get the webhook URL from secret storage.

        Returns dictionary with success status and any error messages.
        """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, message: str, webhook_url: Optional[str] = None, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Async version that handles the core logic"""
        try:
            if not message:
                return {
                    "success": False,
                    "error": "Message is required but not provided"
                }

            # If webhook_url is not provided, try to get it from workflow secrets
            if not webhook_url and workflow_id:
                webhook_url = await get_slack_webhook_for_workflow(workflow_id, self.user_id, self.db)
                
                if not webhook_url:
                    return {
                        "success": False,
                        "error": f"No Slack webhook URL found for workflow {workflow_id}. Please configure it in the secret manager."
                    }

            if not webhook_url:
                return {
                    "success": False,
                    "error": "Webhook URL is required. Either provide webhook_url directly or workflow_id to retrieve from secrets."
                }

            # Use the helper function to send the message
            return await send_slack_message(message, webhook_url)
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Error sending Slack message: {str(e)}"
            }

    def run(self, message: str, webhook_url: Optional[str] = None, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async version"""
        return asyncio.run(self.arun(message, webhook_url, workflow_id))


def send_slack_message_tool(db: Session, user_id: str) -> StructuredTool:
    """
    Create a tool for sending messages to Slack channels.

    Args:
        db: Database session
        user_id: The user ID for context

    Returns:
        A configured StructuredTool for sending Slack messages
    """
    tool_instance = SendSlackMessageTool(db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="Send Slack Message",
        description="""Send a message to a Slack channel via webhook URL.
                       Inputs for the run method:
                       - message (str): The message text to send to Slack
                       - webhook_url (str): The Slack webhook URL to send the message to
                       - workflow_id (str): The workflow ID to get the webhook URL from secret storage""",
        args_schema=SendSlackMessageInput,
    ) 