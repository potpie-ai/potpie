from typing import Dict, Any
import asyncio
import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session


class SendSlackMessageInput(BaseModel):
    message: str = Field(description="The message text to send to Slack")
    webhook_url: str = Field(description="The Slack webhook URL to send the message to")


class SendSlackMessageTool:
    name = "Send Slack Message"
    description = """Send a message to a Slack channel via webhook URL.
        :param message: string, the message text to send to Slack.
        :param webhook_url: string, the Slack webhook URL to send the message to.

        Returns dictionary with success status and any error messages.
        """

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id

    async def arun(self, message: str, webhook_url: str) -> Dict[str, Any]:
        """Async version that handles the core logic"""
        try:
            if not webhook_url:
                return {
                    "success": False,
                    "error": "Webhook URL is required but not provided"
                }

            if not message:
                return {
                    "success": False,
                    "error": "Message is required but not provided"
                }

            payload = {"text": message}

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Slack message sent successfully"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to send Slack message. Status code: {response.status_code}, Response: {response.text}"
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Error sending Slack message: {str(e)}"
            }

    def run(self, message: str, webhook_url: str) -> Dict[str, Any]:
        """Synchronous version that runs the async version"""
        return asyncio.run(self.arun(message, webhook_url))


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
                       - webhook_url (str): The Slack webhook URL to send the message to""",
        args_schema=SendSlackMessageInput,
    ) 