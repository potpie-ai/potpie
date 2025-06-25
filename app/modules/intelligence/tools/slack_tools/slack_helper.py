import httpx
from typing import Dict, Any


async def send_slack_message(message: str, webhook_url: str) -> Dict[str, Any]:
    """
    Send a message to a Slack channel via webhook URL.
    
    Args:
        message: The message text to send to Slack
        webhook_url: The Slack webhook URL to send the message to
        
    Returns:
        Dictionary with success status and any error messages
    """
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