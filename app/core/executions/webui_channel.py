"""
Web UI Channel implementation for HITL.

This channel handles HITL requests through the web UI.
Requests are stored in the database and retrieved via API endpoints.
"""

import logging
from typing import Optional, Dict, Any

from app.core.executions.hitl import HITLRequest, HITLResponse
from app.core.executions.hitl_channel import HITLChannel

logger = logging.getLogger(__name__)


class WebUIChannel(HITLChannel):
    """
    Web UI channel for HITL requests.

    For Web UI channel, requests are stored in the database and retrieved
    via REST API endpoints. This channel implementation mainly serves as
    a placeholder for the channel abstraction pattern.
    """

    @property
    def channel_id(self) -> str:
        """Return the unique identifier for this channel type."""
        return "web"

    @property
    def channel_name(self) -> str:
        """Return the human-readable name for this channel."""
        return "Web UI"

    async def send_request(self, request: HITLRequest) -> bool:
        """
        Send a HITL request through the Web UI channel.

        For Web UI, the request is already stored in the database by the executor.
        This method just confirms the request is available via the API.

        Args:
            request: The HITL request to send

        Returns:
            bool: Always returns True (request is stored in database)
        """
        logger.info(
            f"Web UI channel: Request {request.request_id} is available via API"
        )
        # Request is already stored in database, just return success
        return True

    async def receive_response(
        self, request_id: str
    ) -> Optional[HITLResponse]:
        """
        Receive a response for a HITL request.

        For Web UI, responses are submitted via REST API and stored in database.
        This method is not used for Web UI channel - responses are handled
        directly through the API endpoints.

        Args:
            request_id: The request ID to check for responses

        Returns:
            Optional[HITLResponse]: None (responses handled via API)
        """
        # For Web UI, responses are handled via API endpoints, not polling
        return None

    async def check_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check the status of a HITL request in this channel.

        Args:
            request_id: The request ID to check

        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "channel": self.channel_id,
            "status": "available",
            "message": "Request is available via Web UI API",
        }

