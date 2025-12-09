"""
HITL Channel abstraction for extensible communication channels.

This module defines the abstract base class for HITL channels,
allowing different communication methods (Web UI, Email, Slack, etc.)
to be implemented as pluggable adapters.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from app.core.executions.hitl import HITLRequest, HITLResponse


class HITLChannel(ABC):
    """Abstract base class for HITL communication channels."""

    @abstractmethod
    async def send_request(self, request: HITLRequest) -> bool:
        """
        Send a HITL request through this channel.

        Args:
            request: The HITL request to send

        Returns:
            bool: True if request was sent successfully, False otherwise
        """
        pass

    @abstractmethod
    async def receive_response(
        self, request_id: str
    ) -> Optional[HITLResponse]:
        """
        Receive a response for a HITL request.

        Args:
            request_id: The request ID to check for responses

        Returns:
            Optional[HITLResponse]: The response if available, None otherwise
        """
        pass

    @abstractmethod
    async def check_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check the status of a HITL request in this channel.

        Args:
            request_id: The request ID to check

        Returns:
            Dict[str, Any]: Status information including delivery status, etc.
        """
        pass

    @property
    @abstractmethod
    def channel_id(self) -> str:
        """Return the unique identifier for this channel type."""
        pass

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Return the human-readable name for this channel."""
        pass

