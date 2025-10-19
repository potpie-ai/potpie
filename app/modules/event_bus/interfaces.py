"""
Event Bus Interface

Abstract interface for event bus implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class EventBusInterface(ABC):
    """Abstract interface for event bus implementations."""

    @abstractmethod
    async def publish_webhook_event(
        self,
        integration_id: str,
        integration_type: str,
        event_type: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        source_ip: Optional[str] = None,
    ) -> str:
        """
        Publish a webhook event to the event bus.

        Args:
            integration_id: ID of the integration that sent the webhook
            integration_type: Type of integration (linear, sentry, etc.)
            event_type: Type of event (issue.created, issue.updated, etc.)
            payload: The webhook payload data
            headers: HTTP headers from the webhook request
            source_ip: Source IP address of the webhook request

        Returns:
            Event ID of the published event
        """
        pass

    @abstractmethod
    async def publish_custom_event(
        self,
        topic: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Publish a custom event to a specific topic.

        Args:
            topic: Topic/queue name for the event
            event_type: Type of event
            data: Event data payload
            metadata: Additional metadata for the event

        Returns:
            Event ID of the published event
        """
        pass
