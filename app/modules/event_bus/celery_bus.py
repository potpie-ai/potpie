"""
Celery-based Event Bus Implementation

Implementation of EventBusInterface using Celery and Redis.
"""

import logging
from typing import Any, Dict, Optional

from celery import Celery

from .interfaces import EventBusInterface
from .schemas import CustomEvent, EventMetadata, WebhookEvent

logger = logging.getLogger(__name__)


class CeleryEventBus(EventBusInterface):
    """Celery-based implementation of EventBusInterface."""

    def __init__(self, celery_app: Celery):
        """
        Initialize the Celery event bus.

        Args:
            celery_app: Configured Celery application instance
        """
        self.celery_app = celery_app
        self.logger = logger

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

        Publishes all webhook events to the single 'external-event' queue
        with event_source set to the integration type.
        """
        try:
            # Create webhook event
            webhook_event = WebhookEvent(
                integration_id=integration_id,
                integration_type=integration_type,
                event_type=event_type,
                event_source=integration_type,  # Set event_source to integration_type
                payload=payload,
                headers=headers,
                source_ip=source_ip,
                metadata=EventMetadata(
                    source="webhook",
                    correlation_id=headers.get("X-Correlation-ID") if headers else None,
                ),
            )

            # Queue the event to the single external-event queue
            task = self.celery_app.send_task(
                "app.modules.event_bus.tasks.event_tasks.process_webhook_event",
                args=[webhook_event.model_dump()],
                queue="external-event",
                routing_key=f"external-event.{integration_type}.{event_type}",
            )

            self.logger.info(
                f"Published webhook event {webhook_event.event_id} from {integration_type} "
                f"to external-event queue, task ID: {task.id}"
            )

            return webhook_event.event_id

        except Exception as e:
            self.logger.error(f"Failed to publish webhook event: {str(e)}")
            raise

    async def publish_custom_event(
        self,
        topic: str,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        event_source: str = "custom",
    ) -> str:
        """
        Publish a custom event to the event bus.

        Publishes all custom events to the single 'external-event' queue
        with event_source set to the provided source.
        """
        try:
            # Create custom event
            metadata_dict = metadata or {}
            # Ensure source is set to "custom" and not overridden
            metadata_dict["source"] = "custom"

            custom_event = CustomEvent(
                topic=topic,
                event_type=event_type,
                event_source=event_source,  # Set event_source
                data=data,
                metadata=EventMetadata(**metadata_dict),
            )

            # Queue the event to the single external-event queue
            task = self.celery_app.send_task(
                "app.modules.event_bus.tasks.event_tasks.process_custom_event",
                args=[custom_event.model_dump()],
                queue="external-event",
                routing_key=f"external-event.{topic}",
            )

            self.logger.info(
                f"Published custom event {custom_event.event_id} from {event_source} "
                f"to external-event queue, task ID: {task.id}"
            )

            return custom_event.event_id

        except Exception as e:
            self.logger.error(f"Failed to publish custom event: {str(e)}")
            raise
