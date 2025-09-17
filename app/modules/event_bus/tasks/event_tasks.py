"""
Event Bus Celery Tasks

Celery tasks for processing webhook and custom events.
"""

import logging
from typing import Any, Dict

from celery import Task

from app.celery.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.event_bus.handlers import WebhookEventHandler, CustomEventHandler

logger = logging.getLogger(__name__)


class EventTask(Task):
    """Base task class for event processing with database session management."""

    _db = None

    @property
    def db(self):
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=EventTask,
    name="app.modules.event_bus.tasks.event_tasks.process_webhook_event",
    queue="external-event-webhook",
    routing_key="external-event.webhook.*",
)
def process_webhook_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a webhook event from an integration.

    Args:
        event_data: WebhookEvent data dictionary

    Returns:
        Processing result dictionary
    """
    logger.info(f"Processing webhook event: {event_data.get('event_id')}")

    try:
        # Create event handler
        handler = WebhookEventHandler(self.db)

        # Process the event
        result = handler.process_event(event_data)

        logger.info(
            f"Successfully processed webhook event {event_data.get('event_id')}: {result}"
        )

        return {
            "status": "success",
            "event_id": event_data.get("event_id"),
            "result": result,
        }

    except Exception as e:
        logger.error(
            f"Failed to process webhook event {event_data.get('event_id')}: {str(e)}"
        )

        # Return error result for retry logic
        return {
            "status": "error",
            "event_id": event_data.get("event_id"),
            "error": str(e),
        }


@celery_app.task(
    bind=True,
    base=EventTask,
    name="app.modules.event_bus.tasks.event_tasks.process_custom_event",
    queue="external-event-custom",
    routing_key="external-event.custom.*",
)
def process_custom_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a custom event.

    Args:
        event_data: CustomEvent data dictionary

    Returns:
        Processing result dictionary
    """
    logger.info(f"Processing custom event: {event_data.get('event_id')}")

    try:
        # Create event handler
        handler = CustomEventHandler(self.db)

        # Process the event
        result = handler.process_event(event_data)

        logger.info(
            f"Successfully processed custom event {event_data.get('event_id')}: {result}"
        )

        return {
            "status": "success",
            "event_id": event_data.get("event_id"),
            "result": result,
        }

    except Exception as e:
        logger.error(
            f"Failed to process custom event {event_data.get('event_id')}: {str(e)}"
        )

        # Return error result for retry logic
        return {
            "status": "error",
            "event_id": event_data.get("event_id"),
            "error": str(e),
        }
