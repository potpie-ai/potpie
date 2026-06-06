"""
Event Bus Module

This module provides an event bus interface for capturing webhook events
from integrations and queuing them for downstream services.
"""

from .interfaces import EventBusInterface
from .celery_bus import CeleryEventBus
from .schemas import Event, EventMetadata, WebhookEvent

__all__ = [
    "EventBusInterface",
    "CeleryEventBus",
    "Event",
    "EventMetadata",
    "WebhookEvent",
]
