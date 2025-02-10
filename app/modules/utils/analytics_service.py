import logging
import os

from posthog import Posthog
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AnalyticsService(ABC):
    """Interface for capturing analytics events"""

    @abstractmethod
    def capture_event(self, user_id: str, event_name: str, properties: dict):
        """
        Capture an event in the analytics platform.
        Args:
            user_id (str): The ID of the user performing the action.
            event_name (str): The name of the event to track.
            properties (dict): Additional properties related to the event.
        """
        pass


class MockAnalyticsService(AnalyticsService):
    """Mock analytics client for testing purposes and development"""

    def __init__(self):
        pass

    def capture_event(self, user_id: str, event_name: str, properties: dict):
        logger.info(
            f"captured event: {event_name} for user: {user_id} with properties: {properties}"
        )


class PosthogAnalyticsService(AnalyticsService):
    """PostHog analytics client for capturing events"""

    def __init__(self, api_key: str, host: str):
        self.posthog = Posthog(api_key, host)

    def capture_event(self, user_id: str, event_name: str, properties: dict):
        try:
            self.posthog.capture(user_id, event=event_name, properties=properties)
        except Exception as e:
            logger.warning(f"Failed to send event: {e}")
