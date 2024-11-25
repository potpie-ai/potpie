import logging
import os

from posthog import Posthog

logger = logging.getLogger(__name__)


class PostHogClient:
    def __init__(self):
        self.environment = os.getenv("ENV")

        # Only initialize PostHog if not in development mode
        if self.environment == "development":
            self.posthog = None
        else:
            self.api_key = os.getenv("POSTHOG_API_KEY")
            self.posthog_host = os.getenv("POSTHOG_HOST")
            self.posthog = Posthog(self.api_key, host=self.posthog_host)

    def send_event(self, user_id: str, event_name: str, properties: dict):
        """
        Sends a custom event to PostHog in the stage and production environment.
        Args:
            user_id (str): The ID of the user performing the action.
            event_name (str): The name of the event to track.
            properties (dict): Additional properties related to the event.
        """
        if self.environment == "development":
            logging.info("Development mode enabled. Skipping PostHog event : user_id : %s, event_name %s, properties: %s",
                         user_id, event_name, properties)
            return

        if self.posthog is not None:  # Ensure posthog is initialized
            try:
                self.posthog.capture(
                    user_id,  # User's unique identifier
                    event=event_name,  # The event name
                    properties=properties,  # Additional event metadata
                )
            except Exception as e:
                # Basic error handling; could be expanded based on use case
                logger.warning(f"Failed to send event: {e}")
