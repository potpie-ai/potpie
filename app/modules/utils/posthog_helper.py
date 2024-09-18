import os

from posthog import Posthog


class PostHogClient:
    def __init__(self):
        self.api_key = os.getenv("POSTHOG_API_KEY")
        self.posthog_host = os.getenv("POSTHOG_HOST")
        self.posthog = Posthog(self.api_key, host=self.posthog_host)

    def send_event(self, user_id: str, event_name: str, properties: dict):
        """
        Sends a custom event to PostHog.
        Args:
            user_id (str): The ID of the user performing the action.
            event_name (str): The name of the event to track.
            properties (dict): Additional properties related to the event.
        """
        try:
            self.posthog.capture(
                user_id,  # User's unique identifier
                event=event_name,  # The event name
                properties=properties,  # Additional event metadata
            )
        except Exception as e:
            # Basic error handling; could be expanded based on use case
            print(f"Failed to send event: {e}")
