import os
from posthog import Posthog

class PostHogClient:
    def __init__(self):
        self.is_staging_or_production = os.getenv("ENV") in ["staging", "production"]
        
        # Only initialize PostHog if not in development mode
        if self.is_staging_or_production:
            self.api_key = os.getenv("POSTHOG_API_KEY")
            self.posthog_host = os.getenv("POSTHOG_HOST")
            self.posthog = Posthog(self.api_key, host=self.posthog_host)
        else:
            self.posthog = None

    def send_event(self, user_id: str, event_name: str, properties: dict):
        """
        Sends a custom event to PostHog in the stage and production environment.
        Args:
            user_id (str): The ID of the user performing the action.
            event_name (str): The name of the event to track.
            properties (dict): Additional properties related to the event.
        """
        if not self.is_staging_or_production:
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
                print(f"Failed to send event: {e}")
