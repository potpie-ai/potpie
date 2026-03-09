import asyncio
import os

from posthog import Posthog

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class PostHogClient:
    def __init__(self):
        self.environment = os.getenv("ENV")

        # Only initialize PostHog if not in development mode
        if self.environment == "production":
            self.api_key = os.getenv("POSTHOG_API_KEY")
            self.posthog_host = os.getenv("POSTHOG_HOST")
            self.posthog = Posthog(self.api_key, host=self.posthog_host)
        else:
            self.posthog = None

    def _capture_sync(self, user_id: str, event_name: str, properties: dict) -> None:
        """Sync capture (Posthog SDK is sync-only)."""
        if self.posthog is None:
            return
        try:
            self.posthog.capture(
                user_id,
                event=event_name,
                properties=properties,
            )
        except Exception:
            # Do not log raw user_id to avoid leaking identifiers
            logger.exception(
                "Failed to send PostHog event",
                event_name=event_name,
            )

    def send_event(self, user_id: str, event_name: str, properties: dict) -> None:
        """
        Sends a custom event to PostHog. When called from async context, runs
        in a thread (fire-and-forget) to avoid blocking the event loop.
        """
        if self.environment != "production":
            return
        try:
            loop = asyncio.get_running_loop()
            # Fire-and-forget: run sync capture in thread pool
            loop.run_in_executor(
                None,
                lambda: self._capture_sync(user_id, event_name, properties),
            )
        except RuntimeError:
            # No running event loop (sync context)
            self._capture_sync(user_id, event_name, properties)
