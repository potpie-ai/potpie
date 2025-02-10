from app.modules.utils.analytics_service import (
    AnalyticsService,
    PosthogAnalyticsService,
    MockAnalyticsService,
)
from dotenv import load_dotenv
from starlette.datastructures import State
from fastapi import Request
import logging
import os

logger = logging.getLogger(__name__)


# Analytics service configuration


def init_analytics_service() -> AnalyticsService:
    POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY") or ""
    POSTHOG_HOST = os.getenv("POSTHOG_HOST") or ""

    if POSTHOG_API_KEY != "" and POSTHOG_HOST != "":
        logger.info(f"using PostHog analytics service with host {POSTHOG_HOST}")
        return PosthogAnalyticsService(POSTHOG_API_KEY, POSTHOG_HOST)

    logger.info(
        "no AnalyticsService envs found (POSTHOG_API_KEY, POSTHOG_HOST), using MockAnalytics service"
    )
    return MockAnalyticsService()


def get_analytics_service(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service


# State initialization


def init_state(state: State):
    load_dotenv(override=True)
    state.analytics_service = init_analytics_service()
    return state
