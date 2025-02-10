from app.modules.utils.analytics_service import (
    AnalyticsService,
    PosthogAnalyticsService,
    MockAnalyticsService,
)
from app.modules.utils.ai_observability_service import (
    AiObservabilityService,
    AgentopsAiObservabilityService,
    MockAiObservabilityService,
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


# AIObservability service configuration


def init_ai_observability_service() -> AiObservabilityService:
    AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY") or ""

    if AGENTOPS_API_KEY != "":
        logger.info(f"using Agentops ai observability service")
        return AgentopsAiObservabilityService(AGENTOPS_API_KEY)

    logger.info(
        "no AnalyticsService envs found (POSTHOG_API_KEY, POSTHOG_HOST), using MockAnalytics service"
    )
    return MockAiObservabilityService()


def get_ai_observability_service(request: Request) -> AiObservabilityService:
    return request.app.state.ai_observability_service


# State initialization


def init_state(state: State):
    load_dotenv(override=True)
    state.analytics_service = init_analytics_service()
    state.ai_observability_service = init_ai_observability_service()
    return state
