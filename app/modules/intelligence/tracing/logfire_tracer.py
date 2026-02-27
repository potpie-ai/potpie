"""Logfire Tracing Integration for LLM Monitoring"""

import os
from typing import Any, Dict, Optional

import logfire

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global flag to track if Logfire is initialized
_LOGFIRE_INITIALIZED = False


def initialize_logfire_tracing(
    project_name: Optional[str] = None,
    token: Optional[str] = None,
    environment: Optional[str] = None,
    send_to_logfire: bool = True,
    instrument_pydantic_ai: bool = True,
) -> bool:
    """
    Initialize Logfire tracing for the application.

    This should be called once at application startup, ideally in main.py
    before any LLM calls are made.

    Args:
        project_name: Name of the project in Logfire UI. If None, reads from LOGFIRE_PROJECT_NAME env var
        token: Logfire API token. If None, reads from LOGFIRE_TOKEN env var
        environment: Environment identifier (e.g., "development", "production", "staging", "testing")
        send_to_logfire: Whether to send traces to Logfire cloud (default: True)
        instrument_pydantic_ai: Whether to instrument Pydantic AI for tracing (default: True).
            Set to False in Celery workers to avoid OpenTelemetry contextvar errors
            ("Token was created in a different Context") when async generators yield
            during tool execution with prefork workers.

    Returns:
        bool: True if initialization successful, False otherwise

    Environment Variables:
        LOGFIRE_SEND_TO_CLOUD: Set to "false" to disable sending traces to Logfire cloud (default: "true")
        LOGFIRE_TOKEN: API token for Logfire (required for cloud tracing)
        LOGFIRE_PROJECT_NAME: Project name in Logfire UI (optional)
        ENV: Environment identifier - used as "environment" attribute in traces (default: "local")
    """
    global _LOGFIRE_INITIALIZED

    # Check if cloud sending is disabled via env var
    if os.getenv("LOGFIRE_SEND_TO_CLOUD", "true").lower() == "false":
        send_to_logfire = False

    # Check if already initialized
    if _LOGFIRE_INITIALIZED:
        logger.info("Logfire tracing already initialized")
        return True

    try:
        config_kwargs: Dict[str, Any] = {}

        token = token or os.getenv("LOGFIRE_TOKEN")
        if token:
            config_kwargs["token"] = token
            config_kwargs["send_to_logfire"] = send_to_logfire
        else:
            config_kwargs["send_to_logfire"] = False

        env = environment or os.getenv("ENV", "local")
        config_kwargs["environment"] = env

        project = project_name or os.getenv("LOGFIRE_PROJECT_NAME", "potpie")
        logger.debug(
            "Initializing Logfire tracing",
            project=project,
            environment=env,
            send_to_logfire=send_to_logfire,
        )
        logfire.configure(**config_kwargs)

        if instrument_pydantic_ai:
            logfire.instrument_pydantic_ai()
            logger.info("Instrumented Pydantic AI for Logfire tracing")
        else:
            logger.debug(
                "Skipped Pydantic AI instrumentation (avoids OTel contextvar errors in Celery prefork)"
            )

        logfire.instrument_litellm()
        logger.info("Instrumented LiteLLM for Logfire tracing")

        _LOGFIRE_INITIALIZED = True

        logger.info("Logfire tracing initialized successfully.")
        return True

    except Exception as e:
        logger.warning(
            "Failed to initialize Logfire tracing (non-fatal)",
            error=str(e),
        )
        return False


def is_logfire_enabled() -> bool:
    """Check if Logfire tracing is enabled and initialized."""
    return _LOGFIRE_INITIALIZED


def shutdown_logfire_tracing():
    """
    Shutdown Logfire tracing.

    This should be called on application shutdown to ensure all traces are sent.
    Note: Logfire handles flushing automatically, but this provides a clean shutdown.
    """
    global _LOGFIRE_INITIALIZED

    if not _LOGFIRE_INITIALIZED:
        return

    try:
        import logfire

        # Logfire handles flushing automatically
        # Force a final flush to ensure all spans are sent
        logfire.force_flush()
        logger.info("Logfire tracing shutdown successfully")

        _LOGFIRE_INITIALIZED = False

    except Exception as e:
        logger.warning("Error shutting down Logfire tracing", error=str(e))
