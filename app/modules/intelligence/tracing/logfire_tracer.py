"""Logfire Tracing Integration for LLM Monitoring"""

import os
from typing import Any, Dict, Optional

import logfire

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


def _patch_otel_detach_for_async_context() -> None:
    """
    Patch OpenTelemetry's ContextVarsRuntimeContext.detach to suppress ValueError
    when the token was created in a different async context.

    When pydantic_ai runs tool calls, async generators can yield and resume in a
    different context; OTel then tries to detach a token that belongs to the
    previous context and raises ValueError. This patch catches that and no-ops
    so the request does not crash.
    """
    try:
        from opentelemetry.context import contextvars_context
    except ImportError:
        return

    _original_detach = contextvars_context.ContextVarsRuntimeContext.detach

    def _detach_safe(self: Any, token: Any) -> None:
        try:
            _original_detach(self, token)
        except ValueError as e:
            if "was created in a different Context" in str(e):
                pass  # Safe to ignore when context switched across async boundary
            else:
                raise

    contextvars_context.ContextVarsRuntimeContext.detach = _detach_safe  # type: ignore[method-assign]
    logger.debug("Patched OTel context detach for async context switching")

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
            _patch_otel_detach_for_async_context()
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


def should_instrument_pydantic_ai() -> bool:
    """
    Return whether pydantic_ai Agent should use OpenTelemetry instrumentation.

    Returns False when running in a Celery worker process to avoid OTel contextvar
    errors ('Token was created in a different Context') during async tool execution
    with prefork workers. Set by worker_process_init in celery_app.
    """
    return os.getenv("CELERY_WORKER") != "1"


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
