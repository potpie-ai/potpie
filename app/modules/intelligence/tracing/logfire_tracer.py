"""
Logfire Tracing Integration for LLM Monitoring

This module sets up Pydantic Logfire tracing for monitoring:
- Pydantic AI agent operations (agent runs, delegations, structured outputs)
- LLM API calls (via LiteLLM - Anthropic, OpenAI, etc.)
- Token usage and costs
- Multi-agent delegations (supervisor → subagents)
- Tool calls and results
- Performance metrics and latency

Metadata (user_id, environment, conversation_id, etc.) is attached via Baggage so
every span in a trace gets these attributes. You can then run SQL in Logfire to
filter by user, environment, or project (e.g. attributes->>'user_id' = '...').

CRITICAL SETUP ORDER:
1. Call initialize_logfire_tracing() at application startup (in main.py)
2. This configures Logfire and instruments Pydantic AI BEFORE any agents are created
3. Create agents with instrument=True to enable tracing
4. Use logfire_trace_metadata(user_id=..., conversation_id=..., etc.) around agent
   runs and in request middleware so traces are queryable by user/environment.

Usage:
    from app.modules.intelligence.tracing.logfire_tracer import (
        initialize_logfire_tracing,
        logfire_trace_metadata,
    )

    # Initialize once at application startup (BEFORE creating any agents)
    initialize_logfire_tracing()

    # Wrap agent run or request so all Pydantic AI / LiteLLM spans get metadata
    with logfire_trace_metadata(user_id=user_id, conversation_id=conv_id, run_id=run_id):
        ...

What gets traced:
    - Pydantic AI: Agent.run(), Agent.run_sync(), agent.iter(), structured outputs, retries
    - LiteLLM: completion(), acompletion(), streaming calls
    - Multi-agent system: All supervisor and subagent interactions
    - Tool calls: Function calls and results
    - Tokens: Usage and cost tracking
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import logfire

from app.modules.utils.logger import setup_logger

# Max length for baggage/attribute values (Logfire truncates longer strings)
_LOGFIRE_ATTR_MAX_LEN = 1000

logger = setup_logger(__name__)

# Global flag to track if Logfire is initialized
_LOGFIRE_INITIALIZED = False


def initialize_logfire_tracing(
    project_name: Optional[str] = None,
    token: Optional[str] = None,
    environment: Optional[str] = None,
    send_to_logfire: bool = True,
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

    Returns:
        bool: True if initialization successful, False otherwise

    Environment Variables:
        LOGFIRE_SEND_TO_CLOUD: Set to "false" to disable sending traces to Logfire cloud (default: "true")
        LOGFIRE_TOKEN: API token for Logfire (required for cloud tracing)
        LOGFIRE_PROJECT_NAME: Project name in Logfire UI (optional)
        LOGFIRE_SERVICE_NAME: Service name for resource attributes (default: project or "potpie")
        ENV or LOGFIRE_ENVIRONMENT: Environment (e.g. development, staging, production) for traces (default: "local")
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

        # Environment (queryable in Logfire; use ENV or LOGFIRE_ENVIRONMENT)
        env = (
            environment
            or os.getenv("LOGFIRE_ENVIRONMENT")
            or os.getenv("ENV", "local")
        )
        config_kwargs["environment"] = env

        # Project name (optional)
        project = project_name or os.getenv("LOGFIRE_PROJECT_NAME")
        if project:
            config_kwargs["project_name"] = project

        # Service name for resource attributes (queryable in SQL; defaults to project or "potpie")
        service_name = os.getenv("LOGFIRE_SERVICE_NAME") or project or "potpie"
        config_kwargs["service_name"] = service_name
        logger.debug(
            "Initializing Logfire tracing",
            project=project,
            environment=env,
            send_to_logfire=send_to_logfire,
        )
        logfire.configure(**config_kwargs)

        logfire.instrument_pydantic_ai()
        logger.info("Instrumented Pydantic AI for Logfire tracing")

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


@contextmanager
def logfire_trace_metadata(**kwargs: Any):
    """
    Set trace-wide metadata (Baggage) so every span in the trace gets these attributes.

    Use this around agent runs, Celery tasks, or HTTP request handlers so that
    Pydantic AI and LiteLLM spans are queryable in Logfire by user_id, conversation_id,
    run_id, agent_id, environment, etc.

    All values are stringified and truncated to 1000 chars (Logfire limit).
    When Logfire is not enabled, this is a no-op.

    Example (Celery task):
        with logfire_trace_metadata(
            user_id=user_id,
            conversation_id=conversation_id,
            run_id=run_id,
            agent_id=agent_id,
        ):
            # All Pydantic AI / LiteLLM spans here get these attributes
            ...

    Example (FastAPI middleware): set user_id and request_id so HTTP and LLM spans
    can be filtered in Logfire SQL.
    """
    # Don't rely on our private _LOGFIRE_INITIALIZED flag here — in some
    # processes Logfire may have been configured elsewhere (or via env)
    # so we just best-effort call set_baggage if kwargs are provided.
    if not kwargs:
        yield
        return

    str_attrs: Dict[str, str] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        s = str(value).strip()
        if len(s) > _LOGFIRE_ATTR_MAX_LEN:
            s = s[:_LOGFIRE_ATTR_MAX_LEN]
            logger.debug(
                "Logfire attribute truncated",
                key=key,
                max_len=_LOGFIRE_ATTR_MAX_LEN,
            )
        str_attrs[key] = s

    if not str_attrs:
        yield
        return

    try:
        import logfire

        with logfire.set_baggage(**str_attrs):
            yield
    except Exception as e:
        logger.debug(
            "Logfire set_baggage failed (non-fatal)",
            error=str(e),
        )
        yield


@contextmanager
def logfire_llm_call_metadata(
    user_id: Optional[str] = None,
    environment: Optional[str] = None,
    **extra_attrs: Any,
):
    """
    Set baggage and wrap the next LLM call in a span so user_id and environment
    appear on the trace (including LiteLLM/acompletion spans when possible).

    Use this around every call to litellm.acompletion() so that:
    1. Baggage is set right before the call (so LiteLLM-created spans get user_id etc.)
    2. A parent span "llm_call" is created with user_id and environment, so you can
       always filter by these in Logfire even if the instrumented span doesn't inherit baggage.

    Call from provider_service.call_llm (and similar) with user_id=self.user_id.
    """
    attrs: Dict[str, str] = {}
    if user_id is not None:
        attrs["user_id"] = str(user_id).strip()[: _LOGFIRE_ATTR_MAX_LEN]
    env = (environment or os.getenv("LOGFIRE_ENVIRONMENT") or os.getenv("ENV") or "local").strip()
    attrs["environment"] = env[: _LOGFIRE_ATTR_MAX_LEN]
    for k, v in extra_attrs.items():
        if v is None:
            continue
        s = str(v).strip()
        if len(s) > _LOGFIRE_ATTR_MAX_LEN:
            s = s[:_LOGFIRE_ATTR_MAX_LEN]
        attrs[k] = s

    if not attrs:
        yield
        return

    try:
        import logfire

        with logfire.set_baggage(**attrs):
            span_attrs = {"environment": attrs["environment"]}
            if attrs.get("user_id"):
                span_attrs["user_id"] = attrs["user_id"]
            for k, v in attrs.items():
                if k not in ("user_id", "environment") and v:
                    span_attrs[k] = v
            with logfire.span("llm_call", **span_attrs):
                yield
    except Exception as e:
        logger.debug(
            "Logfire LLM metadata failed (non-fatal)",
            error=str(e),
        )
        yield


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
