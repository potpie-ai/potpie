"""
Logfire Tracing Integration for LLM Monitoring

This module sets up Pydantic Logfire tracing for monitoring:
- Pydantic AI agent operations (agent runs, delegations, structured outputs)
- LLM API calls (via LiteLLM - Anthropic, OpenAI, etc.)
- Token usage and costs
- Multi-agent delegations (supervisor → subagents)
- Tool calls and results
- Performance metrics and latency

CRITICAL SETUP ORDER:
1. Call initialize_logfire_tracing() at application startup (in main.py)
2. This configures Logfire and instruments Pydantic AI BEFORE any agents are created
3. Create agents with instrument=True to enable tracing

Usage:
    from app.modules.intelligence.tracing.logfire_tracer import initialize_logfire_tracing

    # Initialize once at application startup (BEFORE creating any agents)
    initialize_logfire_tracing()

    # Then create agents with instrument=True
    agent = Agent(model=..., tools=..., instrument=True)

What gets traced:
    - Pydantic AI: Agent.run(), Agent.run_sync(), agent.iter(), structured outputs, retries
    - LiteLLM: completion(), acompletion(), streaming calls
    - Multi-agent system: All supervisor and subagent interactions
    - Tool calls: Function calls and results
    - Tokens: Usage and cost tracking
"""

import os
from typing import Any, Dict, Optional

from app.modules.utils.logger import setup_logger

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
        LOGFIRE_ENABLED: Set to "false" to disable Logfire tracing (default: "true")
        LOGFIRE_TOKEN: API token for Logfire (required for cloud tracing)
        LOGFIRE_PROJECT_NAME: Project name in Logfire UI (optional)
        ENV: Environment identifier - used as "environment" attribute in traces (default: "local")
    """
    global _LOGFIRE_INITIALIZED

    # Check if Logfire is disabled
    if os.getenv("LOGFIRE_ENABLED", "true").lower() == "false":
        logger.info("Logfire tracing is disabled via LOGFIRE_ENABLED=false")
        return False

    # Check if already initialized
    if _LOGFIRE_INITIALIZED:
        logger.info("Logfire tracing already initialized")
        return True

    try:
        import logfire

        # Build configuration
        config_kwargs: Dict[str, Any] = {
            "send_to_logfire": send_to_logfire,
        }

        # Token: parameter takes precedence over env var
        token = token or os.getenv("LOGFIRE_TOKEN")
        if token:
            config_kwargs["token"] = token

        # Environment
        env = environment or os.getenv("ENV", "local")
        config_kwargs["environment"] = env

        # Project name (optional)
        project = project_name or os.getenv("LOGFIRE_PROJECT_NAME")
        if project:
            config_kwargs["project_name"] = project

        logger.debug(
            "Initializing Logfire tracing",
            project=project or "default",
            environment=env,
            send_to_logfire=send_to_logfire,
        )

        # Configure Logfire
        logfire.configure(**config_kwargs)

        logfire.instrument_pydantic_ai()
        logger.info("✅ Instrumented Pydantic AI for Logfire tracing")

        logfire.instrument_litellm()
        logger.info("✅ Instrumented LiteLLM for Logfire tracing")

        _LOGFIRE_INITIALIZED = True

        logger.info(
            "Logfire tracing initialized successfully. View traces at: https://logfire.pydantic.dev"
        )
        return True

    except ImportError as e:
        logger.warning(
            "Logfire tracing not available (missing dependencies). "
            "Install with: pip install logfire",
            error=str(e),
        )
        return False

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
