"""
Phoenix Tracing Integration for LLM Monitoring

This module sets up Arize Phoenix tracing for monitoring:
- Pydantic AI agent operations (agent runs, delegations, structured outputs)
- LLM API calls (via LiteLLM - Anthropic, OpenAI, etc.)
- Token usage and costs
- Multi-agent delegations (supervisor → subagents)
- Tool calls and results
- Performance metrics and latency

CRITICAL SETUP ORDER:
1. Call initialize_phoenix_tracing() at application startup (in main.py)
2. This registers Phoenix and instruments Pydantic AI BEFORE any agents are created
3. Create agents with instrument=True to enable tracing

Usage:
    from app.modules.intelligence.tracing.phoenix_tracer import initialize_phoenix_tracing

    # Initialize once at application startup (BEFORE creating any agents)
    initialize_phoenix_tracing()

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
from typing import Optional
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# Global flag to track if Phoenix is initialized
_PHOENIX_INITIALIZED = False


def initialize_phoenix_tracing(
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    auto_instrument: bool = True,
) -> bool:
    """
    Initialize Phoenix tracing for the application.

    This should be called once at application startup, ideally in main.py
    before any LLM calls are made.

    Args:
        project_name: Name of the project in Phoenix UI. If None, reads from PHOENIX_PROJECT_NAME env var
                     (default: "potpie-ai" if env var not set). Function parameter takes precedence over env var.
        endpoint: Phoenix collector endpoint. If None, reads from PHOENIX_COLLECTOR_ENDPOINT env var
                 (default: http://localhost:6006 for local Phoenix)
        api_key: Phoenix API key. If None, reads from PHOENIX_API_KEY env var
                (required for Phoenix Cloud, optional for local)
        auto_instrument: Whether to automatically instrument LiteLLM (default: True)

    Returns:
        bool: True if initialization successful, False otherwise

    Environment Variables:
        PHOENIX_ENABLED: Set to "false" to disable Phoenix tracing (default: "true")
        PHOENIX_COLLECTOR_ENDPOINT: Phoenix collector URL (default: http://localhost:6006)
        PHOENIX_API_KEY: API key for Phoenix Cloud (optional for local)
        PHOENIX_PROJECT_NAME: Project name (used only if project_name parameter is None, default: "potpie-ai")
        ENV: Environment identifier (e.g., "development", "production", "staging", "testing") - used as "source" attribute in traces (default: "local")
    """
    global _PHOENIX_INITIALIZED

    # Check if Phoenix is disabled
    if os.getenv("PHOENIX_ENABLED", "true").lower() == "false":
        logger.info("Phoenix tracing is disabled via PHOENIX_ENABLED=false")
        return False

    # Check if already initialized
    if _PHOENIX_INITIALIZED:
        logger.info("Phoenix tracing already initialized")
        return True

    try:
        # Import required modules
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        # Get configuration from environment variables
        endpoint = endpoint or os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
        )
        api_key = api_key or os.getenv("PHOENIX_API_KEY")
        # Function parameter takes precedence over environment variable
        project_name = project_name or os.getenv("PHOENIX_PROJECT_NAME", "potpie-ai")
        # Get the environment/source from ENV variable (defaults to "local" if not set)
        source = os.getenv("ENV", "local")

        logger.info(
            "Initializing Phoenix tracing:\n"
            "  Project: %s\n"
            "  Endpoint: %s\n"
            "  Source: %s\n"
            "  Auto-instrument: %s",
            project_name,
            endpoint,
            source,
            auto_instrument,
        )

        # STEP 1: Create and set up the tracer provider with resource attributes
        resource = Resource.create(
            {
                "service.name": project_name,
                "source": source,
            }
        )
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # STEP 2: Set up OTLP exporter to send traces to Phoenix
        otlp_endpoint = f"{endpoint}/v1/traces"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        exporter = OTLPSpanExporter(endpoint=otlp_endpoint, headers=headers)

        # STEP 3: Add span processors
        # OpenInferenceSpanProcessor for Pydantic AI (adds semantic conventions)
        tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
        # SimpleSpanProcessor to export spans to Phoenix
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
        logger.info("✅ Added OpenInference span processor for Pydantic AI tracing")

        # STEP 4: Instrument LiteLLM (for underlying LLM API calls)
        litellm_instrumentor = LiteLLMInstrumentor()
        litellm_instrumentor.instrument(tracer_provider=tracer_provider)
        logger.info("✅ Instrumented LiteLLM for Phoenix tracing")

        _PHOENIX_INITIALIZED = True

        logger.info(
            "✅ Phoenix tracing initialized successfully!\n" "   View traces at: %s",
            endpoint,
        )

        return True

    except ImportError as e:
        logger.warning(
            "Phoenix tracing not available (missing dependencies): %s\n"
            "Install with: pip install arize-phoenix arize-phoenix-otel openinference-instrumentation-pydantic-ai openinference-instrumentation-litellm",
            e,
        )
        return False

    except Exception as e:
        logger.error("Failed to initialize Phoenix tracing: %s", e, exc_info=True)
        return False


def get_tracer(name: str = __name__):
    """
    Get an OpenTelemetry tracer for manual span creation.

    Args:
        name: Name for the tracer (usually __name__ of the module)

    Returns:
        Tracer instance or None if Phoenix not initialized

    Example:
        tracer = get_tracer(__name__)

        def my_function(input: str) -> str:
            return process(input)
    """
    try:
        from opentelemetry import trace

        if not _PHOENIX_INITIALIZED:
            logger.debug("Phoenix not initialized, returning default tracer")

        return trace.get_tracer(name)

    except ImportError:
        logger.debug("OpenTelemetry not available")
        return None


def is_phoenix_enabled() -> bool:
    """Check if Phoenix tracing is enabled and initialized."""
    return _PHOENIX_INITIALIZED


def shutdown_phoenix_tracing():
    """
    Shutdown Phoenix tracing and flush any pending traces.

    This should be called on application shutdown to ensure all traces are sent.
    """
    global _PHOENIX_INITIALIZED

    if not _PHOENIX_INITIALIZED:
        return

    try:
        from opentelemetry import trace

        # Get the tracer provider and shutdown
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()
            logger.info("Phoenix tracing shutdown successfully")

        _PHOENIX_INITIALIZED = False

    except Exception as e:
        logger.error("Error shutting down Phoenix tracing: %s", e)