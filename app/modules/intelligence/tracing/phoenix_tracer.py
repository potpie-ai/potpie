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
import logging
from typing import Optional, Any, Dict
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

from langfuse import get_client

langfuse = get_client()

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
        # Compatibility patch: Add missing constants for opentelemetry-semantic-conventions 0.57b0
        # These constants were removed but are still needed by openinference packages
        try:
            from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

            # Check if constants are missing and add them if needed
            # These are runtime additions for compatibility, type checkers may not recognize them
            if not hasattr(gen_ai_attributes, "GEN_AI_INPUT_MESSAGES"):
                setattr(
                    gen_ai_attributes, "GEN_AI_INPUT_MESSAGES", "gen_ai.input.messages"
                )
                logger.debug("Added compatibility constant: GEN_AI_INPUT_MESSAGES")
            if not hasattr(gen_ai_attributes, "GEN_AI_OUTPUT_MESSAGES"):
                setattr(
                    gen_ai_attributes,
                    "GEN_AI_OUTPUT_MESSAGES",
                    "gen_ai.output.messages",
                )
                logger.debug("Added compatibility constant: GEN_AI_OUTPUT_MESSAGES")
            if not hasattr(gen_ai_attributes, "GEN_AI_SYSTEM_INSTRUCTIONS"):
                setattr(
                    gen_ai_attributes,
                    "GEN_AI_SYSTEM_INSTRUCTIONS",
                    "gen_ai.system.instructions",
                )
                logger.debug("Added compatibility constant: GEN_AI_SYSTEM_INSTRUCTIONS")
        except ImportError:
            # If we can't import the module, the openinference imports will fail anyway
            pass

        # Import required modules
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
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

        # Configure timeout (default is 10s, increase to 30s for better resilience)
        # Also configure export timeout separately
        export_timeout = int(os.getenv("OTLP_EXPORT_TIMEOUT", "30"))  # seconds

        # Wrap the exporter with a sanitizing wrapper to filter None values
        # OTLPSpanExporter accepts timeout parameter (in seconds) for connection/read timeouts
        base_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers=headers,
            timeout=export_timeout,  # Connection and read timeout in seconds
        )
        exporter = SanitizingSpanExporter(base_exporter)

        # STEP 3: Add span processors
        # OpenInferenceSpanProcessor for Pydantic AI (adds semantic conventions)
        tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
        # Use BatchSpanProcessor instead of SimpleSpanProcessor for better resilience
        # BatchSpanProcessor batches spans and exports asynchronously, which is better
        # for handling network issues and timeouts without blocking the application
        # Type ignore: SanitizingSpanExporter wraps SpanExporter and implements the interface
        batch_processor = BatchSpanProcessor(
            exporter,  # type: ignore
            max_queue_size=2048,  # Maximum number of spans to queue
            export_timeout_millis=export_timeout
            * 1000,  # Export timeout in milliseconds
            schedule_delay_millis=5000,  # Delay between batch exports (5 seconds)
        )
        tracer_provider.add_span_processor(batch_processor)
        logger.info("✅ Added OpenInference span processor for Pydantic AI tracing")
        logger.info(
            "✅ Added BatchSpanProcessor with timeout=%ds for resilient span export",
            export_timeout,
        )
        logger.info("✅ Added sanitizing exporter wrapper to filter None values")

        # STEP 4: Conditionally instrument LiteLLM (for underlying LLM API calls)
        if auto_instrument:
            litellm_instrumentor = LiteLLMInstrumentor()
            litellm_instrumentor.instrument(tracer_provider=tracer_provider)
            logger.info("✅ Instrumented LiteLLM for Phoenix tracing")
        else:
            logger.debug("Skipped LiteLLM instrumentation: auto_instrument=False")

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


class SanitizingSpanExporter:
    """
    Wrapper around OTLP Span Exporter that sanitizes span data to remove None values
    before they reach the OTLP exporter. This prevents OpenTelemetry encoding errors.

    This wrapper implements the SpanExporter interface and delegates to the underlying
    exporter after sanitizing span attributes.
    """

    def __init__(self, exporter):
        self.exporter = exporter
        self.logger = logging.getLogger(__name__)

    def __getattr__(self, name):
        """Delegate any missing attributes to the underlying exporter."""
        return getattr(self.exporter, name)

    def export(self, spans):
        """Export spans after sanitizing their attributes."""
        if not spans:
            return

        sanitized_spans = []
        for span in spans:
            try:
                # Create a copy of span data to sanitize
                # We need to sanitize the span's internal data structure
                sanitized_span = self._sanitize_span(span)
                if sanitized_span:
                    sanitized_spans.append(sanitized_span)
            except Exception as e:
                # Log but don't fail - tracing errors shouldn't break the application
                span_name = getattr(span, "name", "unknown")
                self.logger.warning(
                    f"Error sanitizing span '{span_name}': {e}. "
                    f"Span will be skipped.",
                    exc_info=True,
                )
                continue

        if sanitized_spans:
            try:
                self.exporter.export(sanitized_spans)
            except Exception as e:
                # Check if this is a timeout error (common with network issues)
                error_msg = str(e).lower()
                error_type = type(e).__name__

                # Timeout errors are common and shouldn't be logged as errors
                # They're expected when the tracing service is slow or unavailable
                if (
                    "timeout" in error_msg
                    or "timed out" in error_msg
                    or "ReadTimeout" in error_type
                    or "ConnectTimeout" in error_type
                ):
                    # Log timeout errors at debug level to reduce noise
                    # These are expected when the tracing service is slow/unavailable
                    self.logger.debug(
                        f"Timeout exporting {len(sanitized_spans)} span(s) to tracing service: {e}. "
                        f"This is expected when the service is slow or unavailable.",
                    )
                # Check if this is an encoding error (None value error)
                elif (
                    "none" in error_msg
                    or "invalid type" in error_msg
                    or "encode" in error_msg
                ):
                    self.logger.warning(
                        f"OpenTelemetry encoding error (likely None values): {e}. "
                        f"{len(sanitized_spans)} span(s) will be lost. "
                        f"This is a known issue with OpenTelemetry and None values in span attributes.",
                        exc_info=True,
                    )
                else:
                    # Other types of export errors (connection errors, etc.)
                    # Log at warning level since they're not critical to application operation
                    self.logger.warning(
                        f"Error exporting {len(sanitized_spans)} span(s) to tracing service: {e}. "
                        f"Spans will be lost but application continues normally.",
                    )

    def _sanitize_span(self, span):
        """
        Sanitize a span by removing None values from its attributes.
        Returns a sanitized span data structure.
        """
        try:
            # Get span data - spans are ReadableSpan objects
            # We need to access the internal _attributes dict
            if hasattr(span, "_attributes") and span._attributes:
                span._attributes = self._sanitize_attributes(span._attributes)

            # Also sanitize resource attributes
            if hasattr(span, "resource") and hasattr(span.resource, "attributes"):
                if span.resource.attributes:
                    span.resource.attributes = self._sanitize_attributes(
                        span.resource.attributes
                    )

            return span
        except Exception as e:
            self.logger.debug(f"Error accessing span internals: {e}")
            # Return span as-is if we can't sanitize it
            return span

    def _sanitize_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize attributes to remove None values and convert
        them to empty strings or remove them entirely.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Sanitized dictionary with None values removed or converted
        """
        if not isinstance(attributes, dict):
            return attributes

        sanitized = {}
        for key, value in attributes.items():
            try:
                if value is None:
                    # Skip None values entirely to avoid encoding errors
                    self.logger.debug(
                        f"Skipping None value for attribute '{key}' in span attributes"
                    )
                    continue
                elif isinstance(value, dict):
                    # Recursively sanitize nested dictionaries
                    sanitized[key] = self._sanitize_attributes(value)
                elif isinstance(value, list):
                    # Sanitize list items
                    sanitized_list = []
                    for item in value:
                        if item is None:
                            # Skip None items in lists
                            self.logger.debug(
                                f"Skipping None item in list for attribute '{key}'"
                            )
                            continue
                        elif isinstance(item, dict):
                            sanitized_list.append(self._sanitize_attributes(item))
                        else:
                            sanitized_list.append(item)
                    sanitized[key] = sanitized_list
                else:
                    sanitized[key] = value
            except Exception as e:
                # Log error but continue processing other attributes
                self.logger.warning(
                    f"Error sanitizing attribute '{key}': {e}. Skipping this attribute.",
                    exc_info=True,
                )
                continue

        return sanitized

    def shutdown(self):
        """Shutdown the exporter."""
        if hasattr(self.exporter, "shutdown"):
            self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000):
        """Force flush the exporter."""
        if hasattr(self.exporter, "force_flush"):
            self.exporter.force_flush(timeout_millis)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


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
        # Flush all span processors to ensure traces are sent
        # Access private attribute safely with getattr
        span_processors = getattr(tracer_provider, "_span_processors", None)
        if span_processors:
            for processor in span_processors:
                if hasattr(processor, "force_flush"):
                    try:
                        processor.force_flush()
                    except Exception as e:
                        logger.debug(f"Error flushing span processor: {e}")
        logger.info("Phoenix tracing shutdown successfully")

        _PHOENIX_INITIALIZED = False

    except Exception as e:
        logger.error("Error shutting down Phoenix tracing: %s", e)
