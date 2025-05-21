import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

def setup_telemetry():
    """
    Configures and enables OpenTelemetry tracing.
    Reads OPENTELEMETRY_ENABLED and OTEL_EXPORTER_OTLP_ENDPOINT from environment variables.
    Telemetry is enabled by default.
    """
    telemetry_enabled = os.getenv("OPENTELEMETRY_ENABLED", "True").lower() != "false"

    if not telemetry_enabled:
        logger.info("OpenTelemetry is disabled via OPENTELEMETRY_ENABLED environment variable.")
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT is not set. Telemetry data will not be exported.")
        # You might choose to still set up the provider without an exporter,
        # or use a default one like ConsoleSpanExporter for local debugging.
        # For now, we'll proceed without an exporter if no endpoint is set.
        resource_provider = TracerProvider()
        trace.set_tracer_provider(resource_provider)
        logger.info("OpenTelemetry TracerProvider configured without a specific exporter due to missing endpoint.")
        return

    try:
        resource_provider = TracerProvider()
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        resource_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(resource_provider)
        logger.info(f"OpenTelemetry configured with OTLP exporter endpoint: {endpoint}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)


def get_tracer(name: str):
    """
    Returns a tracer instance for the given name.
    """
    return trace.get_tracer(name)
