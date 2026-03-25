"""Initialize / shutdown Logfire once per benchmark process."""

from __future__ import annotations

from poc.config.settings import (
    LOGFIRE_ENVIRONMENT,
    LOGFIRE_PROJECT_NAME,
    LOGFIRE_SEND_TO_CLOUD,
    LOGFIRE_TOKEN,
)
from poc.tracing.logfire_tracer import (
    initialize_logfire_tracing,
    shutdown_logfire_tracing,
)


def init_benchmark_tracing() -> bool:
    return initialize_logfire_tracing(
        project_name=LOGFIRE_PROJECT_NAME or None,
        token=LOGFIRE_TOKEN or None,
        environment=LOGFIRE_ENVIRONMENT,
        send_to_logfire=LOGFIRE_SEND_TO_CLOUD,
        instrument_pydantic_ai=True,
    )


def shutdown_benchmark_tracing() -> None:
    shutdown_logfire_tracing()
