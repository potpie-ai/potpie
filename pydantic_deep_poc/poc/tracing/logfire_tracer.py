"""
Simplified Logfire tracing for pydantic-deep PoC (cannot import Potpie app code).

Mirrors app/modules/intelligence/tracing/logfire_tracer.py patterns:
- OTel detach patch for async tool context switches
- logfire.configure + instrument_pydantic_ai
- Baggage via logfire_trace_metadata
- shutdown + force_flush
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOGFIRE_ATTR_MAX_LEN = 1000
_LOGFIRE_INITIALIZED = False


def _patch_otel_detach_for_async_context() -> None:
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
                pass
            else:
                raise

    contextvars_context.ContextVarsRuntimeContext.detach = _detach_safe  # type: ignore[method-assign]
    logger.debug("Patched OTel context detach for async context switching")


def initialize_logfire_tracing(
    project_name: Optional[str] = None,
    token: Optional[str] = None,
    environment: Optional[str] = None,
    send_to_logfire: bool = True,
    instrument_pydantic_ai: bool = True,
) -> bool:
    global _LOGFIRE_INITIALIZED

    if os.getenv("LOGFIRE_SEND_TO_CLOUD", "true").lower() == "false":
        send_to_logfire = False

    if _LOGFIRE_INITIALIZED:
        logger.info("Logfire tracing already initialized")
        return True

    try:
        import logfire

        config_kwargs: Dict[str, Any] = {}

        token = token or os.getenv("LOGFIRE_TOKEN")
        if token:
            config_kwargs["token"] = token
            config_kwargs["send_to_logfire"] = send_to_logfire
        else:
            config_kwargs["send_to_logfire"] = False

        env = (
            environment
            or os.getenv("LOGFIRE_ENVIRONMENT")
            or os.getenv("ENV", "local")
        )
        config_kwargs["environment"] = env

        project = project_name or os.getenv("LOGFIRE_PROJECT_NAME")
        if project:
            config_kwargs["project_name"] = project

        service_name = os.getenv("LOGFIRE_SERVICE_NAME") or project or "pydantic-deep-poc"
        config_kwargs["service_name"] = service_name

        logfire.configure(**config_kwargs)

        if instrument_pydantic_ai:
            _patch_otel_detach_for_async_context()
            logfire.instrument_pydantic_ai()
            logger.info("Instrumented Pydantic AI for Logfire tracing")

        try:
            logfire.instrument_litellm()
            logger.info("Instrumented LiteLLM for Logfire tracing")
        except Exception as e:
            logger.debug("LiteLLM instrumentation skipped: %s", e)

        _LOGFIRE_INITIALIZED = True
        logger.info("Logfire tracing initialized successfully.")
        return True

    except Exception as e:
        logger.warning("Failed to initialize Logfire tracing (non-fatal): %s", e)
        return False


def is_logfire_enabled() -> bool:
    return _LOGFIRE_INITIALIZED


@contextmanager
def logfire_trace_metadata(**kwargs: Any):
    if not kwargs:
        yield
        return

    if not _LOGFIRE_INITIALIZED:
        yield
        return

    str_attrs: Dict[str, str] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        s = str(value).strip()
        if len(s) > _LOGFIRE_ATTR_MAX_LEN:
            s = s[:_LOGFIRE_ATTR_MAX_LEN]
        str_attrs[key] = s

    if not str_attrs:
        yield
        return

    try:
        import logfire

        with logfire.set_baggage(**str_attrs):
            yield
    except Exception as e:
        logger.debug("Logfire set_baggage failed (non-fatal): %s", e)
        yield


def shutdown_logfire_tracing() -> None:
    global _LOGFIRE_INITIALIZED

    if not _LOGFIRE_INITIALIZED:
        return

    try:
        import logfire

        logfire.force_flush()
        logger.info("Logfire tracing shutdown successfully")
        _LOGFIRE_INITIALIZED = False
    except Exception as e:
        logger.warning("Error shutting down Logfire tracing: %s", e)


def should_instrument_pydantic_ai_runtime() -> bool:
    return os.getenv("LOGFIRE_ENABLED", "true").lower() not in ("false", "0", "no")
