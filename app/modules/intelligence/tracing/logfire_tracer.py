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
from typing import Any, Dict, List, Optional

import logfire

from app.modules.utils.logger import setup_logger

# Max length for baggage/attribute values (Logfire truncates longer strings)
_LOGFIRE_ATTR_MAX_LEN = 1000

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
                # Safe to ignore when context switched across async boundary
                pass
            else:
                raise

    contextvars_context.ContextVarsRuntimeContext.detach = _detach_safe  # type: ignore[method-assign]
    logger.debug("Patched OTel context detach for async context switching")


# ---------------------------------------------------------------------------
# Langfuse native SDK integration (v4+)
# ---------------------------------------------------------------------------

_langfuse_enabled = False


def _get_langfuse_host() -> str:
    """Return the Langfuse host from supported env var names."""
    return (
        (
            os.getenv("LANGFUSE_HOST")
            or os.getenv("LANGFUSE_API_BASE_URL")
            or os.getenv("LANGFUSE_BASE_URL")
            or ""
        )
        .strip()
        .rstrip("/")
    )


def _setup_langfuse_otel() -> None:
    """Validate Langfuse credentials and normalize SDK configuration.

    Requires env vars:
    - LANGFUSE_HOST (or legacy LANGFUSE_BASE_URL/LANGFUSE_API_BASE_URL)
    - LANGFUSE_PUBLIC_KEY: pk-lf-...
    - LANGFUSE_SECRET_KEY: sk-lf-...
    """
    global _langfuse_enabled

    host = _get_langfuse_host()
    if not host:
        return

    # The Langfuse Python SDK reads LANGFUSE_HOST. Keep backward compatibility
    # with the repo's older LANGFUSE_BASE_URL/LANGFUSE_API_BASE_URL variables.
    os.environ.setdefault("LANGFUSE_HOST", host)

    try:
        import httpx

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            logger.warning("Langfuse host configured but API keys are missing")
            return

        auth = httpx.BasicAuth(username=public_key, password=secret_key)

        # Validate credentials with a lightweight API call
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                f"{host}/api/public/traces",
                params={"limit": 1},
                auth=auth,
            )
            if resp.status_code == 200:
                _langfuse_enabled = True
                logger.info("Langfuse REST API validated (host={})", host)
            else:
                logger.warning(
                    "Langfuse auth check failed (status={})", resp.status_code
                )
    except Exception as e:
        logger.warning("Langfuse setup failed (non-fatal): {}", e)


def _add_langfuse_otel_exporter() -> None:
    """Register Langfuse as a secondary OTEL span exporter.

    After logfire.configure() owns the TracerProvider, we inject a
    BatchSpanProcessor that ships spans to Langfuse's OTLP endpoint.
    This makes all Pydantic AI / LiteLLM spans visible in Langfuse.
    """
    langfuse_host = _get_langfuse_host()
    if not langfuse_host:
        return

    os.environ.setdefault("LANGFUSE_HOST", langfuse_host)

    try:
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry import trace as otel_trace

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not public_key or not secret_key:
            logger.warning("Langfuse host configured but API keys are missing")
            return

        endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
        import base64

        auth_token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Basic {auth_token}"},
        )

        provider = otel_trace.get_tracer_provider()
        # Logfire wraps the real SDK provider; try .provider, then ._provider
        real_provider = getattr(provider, "provider", None) or getattr(
            provider, "_provider", provider
        )
        if isinstance(real_provider, SdkTracerProvider):
            real_provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info("Added Langfuse OTLP span exporter (endpoint={})", endpoint)
        elif hasattr(provider, "add_span_processor"):
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(
                "Added Langfuse OTLP span exporter via proxy (endpoint={})", endpoint
            )
        else:
            logger.warning(
                "TracerProvider is not SDK type ({}), cannot add Langfuse exporter",
                type(real_provider),
            )
    except Exception as e:
        logger.warning("Failed to add Langfuse OTEL exporter (non-fatal): {}", e)


def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracing is enabled and authenticated."""
    return _langfuse_enabled


@contextmanager
def langfuse_trace_context(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    input: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
):
    """Wrap an agent run in a Langfuse root observation.

    The previous implementation queried Langfuse for the latest trace after the
    run and patched it via ingestion, which is race-prone when multiple agent
    runs finish close together. This creates the trace deterministically up front
    and lets Pydantic AI/LiteLLM spans attach to the active OpenTelemetry context.
    """
    if not _langfuse_enabled:
        yield None
        return

    metadata = metadata or {}
    tags = tags or []
    trace_data: Dict[str, Any] = {
        "name": name,
        "user_id": user_id,
        "session_id": session_id,
        "input": input,
        "metadata": metadata,
        "tags": tags,
        "output": None,
    }

    try:
        from langfuse import get_client, propagate_attributes

        langfuse = get_client()
        trace_seed_parts = [
            str(metadata.get("run_id") or ""),
            str(session_id or ""),
            str(user_id or ""),
            name,
        ]
        trace_seed = ":".join(part for part in trace_seed_parts if part)
        trace_id = langfuse.create_trace_id(seed=trace_seed or None)
        trace_data["trace_id"] = trace_id
    except Exception as e:
        logger.warning(
            "Langfuse trace setup failed; continuing without Langfuse: {}", e
        )
        yield trace_data
        return

    # Langfuse propagate_attributes currently types metadata as str values.
    propagated_metadata = {
        key: str(value)[:_LOGFIRE_ATTR_MAX_LEN]
        for key, value in metadata.items()
        if value is not None
    }

    yielded = False
    body_failed = False
    try:
        with langfuse.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type="agent",
            input=input,
            metadata=metadata,
        ) as root_observation:
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id,
                metadata=propagated_metadata or None,
                tags=tags or None,
                trace_name=name,
            ):
                try:
                    yielded = True
                    yield trace_data
                except BaseException:
                    body_failed = True
                    raise
                finally:
                    output = trace_data.get("output")
                    try:
                        if output is not None:
                            root_observation.update(output=output)
                        langfuse.set_current_trace_io(input=input, output=output)
                    except Exception as e:
                        logger.debug(
                            "Langfuse trace output update failed (non-fatal): {}", e
                        )

        logger.info(
            "Recorded Langfuse trace {} (name={}, user={}, session={})",
            trace_id,
            name,
            user_id,
            session_id,
        )
    except Exception as e:
        if body_failed:
            raise
        if yielded:
            logger.warning("Langfuse trace finalization failed (non-fatal): {}", e)
            return
        logger.warning(
            "Langfuse root observation failed; continuing without Langfuse: {}", e
        )
        yield trace_data
    finally:
        # Flush Logfire/OTEL after the Langfuse root observation has ended so
        # child Pydantic AI/LiteLLM spans are exported promptly.
        try:
            logfire.force_flush()
        except Exception:
            pass
        try:
            langfuse.flush()
        except Exception as e:
            logger.debug("Langfuse flush failed (non-fatal): {}", e)


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
            environment or os.getenv("LOGFIRE_ENVIRONMENT") or os.getenv("ENV", "local")
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

        # Configure Langfuse OTEL tracing alongside Logfire (if env vars set)
        _setup_langfuse_otel()

        # Add Langfuse as a secondary OTEL span exporter
        _add_langfuse_otel_exporter()

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

    This must stay enabled globally so that Pydantic AI emits agent_run spans
    for analytics. Use LOGFIRE_ENABLED=false to disable tracing entirely.
    """
    enabled = os.getenv("LOGFIRE_ENABLED", "true").lower()
    return enabled not in ("false", "0", "no")


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
    if not kwargs or not _LOGFIRE_INITIALIZED:
        # No metadata or Logfire not initialized – no-op
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

        baggage_ctx = logfire.set_baggage(**str_attrs)
    except Exception as e:
        logger.debug(
            "Logfire set_baggage failed (non-fatal)",
            error=str(e),
        )
        yield
        return

    with baggage_ctx:
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
        attrs["user_id"] = str(user_id).strip()[:_LOGFIRE_ATTR_MAX_LEN]
    env = (
        environment or os.getenv("LOGFIRE_ENVIRONMENT") or os.getenv("ENV") or "local"
    ).strip()
    attrs["environment"] = env[:_LOGFIRE_ATTR_MAX_LEN]
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

        baggage_ctx = logfire.set_baggage(**attrs)
        span_attrs = {"environment": attrs["environment"]}
        if attrs.get("user_id"):
            span_attrs["user_id"] = attrs["user_id"]
        for k, v in attrs.items():
            if k not in ("user_id", "environment") and v:
                span_attrs[k] = v
        span_ctx = logfire.span("llm_call", **span_attrs)
    except Exception as e:
        logger.debug(
            "Logfire LLM metadata failed (non-fatal)",
            error=str(e),
        )
        yield
        return

    with baggage_ctx:
        with span_ctx:
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
