"""logfire tracing/instrumentation — subsumes the current logfire_tracer.py.

CONTRACT: one place owns logfire init + instrumentation. Used by both web and
Celery composition roots and shared with sinks/logfire_sink.py (so we never
configure logfire twice — EC2).

EDGE CASES:
 - Celery prefork: callers must pass instrument_pydantic_ai=False (handled by
   profiles.celery()) to avoid OTel 'Token was created in a different
   Context' on async-generator tool execution.
 - No token -> send_to_cloud forced False; local-only is still safe. We emit a
   single visible 'logfire local-only: no token' notice on the package logger
   instead of the audit's silent no-op.
 - logfire library missing -> return False (sink will skip).
"""

from __future__ import annotations

import logging
import os

from .config import ObservabilityConfig

_logger = logging.getLogger(__name__)
_state = {"initialised": False, "pid": None}


def configure_tracing(cfg: ObservabilityConfig) -> bool:
    """Initialise logfire once per process. Returns True if initialised."""
    pid = os.getpid()
    if _state["initialised"] and _state["pid"] == pid:
        return True
    try:
        import logfire
    except ModuleNotFoundError:
        _logger.warning("logfire not installed; tracing disabled")
        return False

    lf = cfg.logfire
    send = bool(lf.token) and lf.send_to_cloud
    if not lf.token:
        _logger.info("logfire local-only: no LOGFIRE_TOKEN set")

    try:
        logfire.configure(
            token=lf.token if send else None,
            send_to_logfire=send,
            environment=cfg.env,
            service_name=cfg.service_name,
        )
    except Exception as exc:  # logfire init failures must NOT crash the app
        _logger.warning("logfire.configure failed: %s", exc)
        return False

    if lf.instrument_litellm:
        try:
            logfire.instrument_litellm()
        except Exception as exc:
            _logger.debug("instrument_litellm skipped: %s", exc)
    if lf.instrument_pydantic_ai:
        try:
            logfire.instrument_pydantic_ai()
        except Exception as exc:
            _logger.debug("instrument_pydantic_ai skipped: %s", exc)

    _state["initialised"] = True
    _state["pid"] = pid
    return True
