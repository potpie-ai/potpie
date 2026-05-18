"""logfire tracing/instrumentation — subsumes the current logfire_tracer.py.

CONTRACT: one place owns logfire init + instrumentation. Used by both web and
Celery composition roots.

EDGE CASES / GAPS:
 - Celery prefork: instrument_pydantic_ai MUST be False or async-generator
   tool execution raises OTel 'Token was created in a different Context'.
   The Celery profile sets this; do not regress it.
 - Idempotent: a module-global guard; second call is a no-op (EC2).
 - No token -> send_to_cloud forced False, instrumentation still local-safe;
   emit ONE visible 'logfire local-only: no token' log (audit anti-pattern
   was silent no-op).
 - logfire can ALSO be a log sink (sinks/logfire.py) — tracing init here,
   log export there; both must share one logfire.configure() call.
"""

from __future__ import annotations

from .config import LogfireConfig


def configure_tracing(config: LogfireConfig) -> bool:
    """STUB (Phase 1): contract only. Returns True if initialised."""
    raise NotImplementedError("Phase 1 scaffold — ported in Phase 3")
