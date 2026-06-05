"""Process-wide correlation context (the propagation spine).

A single :class:`contextvars.ContextVar` carrying the identifiers that flow
through the pipeline: ``trace_id``, ``pot_id``, ``event_id``, ``batch_id``,
``run_id``, ``seq``, ``chunk``. The logging setup injects these into every
log record; Phase B binds them at ingress and re-binds them inside the
batch worker so a log line / span can always be tied back to an event.

Dependency-free by design (stdlib ``contextvars`` only) — the same
precedent the MCP auth context uses. Keys are intentionally a fixed,
documented set so log/trace attribute names stay stable across the codebase.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator, Mapping

# The canonical correlation keys. Anything outside this set is dropped so
# log/trace cardinality stays bounded and attribute names stay stable.
CORRELATION_KEYS = (
    "trace_id",
    "pot_id",
    "event_id",
    "batch_id",
    "run_id",
    "seq",
    "chunk",
    "source",
)

_correlation: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "context_engine_correlation", default={}
)


def get_correlation() -> dict[str, Any]:
    """Return a shallow copy of the current correlation mapping."""
    return dict(_correlation.get())


def _clean(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in values.items()
        if k in CORRELATION_KEYS and v is not None and v != ""
    }


def bind_correlation(**values: Any) -> contextvars.Token:
    """Merge ``values`` into the current correlation context.

    Returns the reset token so callers that own the scope can restore it.
    Prefer :func:`correlation_scope` for block-scoped binding.
    """
    merged = dict(_correlation.get())
    merged.update(_clean(values))
    return _correlation.set(merged)


def reset_correlation(token: contextvars.Token) -> None:
    with contextlib.suppress(Exception):
        _correlation.reset(token)


@contextlib.contextmanager
def correlation_scope(**values: Any) -> Iterator[dict[str, Any]]:
    """Bind ``values`` for the duration of the block, then restore.

    Safe to nest; inner scopes see the merge of all enclosing scopes.
    """
    token = bind_correlation(**values)
    try:
        yield get_correlation()
    finally:
        reset_correlation(token)
